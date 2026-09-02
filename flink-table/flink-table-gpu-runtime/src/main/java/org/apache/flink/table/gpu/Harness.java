/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.flink.table.gpu;

import org.apache.flink.core.memory.MemorySegmentFactory;
import org.apache.flink.table.data.GenericRowData;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.data.binary.BinaryRowData;
import org.apache.flink.table.data.columnar.ColumnarRowData;
import org.apache.flink.table.data.columnar.vector.ColumnVector;
import org.apache.flink.table.data.columnar.vector.VectorizedColumnBatch;
import org.apache.flink.table.data.columnar.vector.heap.HeapDoubleVector;
import org.apache.flink.table.data.columnar.vector.heap.HeapLongVector;
import org.apache.flink.table.gpu.gather.RowGather;
import org.apache.flink.table.gpu.metrics.OffloadMetrics;
import org.apache.flink.table.gpu.operator.FilterProjectEngine;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Standalone harness for the P1 target query, without the planner:
 *
 * <pre>SELECT id, val * 2.0 + 1.0 AS scaled FROM t WHERE val &gt; 0.5</pre>
 *
 * <p>Produces the gather / copy-in / kernel / copy-out breakdown that is P1's exit criterion, per
 * input tier, against a scalar CPU baseline standing in for Flink's generated operator.
 *
 * <p>Usage: {@code Harness [rows] [batchSize] [intensity...]} — one run per intensity, where 0 is
 * the plain two-flop projection and higher values add arithmetic rounds per element.
 */
public final class Harness {

    private static final double MUL = 2.0;
    private static final double ADD = 1.0;
    private static final double THRESHOLD = 0.5;

    /** Matches {@code VectorizedColumnBatch.DEFAULT_SIZE}, i.e. what Flink's readers produce. */
    private static final int COLUMN_BATCH_ROWS = VectorizedColumnBatch.DEFAULT_SIZE;

    public static void main(String[] args) {
        int rows = args.length > 0 ? Integer.parseInt(args[0]) : 8_000_000;
        int batchSize = args.length > 1 ? Integer.parseInt(args[1]) : 262_144;
        int[] intensities = args.length > 2 ? parseIntensities(args) : new int[] {0};

        System.out.printf(
                "rows=%,d  batch=%,d  query: SELECT id, val*%.1f+%.1f WHERE val > %.1f%n",
                rows, batchSize, MUL, ADD, THRESHOLD);

        List<RowData> binary = generateBinary(rows);
        List<RowData> generic = generateGeneric(rows);
        List<RowData> columnar = generateColumnar(rows);

        for (int intensity : intensities) {
            System.out.printf("%n############ intensity=%d ############%n", intensity);
            double[] expected = new double[rows];
            boolean[] expectedMask = new boolean[rows];
            long cpuNanos = cpuBaseline(binary, expected, expectedMask, intensity);
            System.out.printf("CPU baseline (scalar, single-threaded): %.3f ms%n", cpuNanos / 1e6);

            run("tier1-columnar", columnar, batchSize, expected, expectedMask, intensity, true);
            run("tier2-binary", binary, batchSize, expected, expectedMask, intensity, false);
            run("tier4-generic", generic, batchSize, expected, expectedMask, intensity, false);
        }
    }

    private static int[] parseIntensities(String[] args) {
        int[] out = new int[args.length - 2];
        for (int i = 2; i < args.length; i++) {
            out[i - 2] = Integer.parseInt(args[i]);
        }
        return out;
    }

    private static double project(double v, int intensity) {
        double acc = v;
        for (int k = 0; k < intensity; k++) {
            acc = Math.sqrt(acc * acc + 1.0) * 0.5 + acc * 0.25;
        }
        return acc * MUL + ADD;
    }

    /** Stands in for Flink's generated scalar operator: the loop the offload must beat. */
    private static long cpuBaseline(
            List<RowData> rows, double[] outValues, boolean[] outMask, int intensity) {
        int n = rows.size();
        double[] sink = new double[n];
        long t0 = System.nanoTime();
        int emitted = 0;
        for (int i = 0; i < n; i++) {
            double v = rows.get(i).getDouble(1);
            if (v > THRESHOLD) {
                sink[emitted++] = project(v, intensity);
            }
        }
        long elapsed = System.nanoTime() - t0;
        for (int i = 0; i < n; i++) {
            double v = rows.get(i).getDouble(1);
            outValues[i] = project(v, intensity);
            outMask[i] = v > THRESHOLD;
        }
        if (emitted < 0) {
            throw new AssertionError("unreachable; keeps the loop from being optimised away");
        }
        return elapsed;
    }

    private static void run(
            String label,
            List<RowData> rows,
            int batchSize,
            double[] expected,
            boolean[] expectedMask,
            int intensity,
            boolean bulk) {

        try (FilterProjectEngine engine =
                new FilterProjectEngine(batchSize, MUL, ADD, THRESHOLD, true, intensity)) {
            engine.open();

            RowGather gather =
                    RowGather.forDouble(
                            rows.get(0),
                            1,
                            engine.inputColumn(),
                            bulk ? engine.inputSegment() : null);

            warmUp(engine, gather, rows, batchSize);

            int mismatches = 0;
            double maxRelError = 0.0;
            long emittedTotal = 0;
            double[] sink = new double[batchSize];

            for (int start = 0; start < rows.size(); start += batchSize) {
                int count = Math.min(batchSize, rows.size() - start);

                long g0 = System.nanoTime();
                int i = 0;
                while (i < count) {
                    int taken = gather.acceptBulk(rows, start + i, start + count, i);
                    if (taken > 0) {
                        i += taken;
                    } else {
                        gather.accept(rows.get(start + i), i);
                        i++;
                    }
                }
                long gatherNanos = System.nanoTime() - g0;

                FilterProjectEngine.Execution execution = engine.execute();

                // Drain: compaction plus building the output value, i.e. what the operator hands to
                // output.collect(). Validation is deliberately outside this loop.
                long dr0 = System.nanoTime();
                int emitted = 0;
                for (int j = 0; j < count; j++) {
                    if (engine.selected(j)) {
                        sink[emitted] = engine.projected(j);
                        emitted++;
                    }
                }
                long drainNanos = System.nanoTime() - dr0;
                emittedTotal += emitted;

                engine.recordBatch(count, gatherNanos, execution, emitted, drainNanos);

                int seen = 0;
                for (int j = 0; j < count; j++) {
                    boolean want = expectedMask[start + j];
                    if (engine.selected(j) != want) {
                        mismatches++;
                    } else if (want) {
                        double want1 = expected[start + j];
                        double got = sink[seen++];
                        double rel = want1 == 0.0 ? Math.abs(got) : Math.abs((got - want1) / want1);
                        maxRelError = Math.max(maxRelError, rel);
                        if (rel > 1e-12) {
                            mismatches++;
                        }
                    }
                }
            }

            OffloadMetrics m = engine.metrics();
            System.out.print(m.report(label + "  (" + gather.tier() + ")"));
            System.out.printf(
                    "emitted=%,d  max_rel_err=%.3g  mismatches=%d  %s%n",
                    emittedTotal,
                    maxRelError,
                    mismatches,
                    mismatches == 0 ? "RESULTS MATCH CPU" : "*** WRONG ***");
        }
    }

    private static void warmUp(
            FilterProjectEngine engine, RowGather gather, List<RowData> rows, int batchSize) {
        int count = Math.min(batchSize, rows.size());
        int i = 0;
        while (i < count) {
            int taken = gather.acceptBulk(rows, i, count, i);
            if (taken > 0) {
                i += taken;
            } else {
                gather.accept(rows.get(i), i);
                i++;
            }
        }
        engine.executeUntimed();
    }

    /**
     * {@link BinaryRowData} rows pointing into <b>one shared backing array</b>, at the fixed
     * 24-byte stride the format implies for two fields (8-byte null bitset plus two 8-byte slots).
     *
     * <p>Allocating a separate {@code byte[]} per row — the obvious way to write this — makes every
     * row a distinct cache line reached through a four-deep pointer chase, which is not how Flink
     * delivers them. Post-shuffle rows point into shared network buffers or sorter pages, so
     * consecutive rows are contiguous. An earlier version of this harness got that wrong and made
     * tier 2 look slower than tier 4.
     */
    private static List<RowData> generateBinary(int rows) {
        Random random = new Random(42);
        int nullBits = BinaryRowData.calculateBitSetWidthInBytes(2);
        int rowBytes = BinaryRowData.calculateFixPartSizeInBytes(2);
        org.apache.flink.core.memory.MemorySegment segment =
                MemorySegmentFactory.wrap(new byte[rows * rowBytes]);

        List<RowData> out = new ArrayList<>(rows);
        for (int i = 0; i < rows; i++) {
            int base = i * rowBytes;
            // Written directly rather than through BinaryRowWriter, which allocates a fresh
            // byte[] per row in its constructor and re-points the row at it — exactly the
            // per-row allocation this generator exists to avoid. The layout is the one
            // BinaryRowData.getFieldOffset reads: bitset, then one 8-byte slot per field. A
            // zeroed bitset means RowKind.INSERT and no nulls.
            segment.putLong(base + nullBits, i);
            segment.putDouble(base + nullBits + 8, random.nextDouble());

            BinaryRowData row = new BinaryRowData(2);
            row.pointTo(segment, base, rowBytes);
            out.add(row);
        }
        return out;
    }

    private static List<RowData> generateGeneric(int rows) {
        Random random = new Random(42);
        List<RowData> out = new ArrayList<>(rows);
        for (int i = 0; i < rows; i++) {
            out.add(GenericRowData.of((long) i, random.nextDouble()));
        }
        return out;
    }

    /**
     * {@link ColumnarRowData} cursors over real {@link VectorizedColumnBatch}es of {@link
     * VectorizedColumnBatch#DEFAULT_SIZE} rows — the shape a vectorized Parquet reader produces.
     * Columns are plain {@link HeapDoubleVector}/{@link HeapLongVector} with no dictionary and no
     * nulls, which is the only case the bulk gather accepts.
     */
    private static List<RowData> generateColumnar(int rows) {
        Random random = new Random(42);
        List<RowData> out = new ArrayList<>(rows);
        for (int base = 0; base < rows; base += COLUMN_BATCH_ROWS) {
            int n = Math.min(COLUMN_BATCH_ROWS, rows - base);
            HeapLongVector ids = new HeapLongVector(n);
            HeapDoubleVector vals = new HeapDoubleVector(n);
            for (int i = 0; i < n; i++) {
                ids.vector[i] = base + i;
                vals.vector[i] = random.nextDouble();
            }
            VectorizedColumnBatch batch = new VectorizedColumnBatch(new ColumnVector[] {ids, vals});
            batch.setNumRows(n);
            for (int i = 0; i < n; i++) {
                out.add(new ColumnarRowData(batch, i));
            }
        }
        return out;
    }
}
