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

package org.apache.flink.table.gpu.operator;

import org.apache.flink.table.gpu.gather.RowGather;
import org.apache.flink.table.gpu.kernels.FilterProjectKernels;
import org.apache.flink.table.gpu.metrics.OffloadMetrics;

import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;
import uk.ac.manchester.tornado.api.TornadoProfilerResult;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.ProfilerMode;
import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

import java.lang.foreign.MemorySegment;

/**
 * Runs one filter+projection over a batch of staged rows on the device.
 *
 * <p>Lifetime: one instance per operator instance. The {@link TornadoExecutionPlan} is built once
 * in {@link #open()} and re-executed per batch, so TornadoVM's JIT compilation is paid once per
 * task rather than once per batch. That is the same reason a Flink generated operator is compiled
 * at task startup and not per record.
 *
 * <p><b>Buffers are allocated once and reused.</b> Batches are fixed-size; the final partial batch
 * is handled by running the kernel over the whole buffer and having the caller drain only the first
 * {@code count} positions. Reallocating per batch would put an allocation in the hot path and
 * defeat {@code FIRST_EXECUTION} transfer modes later.
 *
 * <p><b>Transfer modes are deliberately EVERY_EXECUTION here.</b> This is a single-kernel task
 * graph, so there is no intermediate to keep resident and nothing to gain from {@code
 * persistOnDevice}. Residency arrives in P2, when a reduction consumes this kernel's output without
 * a host round trip; the copy-out measured here is exactly what P2 is meant to remove.
 */
public final class FilterProjectEngine implements AutoCloseable {

    private final int batchSize;
    private final double mul;
    private final double add;
    private final double threshold;
    private final boolean profile;
    private final int intensity;

    private DoubleArray val;
    private DoubleArray out;
    private IntArray mask;

    private TornadoExecutionPlan plan;
    private final OffloadMetrics metrics = new OffloadMetrics();

    public FilterProjectEngine(
            int batchSize, double mul, double add, double threshold, boolean profile) {
        this(batchSize, mul, add, threshold, profile, 0);
    }

    /**
     * @param intensity extra arithmetic rounds per element; 0 selects the plain 2-flop kernel. Used
     *     to test whether device work can become a meaningful share at all.
     */
    public FilterProjectEngine(
            int batchSize,
            double mul,
            double add,
            double threshold,
            boolean profile,
            int intensity) {
        if (batchSize <= 0) {
            throw new IllegalArgumentException("batchSize must be positive, was " + batchSize);
        }
        this.batchSize = batchSize;
        this.mul = mul;
        this.add = add;
        this.threshold = threshold;
        this.profile = profile;
        this.intensity = intensity;
    }

    public void open() {
        val = new DoubleArray(batchSize);
        out = new DoubleArray(batchSize);
        mask = new IntArray(batchSize);
        // TornadoVM native arrays contain garbage on allocation, unlike Java arrays. The tail of a
        // partial batch is never read by the caller, but leaving it undefined would make the
        // device results non-reproducible across runs.
        val.init(0.0);
        out.init(0.0);
        mask.init(0);

        TaskGraph graph =
                new TaskGraph("calc").transferToDevice(DataTransferMode.EVERY_EXECUTION, val);
        if (intensity == 0) {
            graph =
                    graph.task(
                            "filterProject",
                            FilterProjectKernels::scaleAndSelect,
                            val,
                            out,
                            mask,
                            mul,
                            add,
                            threshold);
        } else {
            graph =
                    graph.task(
                            "filterProject",
                            FilterProjectKernels::heavyScaleAndSelect,
                            val,
                            out,
                            mask,
                            mul,
                            add,
                            threshold,
                            intensity);
        }
        graph = graph.transferToHost(DataTransferMode.EVERY_EXECUTION, out, mask);

        ImmutableTaskGraph snapshot = graph.snapshot();
        plan = new TornadoExecutionPlan(snapshot);
        if (profile) {
            plan = plan.withProfiler(ProfilerMode.SILENT);
        }
    }

    /** Off-heap staging buffer for the input column, for gathers that can bulk-copy into it. */
    public MemorySegment inputSegment() {
        return val.getSegment();
    }

    public int intensity() {
        return intensity;
    }

    /** Write side of the staging column, handed to the gather. */
    public RowGather.DoubleColumn inputColumn() {
        return (position, value) -> val.set(position, value);
    }

    /** Projected value for row {@code i} of the last executed batch. */
    public double projected(int i) {
        return out.get(i);
    }

    /** Whether row {@code i} of the last executed batch passed the filter. */
    public boolean selected(int i) {
        return mask.get(i) != 0;
    }

    public int batchSize() {
        return batchSize;
    }

    public OffloadMetrics metrics() {
        return metrics;
    }

    /**
     * Executes the kernel over the whole staging buffer.
     *
     * <p>Returns the wall time and profiler result rather than recording them, because the drain
     * has not happened yet and a batch must be recorded exactly once with all four segments. See
     * {@link #recordBatch}.
     */
    public Execution execute() {
        long t0 = System.nanoTime();
        TornadoExecutionResult result = plan.execute();
        long wall = System.nanoTime() - t0;
        return new Execution(wall, profile ? result.getProfilerResult() : null);
    }

    /** One batch's device timings, pending the caller's gather and drain numbers. */
    public record Execution(long wallNanos, TornadoProfilerResult profilerResult) {}

    /** Records one complete batch: staged, executed, drained. */
    public void recordBatch(
            int count, long gatherNanos, Execution execution, int emitted, long drainNanos) {
        metrics.recordBatch(
                count,
                emitted,
                gatherNanos,
                execution.wallNanos(),
                drainNanos,
                execution.profilerResult());
    }

    /** Executes without recording, for warm-up iterations that would otherwise skew the report. */
    public void executeUntimed() {
        plan.execute();
    }

    @Override
    public void close() {
        if (plan != null) {
            try {
                plan.close();
            } catch (Exception e) {
                throw new IllegalStateException("failed to release TornadoVM execution plan", e);
            }
            plan = null;
        }
    }
}
