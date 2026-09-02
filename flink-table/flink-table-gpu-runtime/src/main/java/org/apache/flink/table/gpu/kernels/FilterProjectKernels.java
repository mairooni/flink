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
package org.apache.flink.table.gpu.kernels;

import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Query-independent filter + projection kernels.
 *
 * <p>These are ordinary static methods compiled by javac. That is not incidental: TornadoVM
 * resolves a kernel from the {@code SerializedLambda} of the method reference passed to {@code
 * TaskGraph.task(...)}, which requires a {@code writeReplace()} that only javac emits. Kernels
 * generated at runtime (e.g. by Janino, as Flink does for its own operators) cannot be used here.
 *
 * <p><b>Filtering does not compact.</b> Each kernel evaluates the projection for every input row
 * and writes a 0/1 selection mask alongside it. Compaction happens on the host during the drain
 * loop, which already walks every row. Doing it on device would need a prefix sum plus a scatter,
 * which is more transfer and more kernels for no benefit at this stage.
 *
 * <p>Null handling is deliberately absent: the planner-side predicate only admits inputs where the
 * gather has established that the column has no nulls, or has already materialised them away. See
 * {@code RowGather}.
 */
public final class FilterProjectKernels {

    private FilterProjectKernels() {}

    /**
     * Evaluates {@code out[i] = val[i] * mul + add} for every row, and {@code mask[i] = 1} where
     * {@code val[i] > threshold}.
     *
     * <p>This covers the P1 target query {@code SELECT id, val * 2.0 + 1.0 WHERE val > 0.5}. The
     * {@code id} column is never transferred: it is not read by the kernel, so it stays on the host
     * and is emitted directly from the staging buffer during the drain.
     */
    public static void scaleAndSelect(
            DoubleArray val, DoubleArray out, IntArray mask,
            double mul, double add, double threshold) {
        for (@Parallel int i = 0; i < val.getSize(); i++) {
            double v = val.get(i);
            out.set(i, v * mul + add);
            // Written as if/else rather than a ternary: both compile, but the explicit form keeps
            // the generated OpenCL readable when dumping kernels with -Dtornado.print.kernel=True.
            if (v > threshold) {
                mask.set(i, 1);
            } else {
                mask.set(i, 0);
            }
        }
    }

    /** Projection only, no selection. Every row survives. */
    public static void scale(DoubleArray val, DoubleArray out, double mul, double add) {
        for (@Parallel int i = 0; i < val.getSize(); i++) {
            out.set(i, val.get(i) * mul + add);
        }
    }

    /** Selection only, no projection. */
    public static void select(DoubleArray val, IntArray mask, double threshold) {
        for (@Parallel int i = 0; i < val.getSize(); i++) {
            if (val.get(i) > threshold) {
                mask.set(i, 1);
            } else {
                mask.set(i, 0);
            }
        }
    }

    /**
     * Host-side reference implementation of {@link #scaleAndSelect}, used to verify device results
     * in tests and to serve as the CPU arm of the benchmark. Must stay semantically identical.
     */
    public static void scaleAndSelectReference(
            DoubleArray val, DoubleArray out, IntArray mask,
            double mul, double add, double threshold) {
        for (int i = 0; i < val.getSize(); i++) {
            double v = val.get(i);
            out.set(i, v * mul + add);
            mask.set(i, v > threshold ? 1 : 0);
        }
    }

    /**
     * Same selection as {@link #scaleAndSelect}, with {@code intensity} extra rounds of arithmetic
     * per element. Used to answer whether device work can ever become a meaningful share of the
     * offload: at intensity 0 the kernel is 2 flops and measured 0.4% of the breakdown.
     *
     * <p>The inner expression uses only multiply, add and {@code sqrt}, all of which are
     * exactly-rounded in IEEE 754, so host and device should agree closely. Transcendentals
     * (OpenCL {@code native_sin} versus {@code Math.sin}) would introduce differences that are
     * about the math library rather than about correctness, and would obscure the measurement.
     */
    public static void heavyScaleAndSelect(
            DoubleArray val, DoubleArray out, IntArray mask,
            double mul, double add, double threshold, int intensity) {
        for (@Parallel int i = 0; i < val.getSize(); i++) {
            double v = val.get(i);
            double acc = v;
            for (int k = 0; k < intensity; k++) {
                acc = TornadoMath.sqrt(acc * acc + 1.0) * 0.5 + acc * 0.25;
            }
            out.set(i, acc * mul + add);
            if (v > threshold) {
                mask.set(i, 1);
            } else {
                mask.set(i, 0);
            }
        }
    }

    /** Host reference for {@link #heavyScaleAndSelect}. */
    public static void heavyScaleAndSelectReference(
            DoubleArray val, DoubleArray out, IntArray mask,
            double mul, double add, double threshold, int intensity) {
        for (int i = 0; i < val.getSize(); i++) {
            double v = val.get(i);
            double acc = v;
            for (int k = 0; k < intensity; k++) {
                acc = Math.sqrt(acc * acc + 1.0) * 0.5 + acc * 0.25;
            }
            out.set(i, acc * mul + add);
            mask.set(i, v > threshold ? 1 : 0);
        }
    }
}
