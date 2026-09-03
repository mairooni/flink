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

package org.apache.flink.table.runtime.gpu;

import org.apache.flink.table.types.logical.RowType;

import java.io.Serializable;
import java.util.Arrays;

/**
 * Everything the runtime needs to execute one offloaded Calc, expressed without reference to
 * Calcite.
 *
 * <p>The kernel arrives as source rather than as a catalogue selection. A fixed catalogue could
 * only match expressions someone had anticipated, because a kernel had to exist as a compiled
 * method before the query did; generating it at run time removes that ceiling.
 *
 * <p>Serializable because it travels inside the operator into the JobGraph: the planner generates
 * on the client, the TaskManager compiles and runs.
 */
public final class GpuCalcSpec implements Serializable {

    private static final long serialVersionUID = 2L;

    /** Marks an output field produced by the kernel rather than copied from the input. */
    public static final int COMPUTED = -1;

    private final GpuKernelSource kernel;
    private final int[] outputLayout;
    private final RowType outputType;
    private final int batchSize;
    private final int rowCost;

    public GpuCalcSpec(
            GpuKernelSource kernel,
            int[] outputLayout,
            RowType outputType,
            int batchSize,
            int rowCost) {
        this.kernel = kernel;
        this.outputLayout = outputLayout;
        this.outputType = outputType;
        this.batchSize = batchSize;
        this.rowCost = rowCost;
    }

    public GpuKernelSource kernel() {
        return kernel;
    }

    /**
     * For each output field, the input field it is copied from, or {@link #COMPUTED} for one the
     * kernel produces.
     *
     * <p>The kernel's outputs fill the {@code COMPUTED} slots in order, which is the order the
     * projections were generated in.
     *
     * <p>Pass-through columns never reach the device: they are staged host-side and emitted from
     * that buffer. That is both cheaper and the only way to carry a type the kernel cannot
     * represent — a {@code BIGINT} key, for instance, which would lose precision as a double.
     */
    public int[] outputLayout() {
        return outputLayout;
    }

    public RowType outputType() {
        return outputType;
    }

    /** Rows staged before each kernel launch, from {@code table.exec.gpu-offload.batch-size}. */
    public int batchSize() {
        return batchSize;
    }

    /** Estimated weighted work per row, carried through so the runtime can report it. */
    public int rowCost() {
        return rowCost;
    }

    @Override
    public String toString() {
        return "GpuCalcSpec["
                + kernel
                + ", layout="
                + Arrays.toString(outputLayout)
                + ", batch="
                + batchSize
                + ", cost="
                + rowCost
                + "]";
    }
}
