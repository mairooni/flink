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

package org.apache.flink.table.planner.plan.gpu;

import org.apache.flink.table.types.logical.RowType;

import javax.annotation.Nullable;

import java.io.Serializable;

/**
 * What the runtime needs in order to execute one offloaded Calc, expressed without any reference to
 * Calcite.
 *
 * <p>The runtime library holds a fixed set of javac-compiled kernels — it cannot be handed a
 * {@code RexNode} tree and asked to interpret it, because TornadoVM resolves a kernel from a method
 * reference's {@code SerializedLambda} and so the method must exist before the query does. The
 * planner therefore matches the expression tree against the kernel catalogue and passes a
 * descriptor: which kernel, over which column, with which scalar constants.
 *
 * <p>Serializable because it travels inside the operator into the JobGraph: the planner builds it
 * on the client and the TaskManager reconstructs the operator from it.
 *
 * <p>This is the narrow form for the one kernel shape that exists today,
 * {@code out = col * mul + add} with optional selection {@code col > threshold}. Widening it is the
 * point at which either the catalogue grows or the planner starts generating device code, which is
 * the fork in the road recorded as an open question in the design document.
 */
public final class GpuCalcSpec implements Serializable {

    private static final long serialVersionUID = 1L;

    /** Marks the kernel-produced column in {@link #outputLayout()}. */
    public static final int COMPUTED = -1;

    private final int inputFieldIndex;
    private final double mul;
    private final double add;
    private final @Nullable Double threshold;
    private final RowType outputType;
    private final int[] outputLayout;
    private final int rowCost;

    public GpuCalcSpec(
            int inputFieldIndex,
            double mul,
            double add,
            @Nullable Double threshold,
            RowType outputType,
            int[] outputLayout,
            int rowCost) {
        this.inputFieldIndex = inputFieldIndex;
        this.mul = mul;
        this.add = add;
        this.threshold = threshold;
        this.outputType = outputType;
        this.outputLayout = outputLayout;
        this.rowCost = rowCost;
    }

    /** Index of the DOUBLE column the kernel reads. */
    public int inputFieldIndex() {
        return inputFieldIndex;
    }

    public double mul() {
        return mul;
    }

    public double add() {
        return add;
    }

    /** Null when the Calc has no condition, i.e. every row survives. */
    public @Nullable Double threshold() {
        return threshold;
    }

    public RowType outputType() {
        return outputType;
    }

    /**
     * For each output field, the input field it is copied from, or {@link #COMPUTED} for the one
     * the kernel produces.
     *
     * <p>Pass-through columns never reach the device. They are staged host-side alongside the
     * kernel's input and emitted straight from that buffer, which is both cheaper and the only way
     * to carry a type the kernel has no representation for.
     */
    public int[] outputLayout() {
        return outputLayout;
    }

    /** Carried through so the runtime can report it alongside the measured breakdown. */
    public int rowCost() {
        return rowCost;
    }

    @Override
    public String toString() {
        return "GpuCalcSpec[field=" + inputFieldIndex + ", out=col*" + mul + "+" + add
                + (threshold == null ? ", no filter" : ", filter>" + threshold)
                + ", cost=" + rowCost + "]";
    }
}
