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

import java.io.Serializable;
import java.util.Arrays;

/**
 * Java source for a kernel generated from a Calc's expressions, plus the buffer layout the runtime
 * needs in order to call it.
 *
 * <p>This replaces the fixed kernel catalogue. The catalogue could only ever match expressions
 * someone had anticipated — measured at two of thirteen ordinary query shapes — because a kernel
 * must exist as a compiled method before the query does. Generating the method at run time removes
 * that constraint.
 *
 * <p>Serializable because it travels inside the operator into the JobGraph: the planner generates
 * the source on the client and the TaskManager compiles and runs it.
 */
public final class GpuKernelSource implements Serializable {

    private static final long serialVersionUID = 1L;

    private final String className;
    private final String methodName;
    private final String source;
    private final int[] inputFieldIndexes;
    private final int outputCount;
    private final boolean hasFilter;
    private final int[] outputLayout;

    public GpuKernelSource(
            String className,
            String methodName,
            String source,
            int[] inputFieldIndexes,
            int outputCount,
            boolean hasFilter,
            int[] outputLayout) {
        this.className = className;
        this.methodName = methodName;
        this.source = source;
        this.inputFieldIndexes = inputFieldIndexes;
        this.outputCount = outputCount;
        this.hasFilter = hasFilter;
        this.outputLayout = outputLayout;
    }

    public String className() {
        return className;
    }

    public String methodName() {
        return methodName;
    }

    public String source() {
        return source;
    }

    /**
     * Input row fields the kernel reads, in the order its parameters expect them.
     *
     * <p>Only referenced columns appear. A column that is merely projected through is staged
     * host-side and never crosses to the device, so transferring it would be pure cost.
     */
    public int[] inputFieldIndexes() {
        return inputFieldIndexes;
    }

    /** Number of computed output columns the kernel writes. */
    public int outputCount() {
        return outputCount;
    }

    /**
     * Whether the kernel also writes a 0/1 selection mask, i.e. whether the Calc had a condition.
     */
    public boolean hasFilter() {
        return hasFilter;
    }

    /**
     * For each output field, the input field it is copied from, or {@code GpuCalcSpec.COMPUTED} for
     * one the kernel produces.
     */
    public int[] outputLayout() {
        return outputLayout;
    }

    @Override
    public String toString() {
        return "GpuKernelSource["
                + className
                + ", inputs="
                + Arrays.toString(inputFieldIndexes)
                + ", outputs="
                + outputCount
                + (hasFilter ? ", filtered" : "")
                + "]";
    }
}
