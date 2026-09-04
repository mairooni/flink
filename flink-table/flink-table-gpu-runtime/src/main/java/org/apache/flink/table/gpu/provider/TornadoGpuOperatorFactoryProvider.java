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

package org.apache.flink.table.gpu.provider;

import org.apache.flink.streaming.api.operators.SimpleOperatorFactory;
import org.apache.flink.streaming.api.operators.StreamOperatorFactory;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.gpu.operator.GpuCalcOperator;
import org.apache.flink.table.runtime.gpu.GpuCalcSpec;
import org.apache.flink.table.runtime.gpu.GpuOperatorFactoryProvider;
import org.apache.flink.table.types.logical.LogicalType;
import org.apache.flink.table.types.logical.LogicalTypeRoot;

import java.util.List;
import java.util.Optional;

/**
 * Supplies TornadoVM-backed operators to the planner, discovered through {@code META-INF/services}.
 *
 * <p>The planner holds no compile-time reference to TornadoVM; this class is the only place the two
 * meet. If this jar is absent from a TaskManager's classpath, the planner falls back to normal code
 * generation and says so in EXPLAIN, so a heterogeneous cluster degrades rather than fails.
 *
 * <h2>Declining is normal</h2>
 *
 * <p>Returning empty from {@link #createCalcOperatorFactory} is a supported answer, not an error.
 * The planner's cost gate decides whether an expression is <em>worth</em> offloading; this class
 * decides whether the kernel catalogue can actually express it. Those are different questions and
 * the second is currently much narrower.
 */
public class TornadoGpuOperatorFactoryProvider implements GpuOperatorFactoryProvider {

    /**
     * Whether to collect the gather / copy-in / kernel / copy-out breakdown.
     *
     * <p>On while the project is establishing where time goes. It costs one profiler query per
     * batch, not per record.
     */
    private static final boolean PROFILE = true;

    /**
     * Why TornadoVM cannot be used in this JVM, or null if it can.
     *
     * <p>Probed once, by loading a class compiled against TornadoVM's off-heap array types. Those
     * are built on {@code java.lang.foreign}, a preview API on JDK 21, so a JVM started without
     * {@code --enable-preview} cannot load them.
     *
     * <p>The probe exists because failing here is far better than failing later. Without it the
     * planner substitutes a GPU operator, the job is submitted, and the TaskManager dies
     * deserializing the operator with {@code UnsupportedClassVersionError: Preview features are not
     * enabled} -- an error that says nothing about GPU offload and appears a long way from its
     * cause. Declining instead makes the planner fall back to code generation and report the reason
     * in EXPLAIN.
     */
    private static final String UNAVAILABLE = probeTornado();

    private static String probeTornado() {
        try {
            // Loading the class is the check; it drags in DoubleArray and IntArray.
            Class.forName(
                    "org.apache.flink.table.gpu.operator.GeneratedKernelEngine",
                    false,
                    TornadoGpuOperatorFactoryProvider.class.getClassLoader());
            return null;
        } catch (Throwable t) {
            String detail = t.getMessage() == null ? t.getClass().getName() : t.getMessage();
            if (t instanceof UnsupportedClassVersionError) {
                return "TornadoVM needs --enable-preview on this JVM (" + detail + ")";
            }
            return "TornadoVM is not usable in this JVM: " + detail;
        }
    }

    @Override
    public Optional<StreamOperatorFactory<RowData>> createCalcOperatorFactory(GpuCalcSpec spec) {
        if (UNAVAILABLE != null) {
            return Optional.empty();
        }
        if (!canStageOutput(spec)) {
            return Optional.empty();
        }
        GpuCalcOperator operator = new GpuCalcOperator(spec, spec.batchSize(), PROFILE);
        return Optional.of(SimpleOperatorFactory.of(operator));
    }

    /**
     * The kernel writes a DOUBLE, and every other output column is staged host-side in a primitive
     * array. A type outside that set is refused here rather than failing later inside the operator.
     */
    private static boolean canStageOutput(GpuCalcSpec spec) {
        List<LogicalType> fields = spec.outputType().getChildren();
        int[] layout = spec.outputLayout();
        if (fields.size() != layout.length) {
            return false;
        }
        for (int i = 0; i < layout.length; i++) {
            LogicalTypeRoot root = fields.get(i).getTypeRoot();
            if (layout[i] == GpuCalcSpec.COMPUTED) {
                if (root != LogicalTypeRoot.DOUBLE) {
                    return false;
                }
            } else if (root != LogicalTypeRoot.BIGINT
                    && root != LogicalTypeRoot.INTEGER
                    && root != LogicalTypeRoot.DOUBLE) {
                return false;
            }
        }
        return true;
    }

    @Override
    public String describe() {
        return UNAVAILABLE != null ? UNAVAILABLE : "TornadoVM (profile=" + PROFILE + ")";
    }
}
