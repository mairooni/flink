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
import org.apache.flink.table.planner.plan.gpu.GpuCalcSpec;
import org.apache.flink.table.planner.plan.gpu.GpuOperatorFactoryProvider;
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
     * Rows staged before a kernel launch.
     *
     * <p>Measured across 64K to 4M: device work stays around a tenth of the total throughout, and
     * larger batches mainly cut TornadoVM dispatch overhead (51 ms to 3.6 ms over 8M rows). 256K is
     * comfortably past the point where dispatch stops dominating without holding an unreasonable
     * amount of memory per subtask.
     */
    private static final int BATCH_SIZE =
            Integer.getInteger("flink.gpu.offload.batch-size", 262_144);

    /**
     * Whether to collect the gather / copy-in / kernel / copy-out breakdown.
     *
     * <p>On by default while the project is establishing where time goes. It costs a profiler
     * query per batch, not per record.
     */
    private static final boolean PROFILE =
            Boolean.parseBoolean(System.getProperty("flink.gpu.offload.profile", "true"));

    @Override
    public Optional<StreamOperatorFactory<RowData>> createCalcOperatorFactory(GpuCalcSpec spec) {
        if (!canStageOutput(spec)) {
            return Optional.empty();
        }
        GpuCalcOperator operator = new GpuCalcOperator(spec, BATCH_SIZE, PROFILE);
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
        return "TornadoVM (batch=" + BATCH_SIZE + ", profile=" + PROFILE + ")";
    }
}
