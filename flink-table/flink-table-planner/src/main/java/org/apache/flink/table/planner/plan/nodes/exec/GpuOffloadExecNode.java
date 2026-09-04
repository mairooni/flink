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

package org.apache.flink.table.planner.plan.nodes.exec;

import org.apache.flink.table.planner.plan.gpu.GpuOffloadAssignment;
import org.apache.flink.table.planner.plan.gpu.RowCost;

import javax.annotation.Nullable;

/**
 * Capability interface for {@link ExecNode}s whose per-record work can be executed as a
 * data-parallel device kernel.
 *
 * <p>Deliberately shaped like {@link FusionCodegenExecNode}: a per-node predicate defaulting to
 * false on {@link ExecNodeBase}, so nodes opt in one at a time and every node that has not been
 * examined behaves exactly as it does today.
 *
 * <p>Unlike fusion codegen, offload needs a second question answered. Whether an expression
 * <em>can</em> run on a device and whether it is <em>worth</em> running there are independent:
 * {@code val * 2.0 + 1.0} is trivially expressible and measures 0.54x against CPU execution, so a
 * predicate that only checks expressibility would make such queries roughly twice as slow. Hence
 * {@link #estimateRowCost()}, whose result is compared against a calibrated floor before any
 * substitution happens.
 */
public interface GpuOffloadExecNode {

    /** Row count is not known: the node was built without one, or the planner had no estimate. */
    double UNKNOWN_ROW_COUNT = -1.0;

    /**
     * The planner's estimate of how many rows this node will see.
     *
     * <p>Needed because a per-row cost cannot answer whether offloading pays. Staging buffers,
     * javac, and the device's own compilation are paid once per task whatever the row count, so an
     * expression heavy enough to win on every row still loses over a short input. Captured when the
     * physical node is translated, since an {@link ExecNode} has no metadata query of its own.
     */
    default void setEstimatedRowCount(double rowCount) {
        // Discarded by default, like the assignment below: a node that cannot be offloaded is
        // never costed, so there is nothing to remember.
    }

    /**
     * @return the estimate, or {@link #UNKNOWN_ROW_COUNT} when none was captured
     */
    default double getEstimatedRowCount() {
        return UNKNOWN_ROW_COUNT;
    }

    /**
     * Whether this ExecNode's work is expressible as a device kernel at all.
     *
     * <p>Defaulted rather than abstract so that adding this capability does not force every
     * existing and future implementor of {@link ExecNode} to change. A node that has not been
     * examined for offload behaves exactly as it did before.
     */
    default boolean supportGpuOffload() {
        return false;
    }

    /**
     * Estimated device-eligible work for one input row, in the weighted units of {@code
     * OperatorWeights}, or an ineligible {@link RowCost} naming what cannot be expressed.
     *
     * <p>This is a static walk over the node's {@code RexNode}s. It must not execute anything, and
     * must be cheap enough to call for every node of every plan while the flag is enabled.
     *
     * <p>Only meaningful when {@link #supportGpuOffload()} is true.
     */
    default RowCost estimateRowCost() {
        // Reached only if a node claims support without overriding this; naming the class makes
        // that mistake obvious in EXPLAIN rather than silently costing the node as free.
        return RowCost.ineligible(
                "ExecNode " + getClass().getSimpleName() + " declares no GPU row cost");
    }

    /**
     * Records what {@code GpuOffloadProcessor} decided for this node.
     *
     * <p>The processor runs over the whole graph before translation begins, so the decision has to
     * be carried across that boundary to the point where the node builds its {@code
     * Transformation}.
     */
    default void setGpuOffloadAssignment(@Nullable GpuOffloadAssignment assignment) {
        // Discarded by default. A node that does not support offload is never selected, so there
        // is nothing to remember; implementors that can be offloaded store it (see ExecNodeBase).
    }

    /** Null when the processor did not run, or did not examine this node. */
    @Nullable
    default GpuOffloadAssignment getGpuOffloadAssignment() {
        return null;
    }
}
