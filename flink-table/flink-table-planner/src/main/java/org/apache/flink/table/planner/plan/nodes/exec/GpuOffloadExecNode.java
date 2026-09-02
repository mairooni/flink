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

import org.apache.flink.table.planner.plan.gpu.RowCost;

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

    /** Whether this ExecNode's work is expressible as a device kernel at all. */
    boolean supportGpuOffload();

    /**
     * Estimated device-eligible work for one input row, in the weighted units of {@code
     * OperatorWeights}, or an ineligible {@link RowCost} naming what cannot be expressed.
     *
     * <p>This is a static walk over the node's {@code RexNode}s. It must not execute anything, and
     * must be cheap enough to call for every node of every plan while the flag is enabled.
     *
     * <p>Only meaningful when {@link #supportGpuOffload()} is true.
     */
    RowCost estimateRowCost();
}
