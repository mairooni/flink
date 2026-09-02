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

/**
 * What the offload processor decided about one {@code ExecNode}, and which subtree it belongs to.
 *
 * <p>Attached to the node so that {@code translateToPlanInternal} can consult it later: the
 * processor runs over the whole graph before any translation happens, so the decision has to be
 * carried across that boundary.
 *
 * <p>The group id matters as much as the verdict. Offloading is per subtree, not per node — one
 * task graph for the whole group, with intermediates kept on the device — so nodes sharing a group
 * id translate into one operator rather than several.
 */
public final class GpuOffloadAssignment {

    private final int groupId;
    private final GpuOffloadDecision.Verdict verdict;

    public GpuOffloadAssignment(int groupId, GpuOffloadDecision.Verdict verdict) {
        this.groupId = groupId;
        this.verdict = verdict;
    }

    /** Identifies the maximal offloadable subtree this node was grouped into. */
    public int groupId() {
        return groupId;
    }

    public GpuOffloadDecision.Verdict verdict() {
        return verdict;
    }

    public boolean isOffloaded() {
        return verdict.offload();
    }

    @Override
    public String toString() {
        return "group " + groupId + ": " + verdict;
    }
}
