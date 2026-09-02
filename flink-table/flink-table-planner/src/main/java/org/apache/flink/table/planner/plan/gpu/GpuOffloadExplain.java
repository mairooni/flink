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

import org.apache.flink.table.planner.plan.nodes.exec.ExecNode;
import org.apache.flink.table.planner.plan.nodes.exec.ExecNodeGraph;
import org.apache.flink.table.planner.plan.nodes.exec.visitor.AbstractExecNodeExactlyOnceVisitor;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Renders what the offload processor decided, as a section appended to EXPLAIN output.
 *
 * <p>"Why did my query not run on the GPU?" is the first question anyone asks, and without this the
 * answer is invisible: the processor annotates nodes but does not change the plan, so an offloaded
 * and a rejected plan print identically.
 *
 * <p>Deliberately a separate section rather than extra attributes on the node descriptions.
 * {@code ExecNodeBase.description} is fixed at construction and is compared verbatim by a large
 * number of plan-comparison test resources; appending to it would rewrite all of them for a feature
 * that is off by default.
 */
public class GpuOffloadExplain {

    private GpuOffloadExplain() {}

    /** Returns the section, or an empty string when the processor examined nothing. */
    public static String format(ExecNodeGraph graph) {
        Map<Integer, List<ExecNode<?>>> byGroup = new LinkedHashMap<>();
        List<ExecNode<?>> ungrouped = new ArrayList<>();

        AbstractExecNodeExactlyOnceVisitor visitor =
                new AbstractExecNodeExactlyOnceVisitor() {
                    @Override
                    protected void visitNode(ExecNode<?> node) {
                        visitInputs(node);
                        GpuOffloadAssignment assignment = node.getGpuOffloadAssignment();
                        if (assignment == null) {
                            return;
                        }
                        if (assignment.groupId() < 0) {
                            ungrouped.add(node);
                        } else {
                            byGroup.computeIfAbsent(
                                            assignment.groupId(), key -> new ArrayList<>())
                                    .add(node);
                        }
                    }
                };
        graph.getRootNodes().forEach(node -> node.accept(visitor));

        if (byGroup.isEmpty() && ungrouped.isEmpty()) {
            return "";
        }

        StringBuilder sb = new StringBuilder();
        for (Map.Entry<Integer, List<ExecNode<?>>> entry : byGroup.entrySet()) {
            List<ExecNode<?>> members = entry.getValue();
            GpuOffloadAssignment assignment = members.get(0).getGpuOffloadAssignment();
            sb.append(assignment.isOffloaded() ? "GPU  " : "CPU  ");
            sb.append("subtree ").append(entry.getKey()).append(": ");
            for (int i = 0; i < members.size(); i++) {
                if (i > 0) {
                    sb.append(" <- ");
                }
                sb.append(members.get(i).getDescription());
            }
            sb.append(System.lineSeparator());
            sb.append("       ").append(assignment.verdict().reason());
            sb.append(System.lineSeparator());
        }
        for (ExecNode<?> node : ungrouped) {
            sb.append("CPU  ").append(node.getDescription()).append(System.lineSeparator());
            sb.append("       ")
                    .append(node.getGpuOffloadAssignment().verdict().reason())
                    .append(System.lineSeparator());
        }
        return sb.toString();
    }
}
