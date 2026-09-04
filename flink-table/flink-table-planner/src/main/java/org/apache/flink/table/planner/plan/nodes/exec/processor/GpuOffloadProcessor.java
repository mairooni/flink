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

package org.apache.flink.table.planner.plan.nodes.exec.processor;

import org.apache.flink.table.api.TableConfig;
import org.apache.flink.table.planner.plan.gpu.GpuOffloadAssignment;
import org.apache.flink.table.planner.plan.gpu.GpuOffloadDecision;
import org.apache.flink.table.planner.plan.gpu.GpuOffloadOptions;
import org.apache.flink.table.planner.plan.gpu.RowCost;
import org.apache.flink.table.planner.plan.nodes.exec.ExecNode;
import org.apache.flink.table.planner.plan.nodes.exec.ExecNodeGraph;
import org.apache.flink.table.planner.plan.nodes.exec.GpuOffloadExecNode;
import org.apache.flink.table.planner.plan.nodes.exec.visitor.AbstractExecNodeExactlyOnceVisitor;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Finds maximal connected subtrees whose nodes can all run as device kernels, costs each subtree as
 * a unit, and records the decision on its nodes.
 *
 * <p>This processor does not rewrite the graph. It annotates it: the substitution — emitting one
 * offload operator per selected group instead of per-node {@code CodeGenOperatorFactory}s — happens
 * during translation, which is where each node builds its {@code Transformation}. Separating the
 * two keeps the decision in one place and makes it inspectable before anything is built.
 *
 * <h2>Why subtrees rather than nodes</h2>
 *
 * <p>Two independent reasons, both measured.
 *
 * <p><b>Transfer count.</b> A task graph per operator pays a host-to-device-to-host round trip even
 * when the consumer is another offloaded operator on the same device. Grouping turns 2n transfers
 * into 2.
 *
 * <p><b>Clearing the floor.</b> Costs sum across a group, so a chain of Calcs can exceed a
 * threshold that none of its members exceeds alone. A single cheap node is expected to be rejected,
 * and that is the correct outcome rather than a limitation.
 *
 * <h2>Grouping rule</h2>
 *
 * <p>A candidate node joins its consumer's group only when it has exactly one consumer. A node
 * feeding two consumers cannot have its output kept on the device for one of them and materialised
 * on the host for the other without either duplicating the kernel or adding a copy-out that the
 * grouping exists to avoid, so it starts its own group instead.
 */
public class GpuOffloadProcessor implements ExecNodeGraphProcessor {

    /** Group id for a node that was examined but could not join any subtree. */
    public static final int NO_GROUP = -1;

    @Override
    public ExecNodeGraph process(ExecNodeGraph execGraph, ProcessorContext context) {
        TableConfig config = context.getPlanner().getTableConfig();
        if (!config.get(GpuOffloadOptions.ENABLED)) {
            return execGraph;
        }
        GpuOffloadDecision decision =
                new GpuOffloadDecision(
                        config.get(GpuOffloadOptions.MIN_ROW_COST),
                        config.get(GpuOffloadOptions.MIN_TOTAL_WORK));

        List<ExecNode<?>> order = topologicalOrder(execGraph);
        Map<ExecNode<?>, Integer> consumerCount = countConsumers(order);
        Map<ExecNode<?>, RowCost> costs = new IdentityHashMap<>();
        Map<ExecNode<?>, Integer> groupOf = new IdentityHashMap<>();
        Map<Integer, List<ExecNode<?>>> groups = new LinkedHashMap<>();
        int nextGroupId = 0;

        // Producers are visited before consumers, so a node's inputs already have a group when it
        // is considered and can be absorbed into the node's own.
        for (ExecNode<?> node : order) {
            if (!node.supportGpuOffload()) {
                continue;
            }
            RowCost cost = node.estimateRowCost();
            if (!cost.isEligible()) {
                // Record the reason so EXPLAIN can say why, but do not group an ineligible node:
                // one such member would poison an otherwise viable subtree.
                assign(
                        node,
                        new GpuOffloadAssignment(
                                NO_GROUP,
                                decision.forSubtree(
                                        Collections.singletonList(cost), rowCountOf(node))));
                continue;
            }
            costs.put(node, cost);

            Integer group = null;
            for (ExecNode<?> input : inputsOf(node)) {
                Integer inputGroup = groupOf.get(input);
                if (inputGroup == null || consumerCount.getOrDefault(input, 0) != 1) {
                    continue;
                }
                if (group == null) {
                    group = inputGroup;
                } else if (!group.equals(inputGroup)) {
                    // The node joins two upstream groups; merge the smaller into the larger.
                    group = merge(groups, groupOf, group, inputGroup);
                }
            }
            if (group == null) {
                group = nextGroupId++;
                groups.put(group, new ArrayList<>());
            }
            groups.get(group).add(node);
            groupOf.put(node, group);
        }

        for (Map.Entry<Integer, List<ExecNode<?>>> entry : groups.entrySet()) {
            List<RowCost> memberCosts = new ArrayList<>();
            for (ExecNode<?> member : entry.getValue()) {
                memberCosts.add(costs.get(member));
            }
            GpuOffloadDecision.Verdict verdict =
                    decision.forSubtree(memberCosts, rowCountOf(entry.getValue()));
            for (ExecNode<?> member : entry.getValue()) {
                assign(member, new GpuOffloadAssignment(entry.getKey(), verdict));
            }
        }

        return execGraph;
    }

    private static Integer merge(
            Map<Integer, List<ExecNode<?>>> groups,
            Map<ExecNode<?>, Integer> groupOf,
            Integer a,
            Integer b) {
        List<ExecNode<?>> from = groups.get(b);
        List<ExecNode<?>> into = groups.get(a);
        if (from.size() > into.size()) {
            return merge(groups, groupOf, b, a);
        }
        into.addAll(from);
        for (ExecNode<?> node : from) {
            groupOf.put(node, a);
        }
        groups.remove(b);
        return a;
    }

    /**
     * Rows the subtree will see, taken as the largest estimate among its members.
     *
     * <p>Members of one group run as a single kernel over the same batches, and a filter can only
     * narrow what follows it, so the widest member is what the kernel is actually sized against.
     * Unknown wins over any number: a group holding one node the planner could not estimate is a
     * group whose total work is not known, and guessing there would be worse than skipping the
     * gate.
     */
    private static double rowCountOf(List<ExecNode<?>> members) {
        double widest = 0.0;
        for (ExecNode<?> member : members) {
            double rows = rowCountOf(member);
            if (rows == GpuOffloadExecNode.UNKNOWN_ROW_COUNT) {
                return GpuOffloadExecNode.UNKNOWN_ROW_COUNT;
            }
            widest = Math.max(widest, rows);
        }
        return widest;
    }

    private static double rowCountOf(ExecNode<?> node) {
        return node.getEstimatedRowCount();
    }

    private static void assign(ExecNode<?> node, GpuOffloadAssignment assignment) {
        node.setGpuOffloadAssignment(assignment);
    }

    private static List<ExecNode<?>> inputsOf(ExecNode<?> node) {
        List<ExecNode<?>> inputs = new ArrayList<>();
        node.getInputEdges().forEach(edge -> inputs.add(edge.getSource()));
        return inputs;
    }

    /** Producers before consumers. */
    private static List<ExecNode<?>> topologicalOrder(ExecNodeGraph graph) {
        List<ExecNode<?>> order = new ArrayList<>();
        AbstractExecNodeExactlyOnceVisitor visitor =
                new AbstractExecNodeExactlyOnceVisitor() {
                    @Override
                    protected void visitNode(ExecNode<?> node) {
                        visitInputs(node);
                        order.add(node);
                    }
                };
        graph.getRootNodes().forEach(node -> node.accept(visitor));
        return order;
    }

    private static Map<ExecNode<?>, Integer> countConsumers(List<ExecNode<?>> order) {
        // Identity, like the rest of the maps here and like Flink's own exactly-once visitor:
        // ExecNode does not override equals, and two structurally identical nodes are still
        // distinct plan positions.
        Map<ExecNode<?>, Integer> counts = new IdentityHashMap<>();
        for (ExecNode<?> node : order) {
            for (ExecNode<?> input : inputsOf(node)) {
                counts.merge(input, 1, Integer::sum);
            }
        }
        return counts;
    }
}
