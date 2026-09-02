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

import org.apache.flink.table.api.EnvironmentSettings;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.TableEnvironment;
import org.apache.flink.table.planner.delegation.PlannerBase;
import org.apache.flink.table.planner.plan.nodes.exec.ExecNode;
import org.apache.flink.table.planner.plan.nodes.exec.ExecNodeGraph;
import org.apache.flink.table.planner.plan.nodes.exec.batch.BatchExecCalc;
import org.apache.flink.table.planner.utils.TableTestUtil;

import org.apache.calcite.rel.RelNode;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.apache.flink.table.planner.utils.JavaScalaConversionUtil.toScala;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * End-to-end check that the offload gate sees what the planner actually produces.
 *
 * <p>The unit tests build {@code RexNode}s with a {@code RexBuilder}. This one plans real SQL and
 * costs the {@link BatchExecCalc} the optimizer emitted, which is the only way to catch the two
 * things a hand-built tree cannot: that expressions arrive expanded rather than as {@code
 * RexLocalRef} program form, and that constant folding has already happened by the time the gate
 * runs.
 */
class GpuOffloadPlanIT {

    private TableEnvironment tEnv;
    private PlannerBase planner;

    @BeforeEach
    void setUp() {
        tEnv = TableEnvironment.create(EnvironmentSettings.newInstance().inBatchMode().build());
        tEnv.executeSql(
                "CREATE TABLE t (id BIGINT, val DOUBLE) WITH ("
                        + "'connector' = 'datagen', 'number-of-rows' = '10')");
        planner =
                (PlannerBase)
                        ((org.apache.flink.table.api.internal.TableEnvironmentImpl) tEnv)
                                .getPlanner();
    }

    /** Plans {@code sql} and returns the cost of its single Calc node. */
    private RowCost costOfCalc(String sql) {
        Table table = tEnv.sqlQuery(sql);
        RelNode optimized = planner.optimize(TableTestUtil.toRelNode(table));
        ExecNodeGraph graph =
                planner.translateToExecNodeGraph(
                        toScala(Collections.singletonList(optimized)), false);

        List<BatchExecCalc> calcs = new ArrayList<>();
        collectCalcs(graph.getRootNodes(), calcs);
        assertEquals(1, calcs.size(), "expected exactly one Calc in: " + sql);

        BatchExecCalc calc = calcs.get(0);
        assertTrue(calc.supportGpuOffload(), "a Calc must declare offload support");
        return calc.estimateRowCost();
    }

    private static void collectCalcs(List<? extends ExecNode<?>> nodes, List<BatchExecCalc> out) {
        for (ExecNode<?> node : nodes) {
            if (node instanceof BatchExecCalc) {
                out.add((BatchExecCalc) node);
            }
            List<ExecNode<?>> inputs = new ArrayList<>();
            node.getInputEdges().forEach(edge -> inputs.add(edge.getSource()));
            collectCalcs(inputs, out);
        }
    }

    @Test
    @DisplayName("the measured-losing query costs 3 and is refused by the default floor")
    void cheapQueryIsRefused() {
        RowCost cost = costOfCalc("SELECT id, val * 2.0 + 1.0 AS scaled FROM t WHERE val > 0.5");

        assertTrue(cost.isEligible(), cost::rejection);
        // one multiply, one add, one comparison -- the same 3 the unit test predicts from a
        // hand-built tree, confirming expressions arrive expanded rather than as RexLocalRef.
        assertEquals(3, cost.weight());

        GpuOffloadDecision.Verdict v =
                GpuOffloadDecision.withDefaults().forSubtree(Collections.singletonList(cost));
        assertFalse(v.offload(), "measured 0.54x against CPU; must not be offloaded");
    }

    @Test
    @DisplayName("a transcendental-heavy query clears the floor")
    void heavyQueryIsAccepted() {
        RowCost cost =
                costOfCalc(
                        "SELECT id, EXP(val) * LN(val) + SIN(val) * COS(val) + POWER(val, 3.0) AS h "
                                + "FROM t WHERE val > 0.5");

        assertTrue(cost.isEligible(), cost::rejection);
        GpuOffloadDecision.Verdict v =
                GpuOffloadDecision.withDefaults().forSubtree(Collections.singletonList(cost));
        assertTrue(v.offload(), v::reason);
    }

    @Test
    @DisplayName("a VARCHAR projection is refused as inexpressible, not merely as cheap")
    void varcharIsRefused() {
        tEnv.executeSql(
                "CREATE TABLE s (word STRING, n INT) WITH ("
                        + "'connector' = 'datagen', 'number-of-rows' = '10')");
        RowCost cost = costOfCalc("SELECT word, n + 1 FROM s WHERE n > 0");

        assertFalse(cost.isEligible());
        GpuOffloadDecision.Verdict v =
                GpuOffloadDecision.withDefaults().forSubtree(Collections.singletonList(cost));
        assertTrue(v.reason().contains("not expressible on device"), v::reason);
    }
}
