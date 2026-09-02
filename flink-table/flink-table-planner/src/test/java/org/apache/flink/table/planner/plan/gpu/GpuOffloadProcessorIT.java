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
import org.apache.flink.table.api.internal.TableEnvironmentImpl;
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
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Exercises {@code GpuOffloadProcessor} through the real planner pipeline, which is the only place
 * the flag gating, the graph traversal and the cost gate all run together.
 */
class GpuOffloadProcessorIT {

    private TableEnvironment tEnv;
    private PlannerBase planner;

    private static final String HEAVY =
            "EXP(val) * LN(val) + SIN(val) * COS(val) + POWER(val, 3.0)";

    @BeforeEach
    void setUp() {
        tEnv = TableEnvironment.create(EnvironmentSettings.newInstance().inBatchMode().build());
        tEnv.executeSql(
                "CREATE TABLE t (id BIGINT, val DOUBLE) WITH ("
                        + "'connector' = 'datagen', 'number-of-rows' = '10')");
        planner = (PlannerBase) ((TableEnvironmentImpl) tEnv).getPlanner();
    }

    private void enable() {
        tEnv.getConfig().set(GpuOffloadOptions.ENABLED, true);
    }

    /** Plans {@code sql} through the processor pipeline and returns every Calc in the graph. */
    private List<BatchExecCalc> planCalcs(String sql) {
        Table table = tEnv.sqlQuery(sql);
        RelNode optimized = planner.optimize(TableTestUtil.toRelNode(table));
        ExecNodeGraph graph =
                planner.translateToExecNodeGraph(
                        toScala(Collections.singletonList(optimized)), false);
        List<BatchExecCalc> calcs = new ArrayList<>();
        collect(graph.getRootNodes(), calcs);
        return calcs;
    }

    private static void collect(List<? extends ExecNode<?>> nodes, List<BatchExecCalc> out) {
        for (ExecNode<?> node : nodes) {
            if (node instanceof BatchExecCalc && !out.contains(node)) {
                out.add((BatchExecCalc) node);
            }
            List<ExecNode<?>> inputs = new ArrayList<>();
            node.getInputEdges().forEach(e -> inputs.add(e.getSource()));
            collect(inputs, out);
        }
    }

    @Test
    @DisplayName("disabled by default: the processor leaves no trace on the graph")
    void offIsOff() {
        List<BatchExecCalc> calcs = planCalcs("SELECT id, " + HEAVY + " AS h FROM t");

        assertEquals(1, calcs.size());
        assertNull(calcs.get(0).getGpuOffloadAssignment(),
                "with the flag off the processor must not annotate anything");
    }

    @Test
    @DisplayName("enabled: a heavy Calc is grouped and selected")
    void heavyCalcIsSelected() {
        enable();
        List<BatchExecCalc> calcs = planCalcs("SELECT id, " + HEAVY + " AS h FROM t");

        GpuOffloadAssignment assignment = calcs.get(0).getGpuOffloadAssignment();
        assertNotNull(assignment, "the processor must have examined the Calc");
        assertTrue(assignment.isOffloaded(), assignment::toString);
        assertTrue(assignment.groupId() >= 0, "a selected node belongs to a real group");
    }

    @Test
    @DisplayName("enabled: the cheap query is examined, grouped, and rejected by the floor")
    void cheapCalcIsRejected() {
        enable();
        List<BatchExecCalc> calcs =
                planCalcs("SELECT id, val * 2.0 + 1.0 AS scaled FROM t WHERE val > 0.5");

        GpuOffloadAssignment assignment = calcs.get(0).getGpuOffloadAssignment();
        assertNotNull(assignment);
        assertFalse(assignment.isOffloaded(), "measured 0.54x against CPU");
        assertTrue(assignment.verdict().reason().contains("below cost floor"),
                assignment.verdict()::reason);
    }

    @Test
    @DisplayName("lowering the floor flips the same query without touching the plan")
    void floorIsHonoured() {
        enable();
        tEnv.getConfig().set(GpuOffloadOptions.MIN_ROW_COST, 1);
        List<BatchExecCalc> calcs =
                planCalcs("SELECT id, val * 2.0 + 1.0 AS scaled FROM t WHERE val > 0.5");

        GpuOffloadAssignment assignment = calcs.get(0).getGpuOffloadAssignment();
        assertNotNull(assignment);
        assertTrue(assignment.isOffloaded(),
                "cost 3 clears a floor of 1; only the config changed");
    }

    @Test
    @DisplayName("an inexpressible Calc is recorded with a reason and no group")
    void inexpressibleCalcIsNotGrouped() {
        enable();
        tEnv.executeSql(
                "CREATE TABLE s (word STRING, n INT) WITH ("
                        + "'connector' = 'datagen', 'number-of-rows' = '10')");
        List<BatchExecCalc> calcs = planCalcs("SELECT word, n + 1 AS m FROM s WHERE n > 0");

        GpuOffloadAssignment assignment = calcs.get(0).getGpuOffloadAssignment();
        assertNotNull(assignment);
        assertFalse(assignment.isOffloaded());
        assertEquals(-1, assignment.groupId(),
                "an ineligible node must not join a group it would poison");
        assertTrue(assignment.verdict().reason().contains("not expressible on device"),
                assignment.verdict()::reason);
    }
}
