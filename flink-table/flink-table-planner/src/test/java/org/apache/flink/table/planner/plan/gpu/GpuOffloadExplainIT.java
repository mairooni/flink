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
import org.apache.flink.table.api.TableEnvironment;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** EXPLAIN must be able to answer "why did my query not run on the GPU?". */
class GpuOffloadExplainIT {

    private static final String CHEAP =
            "SELECT id, val * 2.0 + 1.0 AS scaled FROM t WHERE val > 0.5";
    private static final String HEAVY =
            "SELECT id, EXP(val) * LN(val) + SIN(val) * COS(val) + POWER(val, 3.0) AS h "
                    + "FROM t WHERE val > 0.5";

    private TableEnvironment tEnv;

    @BeforeEach
    void setUp() {
        tEnv = TableEnvironment.create(EnvironmentSettings.newInstance().inBatchMode().build());
        tEnv.executeSql(
                "CREATE TABLE t (id BIGINT, val DOUBLE) WITH ("
                        + "'connector' = 'datagen', 'number-of-rows' = '10')");
    }

    @Test
    @DisplayName("no section at all when the feature is off")
    void silentWhenDisabled() {
        String plan = tEnv.explainSql(HEAVY);
        assertFalse(plan.contains("GPU Offload"),
                "a feature that is off must not change EXPLAIN output:\n" + plan);
    }

    /** Prints both plans, so the human-facing output is inspectable rather than only asserted. */
    @Test
    @DisplayName("show the rendered section")
    void showOutput() {
        tEnv.getConfig().set(GpuOffloadOptions.ENABLED, true);
        for (String sql : new String[] {CHEAP, HEAVY}) {
            String plan = tEnv.explainSql(sql);
            System.out.println("---------- " + sql);
            System.out.println(plan.substring(plan.indexOf("== Optimized Execution Plan ==")));
        }
    }

    @Test
    @DisplayName("a rejected query says why, naming the cost and the floor")
    void rejectionIsExplained() {
        tEnv.getConfig().set(GpuOffloadOptions.ENABLED, true);
        String plan = tEnv.explainSql(CHEAP);

        assertTrue(plan.contains("== GPU Offload =="), plan);
        assertTrue(plan.contains("CPU"), plan);
        assertTrue(plan.contains("below cost floor"), plan);
        // The numbers matter: "3 < 96" tells the user how far off they are and what to change.
        assertTrue(plan.contains("3 < 96"), plan);
    }

    @Test
    @DisplayName("an accepted query is shown as offloaded, with its subtree")
    void acceptanceIsExplained() {
        tEnv.getConfig().set(GpuOffloadOptions.ENABLED, true);
        String plan = tEnv.explainSql(HEAVY);

        assertTrue(plan.contains("== GPU Offload =="), plan);
        assertTrue(plan.contains("GPU"), plan);
        assertTrue(plan.contains("clears floor"), plan);
        assertTrue(plan.contains("subtree 0"), plan);
    }

    @Test
    @DisplayName("an inexpressible type is reported as such, not as merely too cheap")
    void inexpressibleIsExplained() {
        tEnv.getConfig().set(GpuOffloadOptions.ENABLED, true);
        tEnv.executeSql(
                "CREATE TABLE s (word STRING, n INT) WITH ("
                        + "'connector' = 'datagen', 'number-of-rows' = '10')");
        String plan = tEnv.explainSql("SELECT word, n + 1 AS m FROM s WHERE n > 0");

        assertTrue(plan.contains("not expressible on device"), plan);
    }
}
