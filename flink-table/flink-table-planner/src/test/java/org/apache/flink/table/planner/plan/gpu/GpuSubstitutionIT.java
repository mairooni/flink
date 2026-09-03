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

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The substitution seam: what happens when a node is selected but the runtime cannot serve it.
 *
 * <p>Whether a given expression can be turned into a kernel is covered by {@link
 * GpuKernelGeneratorTest}. This is about the surrounding fallback, which has to be silent to the
 * query and visible in EXPLAIN.
 */
class GpuSubstitutionIT {

    @Test
    @DisplayName("with no provider installed, a selected node falls back and EXPLAIN says so")
    void fallsBackWithoutProvider() {
        TableEnvironment tEnv =
                TableEnvironment.create(EnvironmentSettings.newInstance().inBatchMode().build());
        tEnv.executeSql(
                "CREATE TABLE t (id BIGINT, val DOUBLE) WITH ("
                        + "'connector' = 'datagen', 'number-of-rows' = '10')");
        tEnv.getConfig().set(GpuOffloadOptions.ENABLED, true);
        // Floor of 1 so even a trivial expression is selected by the gate, isolating the
        // provider-absent path from the cost decision.
        tEnv.getConfig().set(GpuOffloadOptions.MIN_ROW_COST, 1);

        String plan =
                tEnv.explainSql("SELECT id, val * 2.0 + 1.0 AS scaled FROM t WHERE val > 0.5");

        // Planning must succeed with no provider present, and the fallback must be reported rather
        // than swallowed. flink-table-planner has no dependency on the GPU runtime, so this is the
        // state every build of this module is in.
        assertTrue(plan.contains("== GPU Offload =="), plan);
        assertTrue(plan.contains("selected but fell back"), plan);
        assertTrue(plan.contains("no GpuOperatorFactoryProvider on the classpath"), plan);
        assertFalse(plan.contains("Exception"), plan);
    }

    @Test
    @DisplayName("an expression with no kernel is reported as such, not as a missing provider")
    void inexpressibleExpressionIsReportedSeparately() {
        TableEnvironment tEnv =
                TableEnvironment.create(EnvironmentSettings.newInstance().inBatchMode().build());
        tEnv.executeSql(
                "CREATE TABLE s (word STRING, n INT) WITH ("
                        + "'connector' = 'datagen', 'number-of-rows' = '10')");
        tEnv.getConfig().set(GpuOffloadOptions.ENABLED, true);
        tEnv.getConfig().set(GpuOffloadOptions.MIN_ROW_COST, 1);

        String plan = tEnv.explainSql("SELECT word, n + 1 AS m FROM s WHERE n > 0");

        assertTrue(plan.contains("not expressible on device"), plan);
    }
}
