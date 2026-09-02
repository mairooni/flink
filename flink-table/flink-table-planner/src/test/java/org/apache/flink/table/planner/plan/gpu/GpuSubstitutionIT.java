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
import org.apache.flink.table.types.logical.DoubleType;
import org.apache.flink.table.types.logical.RowType;

import org.apache.calcite.jdbc.JavaTypeFactoryImpl;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.sql.type.SqlTypeName;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.util.Arrays;
import java.util.Collections;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The substitution seam: kernel matching, and the fallback when no provider is installed. */
class GpuSubstitutionIT {

    private final RelDataTypeFactory types = new JavaTypeFactoryImpl();
    private final RexBuilder rex = new RexBuilder(types);
    private final RowType outputType =
            RowType.of(new DoubleType());

    private RexNode col(int index) {
        return rex.makeInputRef(types.createSqlType(SqlTypeName.DOUBLE), index);
    }

    private RexNode lit(double v) {
        return rex.makeLiteral(BigDecimal.valueOf(v), types.createSqlType(SqlTypeName.DOUBLE));
    }

    @Test
    @DisplayName("matches col * mul + add with a > filter")
    void matchesTheKernelShape() {
        RexNode projection =
                rex.makeCall(SqlStdOperatorTable.PLUS,
                        rex.makeCall(SqlStdOperatorTable.MULTIPLY, col(1), lit(2.0)), lit(1.0));
        RexNode condition = rex.makeCall(SqlStdOperatorTable.GREATER_THAN, col(1), lit(0.5));

        Optional<GpuCalcSpec> spec =
                GpuCalcMatcher.match(
                        Arrays.asList(col(0), projection), condition, outputType, 3);

        assertTrue(spec.isPresent());
        assertEquals(1, spec.get().inputFieldIndex());
        assertEquals(2.0, spec.get().mul());
        assertEquals(1.0, spec.get().add());
        assertEquals(0.5, spec.get().threshold());
    }

    @Test
    @DisplayName("no condition means no threshold, not a failed match")
    void matchesWithoutFilter() {
        RexNode projection = rex.makeCall(SqlStdOperatorTable.MULTIPLY, col(0), lit(3.0));

        Optional<GpuCalcSpec> spec =
                GpuCalcMatcher.match(
                        Collections.singletonList(projection), null, outputType, 1);

        assertTrue(spec.isPresent());
        assertEquals(3.0, spec.get().mul());
        assertEquals(0.0, spec.get().add());
        assertNull(spec.get().threshold());
    }

    @Test
    @DisplayName("an expression outside the catalogue does not match, even though it clears the floor")
    void heavyExpressionHasNoKernel() {
        RexNode heavy =
                rex.makeCall(SqlStdOperatorTable.MULTIPLY,
                        rex.makeCall(SqlStdOperatorTable.EXP, col(0)),
                        rex.makeCall(SqlStdOperatorTable.LN, col(0)));

        assertFalse(GpuCalcMatcher.match(
                        Collections.singletonList(heavy), null, outputType, 40).isPresent(),
                "clearing the cost floor does not imply a kernel exists");
    }

    @Test
    @DisplayName("a filter on a different column than the projection does not match")
    void mismatchedFilterColumn() {
        RexNode projection = rex.makeCall(SqlStdOperatorTable.MULTIPLY, col(1), lit(2.0));
        RexNode condition = rex.makeCall(SqlStdOperatorTable.GREATER_THAN, col(0), lit(0.5));

        assertFalse(GpuCalcMatcher.match(
                Collections.singletonList(projection), condition, outputType, 2).isPresent());
    }

    @Test
    @DisplayName("with no provider installed, a selected node falls back and EXPLAIN says so")
    void fallsBackWithoutProvider() {
        TableEnvironment tEnv =
                TableEnvironment.create(EnvironmentSettings.newInstance().inBatchMode().build());
        tEnv.executeSql(
                "CREATE TABLE t (id BIGINT, val DOUBLE) WITH ("
                        + "'connector' = 'datagen', 'number-of-rows' = '10')");
        tEnv.getConfig().set(GpuOffloadOptions.ENABLED, true);
        // Floor of 1 so the simple kernel-shaped query is selected by the gate.
        tEnv.getConfig().set(GpuOffloadOptions.MIN_ROW_COST, 1);

        String plan = tEnv.explainSql("SELECT id, val * 2.0 + 1.0 AS scaled FROM t WHERE val > 0.5");

        // Planning must succeed with no provider present, and the fallback must be reported
        // rather than swallowed.
        assertTrue(plan.contains("== GPU Offload =="), plan);
        assertTrue(plan.contains("selected but fell back"), plan);
        assertTrue(plan.contains("no GpuOperatorFactoryProvider on the classpath"), plan);
        assertFalse(plan.contains("Exception"), plan);
    }
}
