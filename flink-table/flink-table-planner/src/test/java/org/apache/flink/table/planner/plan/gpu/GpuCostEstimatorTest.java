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

import org.apache.calcite.jdbc.JavaTypeFactoryImpl;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.sql.type.SqlTypeName;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Costing is exercised against real Calcite {@link RexNode} trees built with a {@link RexBuilder} —
 * the same classes the Flink planner hands to an {@code ExecNode} — rather than against a stand-in.
 */
class GpuCostEstimatorTest {

    private final RelDataTypeFactory types = new JavaTypeFactoryImpl();
    private final RexBuilder rex = new RexBuilder(types);

    private RelDataType t(SqlTypeName name) {
        return types.createSqlType(name);
    }

    private RexNode col(SqlTypeName name, int index) {
        return rex.makeInputRef(t(name), index);
    }

    private RexNode lit(double value) {
        return rex.makeLiteral(BigDecimal.valueOf(value), t(SqlTypeName.DOUBLE));
    }

    @Test
    @DisplayName("a BIGINT operand is refused, matching what the generator will accept")
    void bigintOperandIsRefused() {
        // The staging buffers are DoubleArray, so a long beyond 2^53 would come back changed.
        RexNode expr =
                rex.makeCall(SqlStdOperatorTable.MULTIPLY, col(SqlTypeName.BIGINT, 0), lit(2.0));

        RowCost cost = GpuCostEstimator.estimate(expr);

        assertFalse(cost.isEligible(), "estimator and generator must agree about BIGINT");
        assertTrue(cost.rejection().contains("BIGINT operand"), cost::rejection);
    }

    @Test
    @DisplayName("a BIGINT column projected through is still fine: it never reaches the device")
    void bigintPassThroughIsAllowed() {
        RowCost cost = GpuCostEstimator.estimate(col(SqlTypeName.BIGINT, 0));

        assertTrue(cost.isEligible(), "a projected-through column keeps its exact type");
    }

    @Test
    @DisplayName("the original target query is eligible but far below any plausible floor")
    void targetQueryIsCheap() {
        RexNode val = col(SqlTypeName.DOUBLE, 1);
        // val * 2.0 + 1.0
        RexNode projection =
                rex.makeCall(
                        SqlStdOperatorTable.PLUS,
                        rex.makeCall(SqlStdOperatorTable.MULTIPLY, val, lit(2.0)),
                        lit(1.0));
        // val > 0.5
        RexNode condition = rex.makeCall(SqlStdOperatorTable.GREATER_THAN, val, lit(0.5));

        RowCost cost = GpuCostEstimator.estimateAll(List.of(projection, condition));

        assertTrue(cost.isEligible(), cost::rejection);
        // one multiply, one add, one comparison
        assertEquals(3, cost.weight());
        // Measured at 0.54x against CPU. Whatever the calibrated floor turns out to be, it is far
        // above this; the test asserts the estimator puts the query in the "obviously not worth it"
        // regime rather than pinning a specific floor.
        assertTrue(
                cost.weight() < 24,
                "target query must land below the lowest measured losing intensity");
    }

    @Test
    @DisplayName("a transcendental-heavy expression clears the measured crossover band")
    void heavyExpressionIsExpensive() {
        RexNode val = col(SqlTypeName.DOUBLE, 1);
        // EXP(val) * LN(val) + SQRT(val) / POWER(val, 2.0)
        RexNode expTerm =
                rex.makeCall(
                        SqlStdOperatorTable.MULTIPLY,
                        rex.makeCall(SqlStdOperatorTable.EXP, val),
                        rex.makeCall(SqlStdOperatorTable.LN, val));
        RexNode sqrtTerm =
                rex.makeCall(
                        SqlStdOperatorTable.DIVIDE,
                        rex.makeCall(SqlStdOperatorTable.SQRT, val),
                        rex.makeCall(SqlStdOperatorTable.POWER, val, lit(2.0)));
        RexNode projection = rex.makeCall(SqlStdOperatorTable.PLUS, expTerm, sqrtTerm);

        RowCost cost = GpuCostEstimator.estimateAll(List.of(projection));

        assertTrue(cost.isEligible(), cost::rejection);
        // EXP 20 + LN 20 + TIMES 1 + SQRT 8 + POWER 24 + DIVIDE 4 + PLUS 1
        assertEquals(78, cost.weight());
        assertTrue(cost.weight() > 24, "must clear the low end of the measured crossover band");
    }

    @Test
    @DisplayName("DECIMAL is rejected, naming the type")
    void decimalIsRejected() {
        RelDataType decimal = types.createSqlType(SqlTypeName.DECIMAL, 7, 2);
        RexNode price = rex.makeInputRef(decimal, 0);
        RexNode projection = rex.makeCall(SqlStdOperatorTable.MULTIPLY, price, price);

        RowCost cost = GpuCostEstimator.estimate(projection);

        assertFalse(cost.isEligible());
        assertTrue(cost.rejection().contains("DECIMAL"), cost::rejection);
    }

    @Test
    @DisplayName("VARCHAR is rejected")
    void varcharIsRejected() {
        RexNode word = rex.makeInputRef(types.createSqlType(SqlTypeName.VARCHAR, 100), 0);
        RowCost cost = GpuCostEstimator.estimate(word);

        assertFalse(cost.isEligible());
        assertTrue(cost.rejection().contains("unsupported column type"), cost::rejection);
    }

    @Test
    @DisplayName("an unknown function is refused rather than given a default weight")
    void unknownFunctionIsRejected() {
        RexNode val = col(SqlTypeName.DOUBLE, 0);
        RexNode call = rex.makeCall(SqlStdOperatorTable.RAND, val);

        RowCost cost = GpuCostEstimator.estimate(call);

        assertFalse(cost.isEligible());
        assertTrue(cost.rejection().contains("no kernel for operator"), cost::rejection);
    }

    @Test
    @DisplayName("ineligibility is sticky and keeps the first reason")
    void firstRejectionWins() {
        RowCost first = RowCost.ineligible("first");
        RowCost second = RowCost.ineligible("second");

        assertEquals("first", first.plus(second).rejection());
        assertEquals("first", first.plus(RowCost.of(10)).rejection());
        assertEquals("second", RowCost.of(10).plus(second).rejection());
    }

    @Test
    @DisplayName("widening casts are free; narrowing casts are refused")
    void castRules() {
        RexNode intCol = col(SqlTypeName.INTEGER, 0);
        RexNode widening = rex.makeCast(t(SqlTypeName.DOUBLE), intCol);
        RowCost wide = GpuCostEstimator.estimate(widening);
        assertTrue(wide.isEligible(), wide::rejection);
        assertEquals(0, wide.weight(), "a widening numeric cast costs nothing on device");

        RexNode doubleCol = col(SqlTypeName.DOUBLE, 0);
        RexNode narrowing = rex.makeCast(t(SqlTypeName.INTEGER), doubleCol);
        RowCost narrow = GpuCostEstimator.estimate(narrowing);
        // A narrowing cast must raise on overflow, and a device kernel cannot raise.
        assertFalse(narrow.isEligible(), "narrowing cast must be refused");
    }

    @Test
    @DisplayName("costs sum across a subtree, which is how fusion clears the floor")
    void costsSumAcrossExpressions() {
        RexNode val = col(SqlTypeName.DOUBLE, 0);
        RexNode sqrt = rex.makeCall(SqlStdOperatorTable.SQRT, val);

        RowCost one = GpuCostEstimator.estimate(sqrt);
        RowCost three = GpuCostEstimator.estimateAll(List.of(sqrt, sqrt, sqrt));

        assertEquals(8, one.weight());
        assertEquals(24, three.weight());
    }
}
