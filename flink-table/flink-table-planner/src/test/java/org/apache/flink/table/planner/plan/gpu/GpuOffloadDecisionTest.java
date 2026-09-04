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

class GpuOffloadDecisionTest {

    private final RelDataTypeFactory types = new JavaTypeFactoryImpl();
    private final RexBuilder rex = new RexBuilder(types);

    private RexNode val() {
        return rex.makeInputRef(types.createSqlType(SqlTypeName.DOUBLE), 1);
    }

    private RexNode lit(double v) {
        return rex.makeLiteral(BigDecimal.valueOf(v), types.createSqlType(SqlTypeName.DOUBLE));
    }

    @Test
    @DisplayName("the measured-losing query is rejected, and the reason says why")
    void rejectsTheCheapQuery() {
        RexNode projection =
                rex.makeCall(
                        SqlStdOperatorTable.PLUS,
                        rex.makeCall(SqlStdOperatorTable.MULTIPLY, val(), lit(2.0)),
                        lit(1.0));
        RexNode condition = rex.makeCall(SqlStdOperatorTable.GREATER_THAN, val(), lit(0.5));

        GpuOffloadDecision.Verdict v =
                GpuOffloadDecision.withDefaults().decide(List.of(projection, condition));

        assertFalse(v.offload(), "the 2-flop query measured 0.54x and must not be offloaded");
        assertEquals(3, v.rowCost());
        assertTrue(v.reason().contains("below cost floor"), v::reason);
    }

    /** SQRT(SQRT(...)) nested deeply enough to clear the per-row floor on its own. */
    private RexNode heavy() {
        RexNode expr = val();
        for (int i = 0; i < 20; i++) {
            expr = rex.makeCall(SqlStdOperatorTable.SQRT, expr);
        }
        return expr;
    }

    @Test
    @DisplayName("a heavy expression over too few rows is refused, however heavy")
    void tooFewRowsToRepayCompilation() {
        GpuOffloadDecision.Verdict v =
                GpuOffloadDecision.withDefaults().decide(List.of(heavy()), 100);

        assertFalse(v.offload(), "clearing the per-row floor says nothing about a 100-row table");
        assertTrue(v.reason().contains("below total-work floor"), v::reason);
    }

    @Test
    @DisplayName("the same expression over enough rows is accepted")
    void enoughRowsToRepayCompilation() {
        GpuOffloadDecision.Verdict v =
                GpuOffloadDecision.withDefaults().decide(List.of(heavy()), 10_000_000);

        assertTrue(v.offload(), v::reason);
        assertTrue(v.reason().contains("weighted ops against a floor"), v::reason);
    }

    @Test
    @DisplayName("an unknown row count skips the total-work floor rather than guessing")
    void unknownRowCountSkipsTheSecondGate() {
        GpuOffloadDecision.Verdict v = GpuOffloadDecision.withDefaults().decide(List.of(heavy()));

        assertTrue(v.offload(), v::reason);
        assertTrue(v.reason().contains("row count unknown"), v::reason);
    }

    @Test
    @DisplayName("the per-row floor still rejects first: a cheap expression over many rows")
    void perRowFloorIsNotBoughtOffByVolume() {
        RexNode cheap = rex.makeCall(SqlStdOperatorTable.MULTIPLY, val(), lit(2.0));

        GpuOffloadDecision.Verdict v =
                GpuOffloadDecision.withDefaults().decide(List.of(cheap), 1_000_000_000L);

        assertFalse(v.offload(), "a billion rows does not make a 1-flop expression worth staging");
        assertTrue(v.reason().contains("below cost floor"), v::reason);
    }

    @Test
    @DisplayName("an unambiguously heavy expression is accepted")
    void acceptsHeavyWork() {
        // EXP(val)^2 * LN(val) * SIN(val) * COS(val) * ATAN(val)
        RexNode heavy =
                rex.makeCall(
                        SqlStdOperatorTable.MULTIPLY,
                        rex.makeCall(
                                SqlStdOperatorTable.MULTIPLY,
                                rex.makeCall(
                                        SqlStdOperatorTable.POWER,
                                        rex.makeCall(SqlStdOperatorTable.EXP, val()),
                                        lit(2.0)),
                                rex.makeCall(
                                        SqlStdOperatorTable.MULTIPLY,
                                        rex.makeCall(SqlStdOperatorTable.LN, val()),
                                        rex.makeCall(SqlStdOperatorTable.SIN, val()))),
                        rex.makeCall(
                                SqlStdOperatorTable.MULTIPLY,
                                rex.makeCall(SqlStdOperatorTable.COS, val()),
                                rex.makeCall(SqlStdOperatorTable.ATAN, val())));

        GpuOffloadDecision.Verdict v = GpuOffloadDecision.withDefaults().decide(List.of(heavy));

        assertTrue(v.offload(), v::reason);
        assertTrue(v.rowCost() >= GpuOffloadDecision.DEFAULT_MIN_ROW_COST);
    }

    @Test
    @DisplayName("the conservative floor knowingly rejects expressions above break-even")
    void floorGivesUpWinsBetweenBreakEvenAndFloor() {
        // POWER(EXP(val), 2.0) + LN(val) * SIN(val)  ->  86 weighted ops/row.
        // Above the interpolated break-even of ~70, below the 96 default. Offloading it would
        // probably win, and the default declines anyway. This test exists so that trade-off is
        // visible and deliberate rather than discovered later as a surprise.
        RexNode borderline =
                rex.makeCall(
                        SqlStdOperatorTable.PLUS,
                        rex.makeCall(
                                SqlStdOperatorTable.POWER,
                                rex.makeCall(SqlStdOperatorTable.EXP, val()),
                                lit(2.0)),
                        rex.makeCall(
                                SqlStdOperatorTable.MULTIPLY,
                                rex.makeCall(SqlStdOperatorTable.LN, val()),
                                rex.makeCall(SqlStdOperatorTable.SIN, val())));

        int cost = GpuCostEstimator.estimate(borderline).weight();
        assertEquals(86, cost);
        assertTrue(
                cost > GpuOffloadDecision.MEASURED_BREAK_EVEN,
                "this expression is above measured break-even");

        assertFalse(
                GpuOffloadDecision.withDefaults().decide(List.of(borderline)).offload(),
                "the conservative default declines it anyway");
        assertTrue(
                new GpuOffloadDecision(GpuOffloadDecision.MEASURED_BREAK_EVEN)
                        .decide(List.of(borderline))
                        .offload(),
                "a floor calibrated at break-even would take it");
    }

    @Test
    @DisplayName("ineligible types are refused before the floor is even consulted")
    void ineligibilityBeatsTheFloor() {
        RexNode word = rex.makeInputRef(types.createSqlType(SqlTypeName.VARCHAR, 100), 0);

        GpuOffloadDecision.Verdict v = new GpuOffloadDecision(0).decide(List.of(word));

        assertFalse(v.offload(), "a zero floor must not make an inexpressible type offloadable");
        assertTrue(v.reason().contains("not expressible on device"), v::reason);
    }

    @Test
    @DisplayName("a subtree clears a floor that none of its nodes clears alone")
    void fusionRaisesTheNumerator() {
        // Three nodes at 40 each: individually below a floor of 96, together above it.
        List<RowCost> nodes = List.of(RowCost.of(40), RowCost.of(40), RowCost.of(40));
        GpuOffloadDecision decision = GpuOffloadDecision.withDefaults();

        assertFalse(decision.forSubtree(List.of(RowCost.of(40))).offload());
        assertTrue(decision.forSubtree(nodes).offload());
        assertEquals(120, decision.forSubtree(nodes).rowCost());
    }

    @Test
    @DisplayName("one ineligible node poisons the whole subtree")
    void ineligibleNodePoisonsSubtree() {
        GpuOffloadDecision.Verdict v =
                GpuOffloadDecision.withDefaults()
                        .forSubtree(
                                List.of(
                                        RowCost.of(200),
                                        RowCost.ineligible("VARCHAR"),
                                        RowCost.of(200)));

        assertFalse(v.offload());
        assertTrue(v.reason().contains("VARCHAR"), v::reason);
    }
}
