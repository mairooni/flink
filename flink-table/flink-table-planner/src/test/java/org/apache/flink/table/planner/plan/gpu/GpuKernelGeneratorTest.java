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

import org.apache.flink.table.runtime.gpu.GpuKernelSource;

import org.apache.calcite.jdbc.JavaTypeFactoryImpl;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.sql.SqlFunction;
import org.apache.calcite.sql.SqlFunctionCategory;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.sql.type.OperandTypes;
import org.apache.calcite.sql.type.ReturnTypes;
import org.apache.calcite.sql.type.SqlTypeName;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Generation of kernel source from real Calcite {@link RexNode} trees. */
class GpuKernelGeneratorTest {

    private final RelDataTypeFactory types = new JavaTypeFactoryImpl();
    private final RexBuilder rex = new RexBuilder(types);

    /**
     * Stands in for Flink's {@code LEAST}, which is a {@code BuiltInFunctionDefinition} reached
     * through a bridging operator rather than anything in Calcite's standard table. The generator
     * dispatches on the operator's name, so a plain function of the same name exercises the same
     * path without dragging the bridging machinery into a unit test.
     */
    private static SqlFunction named(String name) {
        return new SqlFunction(
                name,
                SqlKind.OTHER_FUNCTION,
                ReturnTypes.ARG0,
                null,
                OperandTypes.VARIADIC,
                SqlFunctionCategory.NUMERIC);
    }

    private RexNode col(int index) {
        return rex.makeInputRef(types.createSqlType(SqlTypeName.DOUBLE), index);
    }

    private RexNode lit(double v) {
        return rex.makeLiteral(BigDecimal.valueOf(v), types.createSqlType(SqlTypeName.DOUBLE));
    }

    private GpuKernelSource generate(List<RexNode> projection, RexNode condition) {
        Optional<GpuKernelSource> kernel = GpuKernelGenerator.generate(projection, condition, "T");
        assertTrue(kernel.isPresent(), "expected the expressions to be expressible");
        return kernel.get();
    }

    @Test
    @DisplayName("the shape the old catalogue matched still generates")
    void simpleProjectionAndFilter() {
        RexNode projection =
                rex.makeCall(
                        SqlStdOperatorTable.PLUS,
                        rex.makeCall(SqlStdOperatorTable.MULTIPLY, col(1), lit(2.0)),
                        lit(1.0));
        RexNode condition = rex.makeCall(SqlStdOperatorTable.GREATER_THAN, col(1), lit(0.5));

        GpuKernelSource kernel = generate(Arrays.asList(col(0), projection), condition);

        assertArrayEquals(
                new int[] {1},
                kernel.inputFieldIndexes(),
                "only the referenced column is staged; field 0 is projected through");
        assertEquals(1, kernel.outputCount());
        assertTrue(kernel.hasFilter());
        assertTrue(kernel.source().contains("out0.set(i, ((c1 * 2.0) + 1.0));"), kernel.source());
        assertTrue(kernel.source().contains("if ((c1 > 0.5))"), kernel.source());
    }

    @Test
    @DisplayName("LEAST folds into nested binary minima")
    void leastFoldsLeft() {
        // LEAST(c0, c1, 2.0) -- n-ary in SQL, and TornadoMath.min takes two
        RexNode expr = rex.makeCall(named("LEAST"), Arrays.asList(col(0), col(1), lit(2.0)));

        GpuKernelSource kernel = generate(Collections.singletonList(expr), null);

        assertTrue(
                kernel.source()
                        .contains("out0.set(i, TornadoMath.min(TornadoMath.min(c0, c1), 2.0));"),
                kernel.source());
    }

    @Test
    @DisplayName("GREATEST folds the same way, onto max")
    void greatestFoldsLeft() {
        RexNode expr = rex.makeCall(named("GREATEST"), Arrays.asList(col(0), col(1)));

        GpuKernelSource kernel = generate(Collections.singletonList(expr), null);

        assertTrue(kernel.source().contains("TornadoMath.max(c0, c1)"), kernel.source());
    }

    @Test
    @DisplayName("an expression the catalogue could never hold")
    void transcendentalExpression() {
        // EXP(c0) * LN(c0) + SIN(c0) * COS(c0)
        RexNode expr =
                rex.makeCall(
                        SqlStdOperatorTable.PLUS,
                        rex.makeCall(
                                SqlStdOperatorTable.MULTIPLY,
                                rex.makeCall(SqlStdOperatorTable.EXP, col(0)),
                                rex.makeCall(SqlStdOperatorTable.LN, col(0))),
                        rex.makeCall(
                                SqlStdOperatorTable.MULTIPLY,
                                rex.makeCall(SqlStdOperatorTable.SIN, col(0)),
                                rex.makeCall(SqlStdOperatorTable.COS, col(0))));

        GpuKernelSource kernel = generate(Collections.singletonList(expr), null);

        assertFalse(kernel.hasFilter());
        assertTrue(kernel.source().contains("TornadoMath.exp(c0)"), kernel.source());
        assertTrue(kernel.source().contains("TornadoMath.log(c0)"), kernel.source());
        assertTrue(kernel.source().contains("TornadoMath.sin(c0)"), kernel.source());
        assertTrue(kernel.source().contains("TornadoMath.cos(c0)"), kernel.source());
    }

    @Test
    @DisplayName("several columns and several outputs")
    void multipleInputsAndOutputs() {
        RexNode first = rex.makeCall(SqlStdOperatorTable.MULTIPLY, col(0), col(2));
        RexNode second = rex.makeCall(SqlStdOperatorTable.DIVIDE, col(2), lit(4.0));

        GpuKernelSource kernel = generate(Arrays.asList(first, second), null);

        assertArrayEquals(new int[] {0, 2}, kernel.inputFieldIndexes());
        assertEquals(2, kernel.outputCount());
        assertTrue(kernel.source().contains("DoubleArray out0, DoubleArray out1"), kernel.source());
        assertTrue(kernel.source().contains("double c2 = c2_in.get(i);"), kernel.source());
    }

    @Test
    @DisplayName("a filter on a different column than the projection is fine now")
    void filterOnAnotherColumn() {
        RexNode projection = rex.makeCall(SqlStdOperatorTable.MULTIPLY, col(1), lit(2.0));
        RexNode condition = rex.makeCall(SqlStdOperatorTable.LESS_THAN, col(3), lit(0.5));

        GpuKernelSource kernel = generate(Collections.singletonList(projection), condition);

        assertArrayEquals(new int[] {1, 3}, kernel.inputFieldIndexes());
        assertTrue(kernel.source().contains("if ((c3 < 0.5))"), kernel.source());
    }

    @Test
    @DisplayName("BIGINT is refused as an expression input: it does not survive a double")
    void bigintInputRefused() {
        RexNode id = rex.makeInputRef(types.createSqlType(SqlTypeName.BIGINT), 0);
        RexNode expr = rex.makeCall(SqlStdOperatorTable.MULTIPLY, id, id);

        assertFalse(
                GpuKernelGenerator.generate(Collections.singletonList(expr), null, "T").isPresent(),
                "values above 2^53 would differ from the CPU plan");
    }

    @Test
    @DisplayName("an unknown function is refused rather than guessed at")
    void unknownFunctionRefused() {
        RexNode call = rex.makeCall(SqlStdOperatorTable.RAND, col(0));
        assertFalse(
                GpuKernelGenerator.generate(Collections.singletonList(call), null, "T")
                        .isPresent());
    }

    @Test
    @DisplayName("a projection of only pass-through columns has nothing to offload")
    void passThroughOnly() {
        assertFalse(
                GpuKernelGenerator.generate(Arrays.asList(col(0), col(1)), null, "T").isPresent());
    }

    @Test
    @DisplayName("the emitted unit has the shape the runtime expects")
    void generatedUnitIsWellFormed() {
        RexNode expr =
                rex.makeCall(
                        SqlStdOperatorTable.PLUS,
                        rex.makeCall(SqlStdOperatorTable.SQRT, col(0)),
                        rex.makeCall(SqlStdOperatorTable.POWER, col(1), lit(3.0)));
        RexNode condition = rex.makeCall(SqlStdOperatorTable.GREATER_THAN, col(0), lit(0.0));

        GpuKernelSource kernel = generate(Collections.singletonList(expr), condition);
        String source = kernel.source();

        // This module cannot compile the source: doing so needs tornado-api, and the planner
        // deliberately has no dependency on it -- that separation is what lets the offload runtime
        // be absent. Compiling and running generated kernels is covered in the runtime module,
        // which has TornadoVM but not Calcite. Here the checks are structural.
        assertTrue(source.contains("public final class " + kernel.className()), source);
        assertTrue(source.contains("public static void " + kernel.methodName() + "("), source);
        assertTrue(source.contains("for (@Parallel int i = 0;"), source);
        assertTrue(source.contains("IntArray mask"), source);
        assertEquals(
                countOccurrences(source, "{"), countOccurrences(source, "}"), "unbalanced braces");
        assertEquals(
                countOccurrences(source, "("), countOccurrences(source, ")"), "unbalanced parens");
    }

    private static int countOccurrences(String text, String needle) {
        int count = 0;
        int at = text.indexOf(needle);
        while (at >= 0) {
            count++;
            at = text.indexOf(needle, at + 1);
        }
        return count;
    }
}
