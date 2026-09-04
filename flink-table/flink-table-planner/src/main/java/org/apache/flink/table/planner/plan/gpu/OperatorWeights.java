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

import org.apache.calcite.sql.SqlKind;

import java.util.Map;

/**
 * Per-operator work weights, in units of one multiply.
 *
 * <h2>Why these numbers do not need to be accurate</h2>
 *
 * <p>The estimate exists to answer one question: is this expression closer to two operations or to
 * forty? The measured crossover on an RTX 4070 sits between intensity 4 and 16 of the calibration
 * kernel — that is, between <b>24 and 96 raw operations per row</b>, and nearer the low end
 * (intensity 4 measured 0.84×, already close to break-even). Separating those two regimes needs
 * roughly one significant figure. Weights tuned finer than that would encode noise, and would also
 * imply a per-device precision the model does not have.
 *
 * <p>Weights are ordinal rather than physical. On a GPU a hardware {@code sqrt} is closer to one
 * instruction than to eight, but what matters for the decision is that an expression full of {@code
 * SQRT} carries more device-eligible work than one full of adds, and that the ratio does not invert
 * between devices.
 *
 * <h2>Absence means rejection</h2>
 *
 * <p>Anything not listed is refused rather than assigned a default. An unknown operator is one the
 * kernel library has not been shown to express, and guessing a weight for it would let an
 * inexpressible expression through the cost gate and fail later at kernel selection.
 */
public final class OperatorWeights {

    /** Reading a column or a constant is free: it is the buffer load the kernel does anyway. */
    public static final int LEAF = 0;

    private static final Map<SqlKind, Integer> BY_KIND =
            Map.ofEntries(
                    // Arithmetic. The unit.
                    Map.entry(SqlKind.PLUS, 1),
                    Map.entry(SqlKind.MINUS, 1),
                    Map.entry(SqlKind.TIMES, 1),
                    Map.entry(SqlKind.MINUS_PREFIX, 1),
                    // Division is not fused and not reciprocal-approximated here.
                    Map.entry(SqlKind.DIVIDE, 4),
                    Map.entry(SqlKind.MOD, 4),
                    // Comparisons and boolean glue: one predicate evaluation each.
                    Map.entry(SqlKind.GREATER_THAN, 1),
                    Map.entry(SqlKind.GREATER_THAN_OR_EQUAL, 1),
                    Map.entry(SqlKind.LESS_THAN, 1),
                    Map.entry(SqlKind.LESS_THAN_OR_EQUAL, 1),
                    Map.entry(SqlKind.EQUALS, 1),
                    Map.entry(SqlKind.NOT_EQUALS, 1),
                    Map.entry(SqlKind.AND, 1),
                    Map.entry(SqlKind.OR, 1),
                    Map.entry(SqlKind.NOT, 1),
                    // A widening numeric cast is free on device; narrowing and non-numeric casts
                    // are
                    // rejected in the estimator before reaching this table.
                    Map.entry(SqlKind.CAST, 0));

    /**
     * Named scalar functions, which Calcite reports as {@link SqlKind#OTHER_FUNCTION} with the name
     * on the operator rather than as a distinct kind.
     */
    private static final Map<String, Integer> BY_NAME =
            Map.ofEntries(
                    Map.entry("ABS", 1),
                    Map.entry("FLOOR", 1),
                    Map.entry("CEIL", 1),
                    Map.entry("ROUND", 2),
                    Map.entry("SIGN", 1),
                    Map.entry("SQRT", 8),
                    Map.entry("EXP", 20),
                    Map.entry("LN", 20),
                    Map.entry("LOG10", 20),
                    Map.entry("LOG2", 20),
                    Map.entry("POWER", 24),
                    Map.entry("SIN", 20),
                    Map.entry("COS", 20),
                    Map.entry("TAN", 20),
                    Map.entry("ASIN", 24),
                    Map.entry("ACOS", 24),
                    Map.entry("ATAN", 24),
                    Map.entry("ATAN2", 24),
                    Map.entry("SINH", 24),
                    Map.entry("COSH", 24),
                    Map.entry("TANH", 24),
                    // One comparison and a select. An n-ary LEAST folds into n-1 of them, but the
                    // operands it is comparing dominate the cost by orders of magnitude.
                    Map.entry("LEAST", 1),
                    Map.entry("GREATEST", 1));

    private OperatorWeights() {}

    /**
     * @return the weight, or null if this operator is not expressible as a kernel
     */
    public static Integer forKind(SqlKind kind) {
        return BY_KIND.get(kind);
    }

    /**
     * @return the weight, or null if this named function is not expressible as a kernel
     */
    public static Integer forFunction(String upperCaseName) {
        return BY_NAME.get(upperCaseName);
    }
}
