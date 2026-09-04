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

import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexCorrelVariable;
import org.apache.calcite.rex.RexDynamicParam;
import org.apache.calcite.rex.RexFieldAccess;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexLiteral;
import org.apache.calcite.rex.RexLocalRef;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexOver;
import org.apache.calcite.rex.RexPatternFieldRef;
import org.apache.calcite.rex.RexRangeRef;
import org.apache.calcite.rex.RexSubQuery;
import org.apache.calcite.rex.RexTableInputRef;
import org.apache.calcite.rex.RexVisitor;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.type.SqlTypeName;

import java.util.List;

/**
 * Decides, at plan time, whether an expression can run as a device kernel and how much arithmetic
 * it does per row.
 *
 * <p><b>Nothing is executed.</b> This is a static walk over the {@code RexNode} tree the planner
 * already holds after optimisation, before any {@code Transformation} is built and long before a
 * record is read. Cost is counted in weighted operations per row (see {@link OperatorWeights}), and
 * compared against a floor calibrated once per device.
 *
 * <p>Why a floor is needed at all: a predicate that only checks types admits {@code val * 2.0 +
 * 1.0}, which measures 0.54× against CPU — the offload would make such queries roughly twice as
 * slow, silently. Type eligibility is necessary and nowhere near sufficient.
 *
 * <h2>Common subexpressions are counted more than once</h2>
 *
 * <p>{@code BatchPhysicalCalc} builds its expressions with {@code RexProgram.expandLocalRef}, which
 * inlines shared subexpressions, so {@code SQRT(x) + SQRT(x)} arrives as two {@code SQRT} nodes and
 * is costed as two. A kernel generator that recognised the common subexpression would do one. The
 * estimate therefore runs slightly high on expression-sharing queries, which biases toward
 * offloading — the less safe direction. It is left uncorrected because the effect is small next to
 * the order of magnitude the floor has to resolve, and because the current kernel library does no
 * CSE either, so the estimate matches what actually executes.
 *
 * <h2>Rejection is the safe direction</h2>
 *
 * <p>Every unrecognised node, operator or type is refused. A false rejection costs a missed
 * speedup; a false acceptance produces either a wrong answer or a regression, and both are worse.
 * Rejection reasons are carried in the {@link RowCost} so they can be surfaced in {@code EXPLAIN}
 * rather than leaving a user wondering why a query did not offload.
 */
public final class GpuCostEstimator implements RexVisitor<RowCost> {

    /**
     * Types the staging buffers and kernels handle. Deliberately narrow.
     *
     * <p>{@code DECIMAL} is {@code DecimalData}, an object; {@code VARCHAR} is variable-length
     * {@code BinaryStringData} backed by a memory segment. Neither has a fixed-width columnar
     * representation the gather can produce, so both are excluded rather than costed.
     */
    private static boolean isSupportedType(RelDataType type) {
        switch (type.getSqlTypeName()) {
            case INTEGER:
            case BIGINT:
            case FLOAT:
            case DOUBLE:
            case REAL:
            case BOOLEAN:
                return true;
            default:
                return false;
        }
    }

    /** Costs one expression tree. */
    public static RowCost estimate(RexNode node) {
        return node.accept(new GpuCostEstimator());
    }

    /**
     * Costs a set of expressions — a Calc's projections plus its condition — as one unit.
     *
     * <p>Each expression's own result type is checked here, because that is what the staging
     * buffers must be able to hold. Interior types are checked as the tree is walked; leaf literals
     * are not (see {@link #visitLiteral}).
     */
    public static RowCost estimateAll(List<? extends RexNode> nodes) {
        RowCost total = RowCost.ZERO;
        for (RexNode node : nodes) {
            if (!isSupportedType(node.getType())) {
                return RowCost.ineligible("unsupported output type " + node.getType());
            }
            total = total.plus(estimate(node));
            if (!total.isEligible()) {
                return total;
            }
        }
        return total;
    }

    @Override
    public RowCost visitInputRef(RexInputRef ref) {
        return isSupportedType(ref.getType())
                ? RowCost.of(OperatorWeights.LEAF)
                : RowCost.ineligible("unsupported column type " + ref.getType());
    }

    @Override
    public RowCost visitLiteral(RexLiteral literal) {
        // A literal becomes a kernel scalar argument, so it costs nothing per row.
        //
        // Its own type is deliberately NOT checked. Calcite parses an unsuffixed SQL decimal such
        // as `2.0` or `0.5` as DECIMAL(2,1) even inside an all-DOUBLE expression, so rejecting
        // literals by type refuses `val * 2.0 + 1.0 WHERE val > 0.5` -- which is to say, nearly
        // every query anyone writes. The literal's value is folded into the kernel as a scalar
        // argument of the enclosing call's type, and that call's result type is checked in
        // visitCall, as is each top-level expression's in estimateAll. A DECIMAL literal reaching
        // a context that stays DECIMAL is therefore still rejected -- by the surrounding type,
        // which is the thing that actually determines what the kernel computes.
        //
        // Where the enclosing type is DOUBLE, Flink's own generated code narrows the literal the
        // same way, so matching it is agreement with the CPU path rather than a new rounding risk.
        return RowCost.of(OperatorWeights.LEAF);
    }

    @Override
    public RowCost visitCall(RexCall call) {
        if (!isSupportedType(call.getType())) {
            return RowCost.ineligible(
                    "unsupported result type "
                            + call.getType()
                            + " from "
                            + call.getOperator().getName());
        }

        Integer weight = weightOf(call);
        if (weight == null) {
            return RowCost.ineligible("no kernel for operator " + call.getOperator().getName());
        }
        if (call.getKind() == SqlKind.CAST && !isSafeCast(call)) {
            return RowCost.ineligible("unsupported cast to " + call.getType());
        }

        RowCost total = RowCost.of(weight);
        for (RexNode operand : call.getOperands()) {
            // BIGINT is admitted as a column but refused as an operand, matching what the
            // generator will accept. Staging is DoubleArray, and a long beyond 2^53 does not
            // survive that round trip, so an expression over one would quietly disagree with the
            // CPU plan for large keys. Projecting such a column through is still fine: it never
            // reaches the device. Costing it here as eligible and having generation refuse it
            // later showed up in EXPLAIN as an unexplained fallback.
            if (operand.getType().getSqlTypeName() == SqlTypeName.BIGINT) {
                return RowCost.ineligible(
                        "BIGINT operand of "
                                + call.getOperator().getName()
                                + ": values beyond 2^53 would not survive the double staging "
                                + "buffers, so the device could disagree with the CPU plan");
            }
            total = total.plus(operand.accept(this));
            if (!total.isEligible()) {
                return total;
            }
        }
        return total;
    }

    private static Integer weightOf(RexCall call) {
        Integer byKind = OperatorWeights.forKind(call.getKind());
        if (byKind != null) {
            return byKind;
        }
        // Named scalar functions arrive as OTHER_FUNCTION with the name on the operator.
        return OperatorWeights.forFunction(
                call.getOperator().getName().toUpperCase(java.util.Locale.ROOT));
    }

    /**
     * Only casts between the supported numeric types are allowed, and only where the target holds
     * the source. A narrowing cast has overflow semantics the kernel does not implement, and SQL
     * requires an error rather than a wrap — which a device kernel cannot raise (TornadoVM has no
     * exception support).
     */
    private static boolean isSafeCast(RexCall call) {
        SqlTypeName from = call.getOperands().get(0).getType().getSqlTypeName();
        SqlTypeName to = call.getType().getSqlTypeName();
        if (from == to) {
            return true;
        }
        switch (from) {
            case INTEGER:
                return to == SqlTypeName.BIGINT
                        || to == SqlTypeName.DOUBLE
                        || to == SqlTypeName.FLOAT
                        || to == SqlTypeName.REAL;
            case BIGINT:
                return to == SqlTypeName.DOUBLE;
            case FLOAT:
            case REAL:
                return to == SqlTypeName.DOUBLE;
            default:
                return false;
        }
    }

    // --- Everything below has no device representation. ---

    @Override
    public RowCost visitLocalRef(RexLocalRef ref) {
        // Calc uses RexLocalRef into a common sub-expression program. Supporting it means costing
        // the program rather than the tree, which is a separate piece of work; refuse for now
        // rather than silently costing a reference as free.
        return RowCost.ineligible("RexLocalRef (Calc program form) not yet costed");
    }

    @Override
    public RowCost visitOver(RexOver over) {
        return RowCost.ineligible("window function " + over.getOperator().getName());
    }

    @Override
    public RowCost visitCorrelVariable(RexCorrelVariable v) {
        return RowCost.ineligible("correlation variable");
    }

    @Override
    public RowCost visitDynamicParam(RexDynamicParam p) {
        return RowCost.ineligible("dynamic parameter");
    }

    @Override
    public RowCost visitRangeRef(RexRangeRef r) {
        return RowCost.ineligible("range reference");
    }

    @Override
    public RowCost visitFieldAccess(RexFieldAccess a) {
        return RowCost.ineligible("nested field access");
    }

    @Override
    public RowCost visitSubQuery(RexSubQuery q) {
        return RowCost.ineligible("subquery");
    }

    @Override
    public RowCost visitTableInputRef(RexTableInputRef r) {
        return RowCost.ineligible("table input reference");
    }

    @Override
    public RowCost visitPatternFieldRef(RexPatternFieldRef r) {
        return RowCost.ineligible("MATCH_RECOGNIZE pattern reference");
    }
}
