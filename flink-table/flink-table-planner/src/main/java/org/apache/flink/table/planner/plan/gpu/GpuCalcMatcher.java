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

import org.apache.flink.table.runtime.gpu.GpuCalcSpec;
import org.apache.flink.table.types.logical.RowType;

import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexLiteral;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.sql.SqlKind;

import javax.annotation.Nullable;

import java.math.BigDecimal;
import java.util.List;
import java.util.Optional;

/**
 * Matches a Calc's expressions against the runtime's kernel catalogue.
 *
 * <p>Clearing the cost floor says an expression is <em>worth</em> offloading; it does not say the
 * runtime has a kernel for it. Today the catalogue holds one shape — {@code col * mul + add} with
 * an optional {@code col > threshold} selection — so almost everything that clears the floor will
 * fail to match here and fall back to the CPU.
 *
 * <p>That asymmetry is temporary but it is the honest state of things, and the gap between "worth
 * offloading" and "can be offloaded" is exactly what a code-generating backend would close. Keeping
 * the two checks separate means the measurement of the first is not contaminated by the limits of
 * the second.
 */
public class GpuCalcMatcher {

    private GpuCalcMatcher() {}

    /**
     * @param projection the Calc's projections, already expanded from program form
     * @param condition the Calc's condition, or null
     * @param outputType the Calc's output row type
     * @param rowCost the cost the gate already computed, carried into the spec
     * @param batchSize rows to stage per kernel launch
     */
    public static Optional<GpuCalcSpec> match(
            List<RexNode> projection,
            @Nullable RexNode condition,
            RowType outputType,
            int rowCost,
            int batchSize) {

        // The kernel writes exactly one computed column; pass-through columns alongside it are
        // served by the operator from its staging buffer without going near the device.
        RexNode computed = null;
        int[] layout = new int[projection.size()];
        for (int i = 0; i < projection.size(); i++) {
            RexNode expr = projection.get(i);
            if (expr instanceof RexInputRef) {
                layout[i] = ((RexInputRef) expr).getIndex();
                continue;
            }
            if (computed != null) {
                return Optional.empty();
            }
            computed = expr;
            layout[i] = GpuCalcSpec.COMPUTED;
        }
        if (computed == null) {
            return Optional.empty();
        }

        Scale scale = matchScale(computed);
        if (scale == null) {
            return Optional.empty();
        }

        Double threshold = null;
        if (condition != null) {
            threshold = matchThreshold(condition, scale.fieldIndex);
            if (threshold == null) {
                return Optional.empty();
            }
        }
        return Optional.of(
                new GpuCalcSpec(
                        scale.fieldIndex,
                        scale.mul,
                        scale.add,
                        threshold,
                        outputType,
                        layout,
                        rowCost,
                        batchSize));
    }

    private static final class Scale {
        final int fieldIndex;
        final double mul;
        final double add;

        Scale(int fieldIndex, double mul, double add) {
            this.fieldIndex = fieldIndex;
            this.mul = mul;
            this.add = add;
        }
    }

    /** Matches {@code col * mul + add}, {@code col * mul}, or a bare {@code col}. */
    private static @Nullable Scale matchScale(RexNode node) {
        if (node.getKind() == SqlKind.PLUS) {
            List<RexNode> operands = ((RexCall) node).getOperands();
            Double add = constantOf(operands.get(1));
            if (add == null) {
                return null;
            }
            Scale inner = matchScale(operands.get(0));
            return inner == null ? null : new Scale(inner.fieldIndex, inner.mul, inner.add + add);
        }
        if (node.getKind() == SqlKind.TIMES) {
            List<RexNode> operands = ((RexCall) node).getOperands();
            Double mul = constantOf(operands.get(1));
            if (mul == null) {
                return null;
            }
            Scale inner = matchScale(operands.get(0));
            // Folding a multiply into an inner add would change the arithmetic, so only a pure
            // column times a constant is accepted here.
            return inner == null || inner.add != 0.0 || inner.mul != 1.0
                    ? null
                    : new Scale(inner.fieldIndex, mul, 0.0);
        }
        if (node instanceof RexInputRef) {
            return new Scale(((RexInputRef) node).getIndex(), 1.0, 0.0);
        }
        return null;
    }

    /** Matches {@code col > threshold} on the same column the projection reads. */
    private static @Nullable Double matchThreshold(RexNode condition, int fieldIndex) {
        if (condition.getKind() != SqlKind.GREATER_THAN) {
            return null;
        }
        List<RexNode> operands = ((RexCall) condition).getOperands();
        if (!(operands.get(0) instanceof RexInputRef)
                || ((RexInputRef) operands.get(0)).getIndex() != fieldIndex) {
            return null;
        }
        return constantOf(operands.get(1));
    }

    /** A numeric literal as a double, or null if the node is not a numeric constant. */
    private static @Nullable Double constantOf(RexNode node) {
        if (!(node instanceof RexLiteral)) {
            return null;
        }
        Object value = ((RexLiteral) node).getValue3();
        if (value instanceof BigDecimal) {
            return ((BigDecimal) value).doubleValue();
        }
        if (value instanceof Number) {
            return ((Number) value).doubleValue();
        }
        return null;
    }
}
