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

import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexLiteral;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.sql.type.SqlTypeName;

import javax.annotation.Nullable;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * Generates a TornadoVM kernel, as Java source, from a Calc's {@code RexNode} expressions.
 *
 * <p>The generated method is compiled at run time and named to TornadoVM through {@code
 * TaskGraph.task(String, Method, Object...)}. Before that overload existed a kernel had to be a
 * method that already existed at compile time, which forced a fixed catalogue of shapes; this
 * removes that limit.
 *
 * <h2>Everything is a double on the device</h2>
 *
 * <p>Inputs to the expression are staged as {@code DoubleArray} and results are written as doubles.
 * That is exact for the input types admitted here — {@code INTEGER} and {@code FLOAT} are both
 * representable in a double without loss — and it keeps one buffer type rather than a matrix of
 * kernel signatures.
 *
 * <p><b>{@code BIGINT} is deliberately excluded as an expression input.</b> Values beyond 2^53 do
 * not survive the round trip through a double, so admitting it would silently return different
 * answers from the CPU plan for large keys. A {@code BIGINT} column may still be selected: a
 * projected-through column never reaches the device and keeps its exact type.
 *
 * <h2>Filtering does not compact</h2>
 *
 * <p>A condition becomes a 0/1 mask written alongside the projections rather than a compaction.
 * Compacting on the device needs a prefix sum and a scatter — two more kernels and another buffer —
 * whereas the host has to walk the results anyway to build output rows.
 */
public final class GpuKernelGenerator {

    /**
     * Words that are legal Java identifiers but reserved in OpenCL C.
     *
     * <p>TornadoVM translates the generated Java into OpenCL, so a method or variable named after
     * an OpenCL keyword fails late, inside the sketcher, with {@code "Java method name corresponds
     * to an OpenCL Token"}. Generated names are checked against this list rather than left to
     * produce that error.
     */
    private static final Set<String> OPENCL_RESERVED =
            new HashSet<>(
                    Arrays.asList(
                            "kernel",
                            "global",
                            "local",
                            "constant",
                            "private",
                            "read_only",
                            "write_only",
                            "read_write",
                            "uniform",
                            "pipe",
                            "half",
                            "quad",
                            "complex",
                            "imaginary",
                            "generic"));

    private static final String INDENT = "        ";

    private GpuKernelGenerator() {}

    /**
     * Generates a kernel for the given expressions, or empty if any of them cannot be expressed.
     *
     * <p>Returning empty is an ordinary outcome. {@link GpuCostEstimator} decides whether an
     * expression is <em>worth</em> offloading; this decides whether it can be <em>written</em> as a
     * kernel, and the two are independent.
     *
     * @param projection the Calc's projections, already expanded from program form
     * @param condition the Calc's condition, or null
     * @param classNameSuffix appended to the generated class name to keep it unique per query
     */
    public static Optional<GpuKernelSource> generate(
            List<RexNode> projection, @Nullable RexNode condition, String classNameSuffix) {

        // Only referenced columns are staged; a projected-through column stays on the host.
        final Map<Integer, String> inputs = new LinkedHashMap<>();
        final List<String> computed = new ArrayList<>();

        for (RexNode expr : projection) {
            if (expr instanceof RexInputRef) {
                continue;
            }
            if (!isDoubleResult(expr)) {
                return Optional.empty();
            }
            String rendered = render(expr, inputs);
            if (rendered == null) {
                return Optional.empty();
            }
            computed.add(rendered);
        }
        if (computed.isEmpty()) {
            // Nothing to compute; the CPU path already handles pure projection.
            return Optional.empty();
        }

        String renderedCondition = null;
        if (condition != null) {
            renderedCondition = render(condition, inputs);
            if (renderedCondition == null) {
                return Optional.empty();
            }
        }
        if (inputs.isEmpty()) {
            // A kernel with no input column has nothing to size the parallel loop by.
            return Optional.empty();
        }

        String className = "GpuCalcKernel$" + classNameSuffix;
        String methodName = "evaluate";
        if (OPENCL_RESERVED.contains(methodName.toLowerCase(Locale.ROOT))) {
            throw new IllegalStateException(
                    "generated method name collides with an OpenCL keyword");
        }

        int[] inputFields = inputs.keySet().stream().mapToInt(Integer::intValue).toArray();
        String source = renderClass(className, methodName, inputs, computed, renderedCondition);
        return Optional.of(
                new GpuKernelSource(
                        className,
                        methodName,
                        source,
                        inputFields,
                        computed.size(),
                        renderedCondition != null));
    }

    private static String renderClass(
            String className,
            String methodName,
            Map<Integer, String> inputs,
            List<String> computed,
            @Nullable String condition) {

        StringBuilder sb = new StringBuilder();
        sb.append("import uk.ac.manchester.tornado.api.annotations.Parallel;\n");
        sb.append("import uk.ac.manchester.tornado.api.math.TornadoMath;\n");
        sb.append("import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;\n");
        sb.append("import uk.ac.manchester.tornado.api.types.arrays.IntArray;\n\n");
        sb.append("public final class ").append(className).append(" {\n\n");
        sb.append("    public static void ").append(methodName).append("(");

        List<String> params = new ArrayList<>();
        for (String var : inputs.values()) {
            params.add("DoubleArray " + var + "_in");
        }
        for (int i = 0; i < computed.size(); i++) {
            params.add("DoubleArray out" + i);
        }
        if (condition != null) {
            params.add("IntArray mask");
        }
        sb.append(String.join(", ", params)).append(") {\n");

        String first = inputs.values().iterator().next();
        sb.append("        for (@Parallel int i = 0; i < ")
                .append(first)
                .append("_in.getSize(); i++) {\n");
        for (String var : inputs.values()) {
            sb.append(INDENT)
                    .append("    double ")
                    .append(var)
                    .append(" = ")
                    .append(var)
                    .append("_in.get(i);\n");
        }
        for (int i = 0; i < computed.size(); i++) {
            sb.append(INDENT)
                    .append("    out")
                    .append(i)
                    .append(".set(i, ")
                    .append(computed.get(i))
                    .append(");\n");
        }
        if (condition != null) {
            // Written as if/else rather than a ternary: both compile, but this keeps the generated
            // OpenCL readable under -Dtornado.printKernel=True.
            sb.append(INDENT).append("    if (").append(condition).append(") {\n");
            sb.append(INDENT).append("        mask.set(i, 1);\n");
            sb.append(INDENT).append("    } else {\n");
            sb.append(INDENT).append("        mask.set(i, 0);\n");
            sb.append(INDENT).append("    }\n");
        }
        sb.append("        }\n");
        sb.append("    }\n");
        sb.append("}\n");
        return sb.toString();
    }

    /** Renders one expression, registering any column it reads. Null if it cannot be expressed. */
    private static @Nullable String render(RexNode node, Map<Integer, String> inputs) {
        if (node instanceof RexInputRef) {
            RexInputRef ref = (RexInputRef) node;
            if (!isDoubleSafeInput(ref.getType().getSqlTypeName())) {
                return null;
            }
            return inputs.computeIfAbsent(ref.getIndex(), index -> "c" + index);
        }
        if (node instanceof RexLiteral) {
            return renderLiteral((RexLiteral) node);
        }
        if (!(node instanceof RexCall)) {
            return null;
        }
        RexCall call = (RexCall) node;
        List<String> operands = new ArrayList<>();
        for (RexNode operand : call.getOperands()) {
            String rendered = render(operand, inputs);
            if (rendered == null) {
                return null;
            }
            operands.add(rendered);
        }
        return renderCall(call, operands);
    }

    private static @Nullable String renderCall(RexCall call, List<String> operands) {
        switch (call.getKind()) {
            case PLUS:
                return infix(operands, "+");
            case MINUS:
                return infix(operands, "-");
            case TIMES:
                return infix(operands, "*");
            case DIVIDE:
                return infix(operands, "/");
            case MINUS_PREFIX:
                return operands.size() == 1 ? "(-" + operands.get(0) + ")" : null;
            case GREATER_THAN:
                return infix(operands, ">");
            case GREATER_THAN_OR_EQUAL:
                return infix(operands, ">=");
            case LESS_THAN:
                return infix(operands, "<");
            case LESS_THAN_OR_EQUAL:
                return infix(operands, "<=");
            case EQUALS:
                return infix(operands, "==");
            case NOT_EQUALS:
                return infix(operands, "!=");
            case AND:
                return infix(operands, "&&");
            case OR:
                return infix(operands, "||");
            case NOT:
                return operands.size() == 1 ? "(!" + operands.get(0) + ")" : null;
            case CAST:
                // Every value on the device is already a double, so a widening numeric cast is a
                // no-op. Narrowing casts must raise on overflow and a kernel cannot, so they are
                // refused here as they are in the cost estimator.
                return operands.size() == 1 && isDoubleResult(call) ? operands.get(0) : null;
            default:
                return renderFunction(call, operands);
        }
    }

    /** Maps a named SQL function onto {@code TornadoMath}, which is what compiles to the device. */
    private static @Nullable String renderFunction(RexCall call, List<String> operands) {
        String name = call.getOperator().getName().toUpperCase(Locale.ROOT);
        String fn;
        int arity = 1;
        switch (name) {
            case "ABS":
                fn = "abs";
                break;
            case "SQRT":
                fn = "sqrt";
                break;
            case "EXP":
                fn = "exp";
                break;
            case "LN":
                fn = "log";
                break;
            case "LOG2":
                fn = "log2";
                break;
            case "SIN":
                fn = "sin";
                break;
            case "COS":
                fn = "cos";
                break;
            case "TAN":
                fn = "tan";
                break;
            case "ASIN":
                fn = "asin";
                break;
            case "ACOS":
                fn = "acos";
                break;
            case "ATAN":
                fn = "atan";
                break;
            case "TANH":
                fn = "tanh";
                break;
            case "FLOOR":
                fn = "floor";
                break;
            case "CEIL":
                fn = "ceil";
                break;
            case "SIGN":
                fn = "signum";
                break;
            case "POWER":
                fn = "pow";
                arity = 2;
                break;
            case "ATAN2":
                fn = "atan2";
                arity = 2;
                break;
            default:
                return null;
        }
        if (operands.size() != arity) {
            return null;
        }
        return "TornadoMath." + fn + "(" + String.join(", ", operands) + ")";
    }

    private static @Nullable String infix(List<String> operands, String op) {
        if (operands.size() < 2) {
            return null;
        }
        // Fully parenthesised: the tree already encodes precedence, so nothing is left to Java's.
        return "(" + String.join(" " + op + " ", operands) + ")";
    }

    private static @Nullable String renderLiteral(RexLiteral literal) {
        if (literal.getType().getSqlTypeName() == SqlTypeName.BOOLEAN) {
            Object value = literal.getValue3();
            return value instanceof Boolean ? value.toString() : null;
        }
        Object value = literal.getValue3();
        double d;
        if (value instanceof BigDecimal) {
            d = ((BigDecimal) value).doubleValue();
        } else if (value instanceof Number) {
            d = ((Number) value).doubleValue();
        } else {
            return null;
        }
        if (!Double.isFinite(d)) {
            return null;
        }
        // Double.toString round-trips exactly, so the constant the kernel sees is the constant the
        // CPU plan folded.
        return Double.toString(d);
    }

    /**
     * Input types that survive a round trip through a double unchanged.
     *
     * <p>{@code BIGINT} is absent on purpose; see the class comment.
     */
    private static boolean isDoubleSafeInput(SqlTypeName type) {
        return type == SqlTypeName.DOUBLE
                || type == SqlTypeName.FLOAT
                || type == SqlTypeName.REAL
                || type == SqlTypeName.INTEGER;
    }

    private static boolean isDoubleResult(RexNode node) {
        SqlTypeName type = node.getType().getSqlTypeName();
        return type == SqlTypeName.DOUBLE || type == SqlTypeName.FLOAT || type == SqlTypeName.REAL;
    }
}
