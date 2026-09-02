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

package org.apache.flink.table.planner.plan.nodes.exec.common;

import org.apache.flink.api.dag.Transformation;
import org.apache.flink.configuration.ReadableConfig;
import org.apache.flink.streaming.api.operators.StreamOperatorFactory;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.planner.plan.gpu.GpuCalcMatcher;
import org.apache.flink.table.planner.plan.gpu.GpuCalcSpec;
import org.apache.flink.table.planner.plan.gpu.GpuOffloadAssignment;
import org.apache.flink.table.planner.plan.gpu.GpuOffloadDecision;
import org.apache.flink.table.planner.plan.gpu.GpuOperatorFactoryProvider;
import org.apache.flink.table.planner.codegen.CalcCodeGenerator;
import org.apache.flink.table.planner.codegen.CodeGeneratorContext;
import org.apache.flink.table.planner.delegation.PlannerBase;
import org.apache.flink.table.planner.plan.nodes.exec.ExecEdge;
import org.apache.flink.table.planner.plan.nodes.exec.ExecNodeBase;
import org.apache.flink.table.planner.plan.nodes.exec.ExecNodeConfig;
import org.apache.flink.table.planner.plan.nodes.exec.ExecNodeContext;
import org.apache.flink.table.planner.plan.nodes.exec.InputProperty;
import org.apache.flink.table.planner.plan.nodes.exec.SingleTransformationTranslator;
import org.apache.flink.table.planner.plan.nodes.exec.utils.ExecNodeUtil;
import org.apache.flink.table.planner.utils.JavaScalaConversionUtil;
import org.apache.flink.table.runtime.operators.CodeGenOperatorFactory;
import org.apache.flink.table.runtime.typeutils.InternalTypeInfo;
import org.apache.flink.table.types.logical.RowType;

import org.apache.flink.shaded.jackson2.com.fasterxml.jackson.annotation.JsonProperty;

import org.apache.calcite.rex.RexNode;

import javax.annotation.Nullable;

import java.util.List;
import java.util.Optional;

import static org.apache.flink.util.Preconditions.checkArgument;
import static org.apache.flink.util.Preconditions.checkNotNull;

/** Base class for exec Calc. */
public abstract class CommonExecCalc extends ExecNodeBase<RowData>
        implements SingleTransformationTranslator<RowData> {

    public static final String CALC_TRANSFORMATION = "calc";

    public static final String FIELD_NAME_PROJECTION = "projection";
    public static final String FIELD_NAME_CONDITION = "condition";

    @JsonProperty(FIELD_NAME_PROJECTION)
    protected final List<RexNode> projection;

    @JsonProperty(FIELD_NAME_CONDITION)
    protected final @Nullable RexNode condition;

    private final Class<?> operatorBaseClass;
    private final boolean retainHeader;

    protected CommonExecCalc(
            int id,
            ExecNodeContext context,
            ReadableConfig persistedConfig,
            List<RexNode> projection,
            @Nullable RexNode condition,
            Class<?> operatorBaseClass,
            boolean retainHeader,
            List<InputProperty> inputProperties,
            RowType outputType,
            String description) {
        super(id, context, persistedConfig, inputProperties, outputType, description);
        checkArgument(inputProperties.size() == 1);
        this.projection = checkNotNull(projection);
        this.condition = condition;
        this.operatorBaseClass = checkNotNull(operatorBaseClass);
        this.retainHeader = retainHeader;
    }

    @SuppressWarnings("unchecked")
    @Override
    protected Transformation<RowData> translateToPlanInternal(
            PlannerBase planner, ExecNodeConfig config) {
        final ExecEdge inputEdge = getInputEdges().get(0);
        final Transformation<RowData> inputTransform =
                (Transformation<RowData>) inputEdge.translateToPlan(planner);
        final CodeGeneratorContext ctx =
                new CodeGeneratorContext(config, planner.getFlinkContext().getClassLoader())
                        .setOperatorBaseClass(operatorBaseClass);

        final StreamOperatorFactory<RowData> gpuOperator =
                tryGpuOffload(planner, (RowType) getOutputType());
        if (gpuOperator != null) {
            return ExecNodeUtil.createOneInputTransformation(
                    inputTransform,
                    createTransformationMeta(CALC_TRANSFORMATION, config),
                    gpuOperator,
                    InternalTypeInfo.of(getOutputType()),
                    inputTransform.getParallelism(),
                    false);
        }

        final CodeGenOperatorFactory<RowData> substituteStreamOperator =
                CalcCodeGenerator.generateCalcOperator(
                        ctx,
                        inputTransform,
                        (RowType) getOutputType(),
                        JavaScalaConversionUtil.toScala(projection),
                        JavaScalaConversionUtil.toScala(Optional.ofNullable(this.condition)),
                        retainHeader,
                        getClass().getSimpleName());
        return ExecNodeUtil.createOneInputTransformation(
                inputTransform,
                createTransformationMeta(CALC_TRANSFORMATION, config),
                substituteStreamOperator,
                InternalTypeInfo.of(getOutputType()),
                inputTransform.getParallelism(),
                false);
    }

    /**
     * Returns a GPU operator factory for this node, or null to fall through to code generation.
     *
     * <p>Three independent things must hold, and any of them failing is a normal outcome rather
     * than an error: the offload processor must have selected this node, a provider must be on the
     * classpath, and the runtime's kernel catalogue must have a match for the expressions
     * (clearing the cost floor says an expression is worth offloading, not that a kernel exists for
     * it).
     *
     * <p>Every fallback path is silent to the query but visible in EXPLAIN, so a plan that was
     * expected to offload and did not can be diagnosed without attaching a debugger.
     */
    private @Nullable StreamOperatorFactory<RowData> tryGpuOffload(
            PlannerBase planner, RowType outputType) {
        GpuOffloadAssignment assignment = getGpuOffloadAssignment();
        if (assignment == null || !assignment.isOffloaded()) {
            return null;
        }
        Optional<GpuOperatorFactoryProvider> provider =
                GpuOperatorFactoryProvider.find(planner.getFlinkContext().getClassLoader());
        if (!provider.isPresent()) {
            recordGpuFallback(assignment, "no GpuOperatorFactoryProvider on the classpath");
            return null;
        }
        Optional<GpuCalcSpec> spec =
                GpuCalcMatcher.match(
                        projection, condition, outputType, assignment.verdict().rowCost());
        if (!spec.isPresent()) {
            recordGpuFallback(assignment, "no kernel in the catalogue matches these expressions");
            return null;
        }
        Optional<StreamOperatorFactory<RowData>> factory =
                provider.get().createCalcOperatorFactory(spec.get());
        if (!factory.isPresent()) {
            recordGpuFallback(
                    assignment, provider.get().describe() + " declined " + spec.get());
            return null;
        }
        return factory.get();
    }

    /** Rewrites the node's verdict so EXPLAIN reports why a selected node still ran on the CPU. */
    private void recordGpuFallback(GpuOffloadAssignment assignment, String reason) {
        setGpuOffloadAssignment(
                new GpuOffloadAssignment(
                        assignment.groupId(),
                        GpuOffloadDecision.fallback(assignment.verdict(), reason)));
    }
}
