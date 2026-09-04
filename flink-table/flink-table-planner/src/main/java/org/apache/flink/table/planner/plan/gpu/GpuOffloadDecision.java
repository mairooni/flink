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

import org.apache.flink.table.planner.plan.nodes.exec.GpuOffloadExecNode;

import org.apache.calcite.rex.RexNode;

import java.util.List;

/**
 * The offload gate: combines type eligibility, estimated per-row work and the calibrated floor into
 * a single yes/no with a reason.
 *
 * <p>Kept separate from {@link GpuCostEstimator} so the floor — which is deployment configuration —
 * does not leak into the part that only reads the expression tree. The estimator answers "how much
 * work is this"; this class answers "is that enough, here".
 */
public final class GpuOffloadDecision {

    /** Master switch. Off by default: this is opt-in until the floor has a justified default. */
    public static final String ENABLED_KEY = "table.exec.gpu-offload.enabled";

    /** Calibrated floor, in the weighted units of {@link OperatorWeights}. */
    public static final String MIN_ROW_COST_KEY = "table.exec.gpu-offload.min-row-cost";

    /**
     * Default floor, deliberately set above the measured break-even.
     *
     * <h2>How this number was derived</h2>
     *
     * <p>The floor must be expressed in the same weighted units the estimator produces, so the
     * calibration kernel is costed with the same {@link OperatorWeights} table. Its inner round —
     * {@code sqrt(acc*acc + 1.0)*0.5 + acc*0.25} — is 13 weighted units (SQRT 8, three multiplies,
     * two adds), over a base of 3 for the output projection and the filter comparison:
     *
     * <pre>
     *   intensity   weighted ops/row   measured tier-1 speedup
     *           0                  3                     0.54x
     *           4                 55                     0.84x
     *          16                211                     2.22x
     *          64                835                    10.19x
     * </pre>
     *
     * <p>Break-even therefore brackets between 55 and 211; log-interpolating between those two
     * points puts it at roughly <b>70 weighted ops/row</b> on an RTX 4070 over PCIe 4.
     *
     * <p>This default sits about 1.4x above that, which gives up real wins between 70 and 96 in
     * exchange for margin against a regression. The asymmetry is deliberate: a missed speedup is
     * invisible, whereas a query that silently got twice as slow because a flag was enabled is a
     * support incident.
     *
     * <p><b>Treat this as a placeholder, not a recommendation.</b> It is one device, one input
     * shape, and single runs. It is also only meaningful alongside the weight table it was derived
     * with — changing a weight without re-deriving the floor invalidates both.
     */
    public static final int DEFAULT_MIN_ROW_COST = 96;

    /**
     * Log-interpolated break-even from the calibration sweep; see {@link #DEFAULT_MIN_ROW_COST}.
     */
    public static final int MEASURED_BREAK_EVEN = 70;

    /** Second gate: total work across all rows. See {@link #DEFAULT_MIN_TOTAL_WORK}. */
    public static final String MIN_TOTAL_WORK_KEY = "table.exec.gpu-offload.min-total-work";

    /**
     * Roughly ten million weighted operations, derived from the end-to-end benchmark rather than
     * guessed.
     *
     * <p>Measured on an RTX 4070 Laptop: 2M rows of a 3458 weighted-op expression took 14.4 s on
     * the CPU and 121 ms of kernel time, so the device does about 5.7e10 weighted ops/s against the
     * CPU's 4.8e8 -- 119x. Offloading also costs about 20 ms per task before any row is processed,
     * nearly all of it compiling the kernel twice, once with javac and once for the device. Setting
     * work/cpu_rate equal to that fixed cost plus work/gpu_rate puts break-even near 1e7 weighted
     * operations: about 100k rows of a floor-weight expression, or 2.8k rows of the benchmark's.
     *
     * <p>Like the per-row floor this is device-specific, and it is a floor rather than a target:
     * clearing it means offloading should not lose, not that anyone will notice it winning.
     */
    public static final long DEFAULT_MIN_TOTAL_WORK = 10_000_000L;

    private final int minRowCost;

    private final long minTotalWork;

    public GpuOffloadDecision(int minRowCost) {
        this(minRowCost, DEFAULT_MIN_TOTAL_WORK);
    }

    public GpuOffloadDecision(int minRowCost, long minTotalWork) {
        if (minRowCost < 0) {
            throw new IllegalArgumentException(
                    "min-row-cost must be non-negative, was " + minRowCost);
        }
        if (minTotalWork < 0) {
            throw new IllegalArgumentException(
                    "min-total-work must be non-negative, was " + minTotalWork);
        }
        this.minRowCost = minRowCost;
        this.minTotalWork = minTotalWork;
    }

    public static GpuOffloadDecision withDefaults() {
        return new GpuOffloadDecision(DEFAULT_MIN_ROW_COST, DEFAULT_MIN_TOTAL_WORK);
    }

    /**
     * Outcome of the gate. {@code reason} is always populated, including when offloading, so that
     * EXPLAIN can report why a node was or was not moved.
     *
     * <p>Written out longhand rather than as a record because flink-table-planner compiles at
     * {@code -source 11}.
     */
    public static final class Verdict {
        private final boolean offload;
        private final int rowCost;
        private final String reason;

        Verdict(boolean offload, int rowCost, String reason) {
            this.offload = offload;
            this.rowCost = rowCost;
            this.reason = reason;
        }

        public boolean offload() {
            return offload;
        }

        public int rowCost() {
            return rowCost;
        }

        public String reason() {
            return reason;
        }

        @Override
        public String toString() {
            return (offload ? "offload" : "cpu") + "[cost=" + rowCost + ", " + reason + "]";
        }
    }

    /**
     * Decides for one node's expressions.
     *
     * <p>A node that fails the floor alone may still be offloaded as part of a subtree, because
     * costs sum: see {@link #forSubtree}. This method is the single-node view and is expected to
     * say no most of the time.
     */
    public Verdict decide(List<? extends RexNode> expressions) {
        return decide(expressions, GpuOffloadExecNode.UNKNOWN_ROW_COUNT);
    }

    /** As {@link #decide(List)}, with the planner's row-count estimate for the node. */
    public Verdict decide(List<? extends RexNode> expressions, double rowCount) {
        RowCost cost = GpuCostEstimator.estimateAll(expressions);
        return verdict(cost, rowCount);
    }

    /**
     * Decides for a whole matched subtree by summing its nodes' costs.
     *
     * <p>This is the second reason the unit of offload is a subtree rather than a node: fusion
     * raises the numerator against a fixed floor, so {@code Calc → Calc → Agg} can clear a
     * threshold that none of its members clears alone. The first reason — one task graph instead of
     * one per node — is about transfer count.
     */
    public Verdict forSubtree(List<RowCost> nodeCosts) {
        return forSubtree(nodeCosts, GpuOffloadExecNode.UNKNOWN_ROW_COUNT);
    }

    /** As {@link #forSubtree(List)}, with the planner's row-count estimate for the subtree. */
    public Verdict forSubtree(List<RowCost> nodeCosts, double rowCount) {
        RowCost total = RowCost.ZERO;
        for (RowCost cost : nodeCosts) {
            total = total.plus(cost);
        }
        return verdict(total, rowCount);
    }

    /**
     * Turns a selected verdict into a fallback one, keeping the cost so EXPLAIN still shows what
     * the node was worth alongside why it did not run on the device.
     */
    public static Verdict fallback(Verdict selected, String reason) {
        return new Verdict(false, selected.rowCost(), "selected but fell back: " + reason);
    }

    private Verdict verdict(RowCost cost, double rowCount) {
        if (!cost.isEligible()) {
            return new Verdict(false, 0, "not expressible on device: " + cost.rejection());
        }
        if (cost.weight() < minRowCost) {
            return new Verdict(
                    false,
                    cost.weight(),
                    String.format(
                            "below cost floor: %d < %d weighted ops/row; CPU execution is expected to be "
                                    + "faster than the staging and transfer this would add",
                            cost.weight(), minRowCost));
        }
        // Second gate. The floor above says the device wins on each row; this says there are
        // enough rows for that to repay what offloading costs once per task. An unknown row count
        // skips it rather than guessing -- refusing on no evidence would silently disable offload
        // wherever the planner has no estimate.
        if (rowCount != GpuOffloadExecNode.UNKNOWN_ROW_COUNT && rowCount >= 0) {
            double totalWork = rowCount * cost.weight();
            if (totalWork < minTotalWork) {
                return new Verdict(
                        false,
                        cost.weight(),
                        String.format(
                                "below total-work floor: %.0f rows x %d weighted ops/row = %.3g < "
                                        + "%d; too few rows to repay compiling the kernel",
                                rowCount, cost.weight(), totalWork, minTotalWork));
            }
            return new Verdict(
                    true,
                    cost.weight(),
                    String.format(
                            "%d weighted ops/row clears floor of %d, and %.0f rows give %.3g "
                                    + "weighted ops against a floor of %d",
                            cost.weight(), minRowCost, rowCount, totalWork, minTotalWork));
        }
        return new Verdict(
                true,
                cost.weight(),
                String.format(
                        "%d weighted ops/row clears floor of %d; row count unknown, so the "
                                + "total-work floor was not applied",
                        cost.weight(), minRowCost));
    }
}
