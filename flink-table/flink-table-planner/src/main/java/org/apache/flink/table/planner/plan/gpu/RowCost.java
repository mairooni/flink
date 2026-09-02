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

import java.util.Objects;

/**
 * Result of costing an expression tree for GPU offload: either an estimated per-row work weight, or
 * a reason the expression cannot run on the device at all.
 *
 * <p>The two are one type because a single walk of the {@code RexNode} tree decides both, and
 * splitting them into separate visitors would let the eligibility rules and the cost rules drift
 * apart — which is the failure mode where a query is judged offloadable and then costed with
 * weights for an operator the kernel library cannot actually express.
 *
 * <p>Ineligible dominates: adding anything to an ineligible cost keeps the first rejection reason,
 * so the message names the expression that actually caused it rather than the last one visited.
 */
public final class RowCost {

    /** Eligible, no work. Identity for {@link #plus}. */
    public static final RowCost ZERO = new RowCost(0, null);

    private final int weight;
    private final String rejection;

    private RowCost(int weight, String rejection) {
        this.weight = weight;
        this.rejection = rejection;
    }

    public static RowCost of(int weight) {
        if (weight < 0) {
            throw new IllegalArgumentException("weight must be non-negative, was " + weight);
        }
        return weight == 0 ? ZERO : new RowCost(weight, null);
    }

    /**
     * @param reason what about the expression cannot be expressed as a kernel; surfaced in EXPLAIN
     */
    public static RowCost ineligible(String reason) {
        return new RowCost(0, Objects.requireNonNull(reason));
    }

    public boolean isEligible() {
        return rejection == null;
    }

    /** Estimated weighted operations per row. Meaningless unless {@link #isEligible()}. */
    public int weight() {
        return weight;
    }

    /** Null when eligible. */
    public String rejection() {
        return rejection;
    }

    public RowCost plus(RowCost other) {
        if (!isEligible()) {
            return this;
        }
        if (!other.isEligible()) {
            return other;
        }
        return of(weight + other.weight);
    }

    @Override
    public String toString() {
        return isEligible() ? "RowCost[" + weight + "]" : "RowCost[ineligible: " + rejection + "]";
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof RowCost)) {
            return false;
        }
        RowCost other = (RowCost) o;
        return other.weight == weight && Objects.equals(other.rejection, rejection);
    }

    @Override
    public int hashCode() {
        return Objects.hash(weight, rejection);
    }
}
