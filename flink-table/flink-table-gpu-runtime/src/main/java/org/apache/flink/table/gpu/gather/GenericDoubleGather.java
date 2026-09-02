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
package org.apache.flink.table.gpu.gather;

import org.apache.flink.table.data.RowData;

/**
 * Tier 4 — the fallback. Reads through the {@link RowData} interface, which for
 * {@code GenericRowData} and {@code BoxedWrapperRowData} means an object dereference plus an unbox
 * per field.
 *
 * <p>Correct for every implementation, and expected to be the slowest by a wide margin. It exists
 * so the operator is never wrong, not because it is expected to pay: the planner-side predicate
 * should reject inputs known to land here (chained after another Calc is knowable at plan time).
 */
final class GenericDoubleGather implements RowGather {

    private final int field;
    private final DoubleColumn target;

    GenericDoubleGather(int field, DoubleColumn target) {
        this.field = field;
        this.target = target;
    }

    @Override
    public void accept(RowData row, int position) {
        target.set(position, row.getDouble(field));
    }

    @Override
    public String tier() {
        return "tier4-generic";
    }
}
