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
import org.apache.flink.table.data.columnar.ColumnarRowData;

/**
 * Tier 1 — {@code ColumnarRowData} from a vectorized Parquet/ORC source. The data is already
 * column-major and packed ({@code HeapIntVector.vector} is a plain {@code int[]}), so no
 * transposition is needed at all.
 *
 * <p><b>Currently degrades to per-row access.</b> {@code ColumnarRowData} exposes
 * {@code setVectorizedColumnBatch}/{@code setRowId} but no getters, so the underlying
 * {@code VectorizedColumnBatch} is unreachable without either reflection or a small patch to
 * {@code flink-table-common}. Until that patch lands this class is a correctness placeholder that
 * records the tier honestly rather than claiming a fast path it does not take.
 *
 * <p>When the batch does become reachable, a bulk copy is only valid under
 * {@code dictionary == null && !isRepeating && noNulls}: {@code HeapIntVector.getInt} returns
 * {@code dictionary.decodeToInt(dictionaryIds.vector[i])} when dictionary-encoded, and
 * {@code OrcLongColumnVector} indexes {@code vector[isRepeating ? 0 : i]}. Both would silently
 * produce wrong values rather than fail.
 */
final class ColumnarDoubleGather implements RowGather {

    private final int field;
    private final DoubleColumn target;

    ColumnarDoubleGather(int field, DoubleColumn target) {
        this.field = field;
        this.target = target;
    }

    @Override
    public void accept(RowData row, int position) {
        target.set(position, ((ColumnarRowData) row).getDouble(field));
    }

    @Override
    public String tier() {
        return "tier1-columnar(per-row; bulk path needs accessor patch)";
    }
}
