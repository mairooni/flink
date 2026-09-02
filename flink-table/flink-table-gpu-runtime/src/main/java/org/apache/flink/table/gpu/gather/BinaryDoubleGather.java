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
import org.apache.flink.table.data.binary.BinaryRowData;

/**
 * Tier 2 — {@code BinaryRowData}. Row-major with a null bitset followed by 8-byte slots, so a
 * column is strided: {@code offset + nullBitsSizeInBytes + pos * 8}. Reading it is a single
 * computed offset and one {@code UNSAFE} load, which is cheap; the cost is the stride, making this
 * a transpose from row-major to column-packed.
 *
 * <p>This is the tier that matters most, because it is guaranteed downstream of any shuffling
 * Exchange — {@code RowDataSerializer.deserialize} delegates unconditionally to the binary
 * serializer.
 *
 * <p>Two properties of this format remove work that a previous Flink/TornadoVM integration had to
 * do by hand. Field bytes are in <b>native</b> order, because access goes through
 * {@code MemorySegment.getDouble} to {@code UNSAFE}, never through {@code DataOutputSerializer}
 * (which byte-swaps to big-endian). And every fixed-length field is already 8-byte aligned by the
 * {@code pos * 8} slot layout, so no padding pass is needed.
 */
final class BinaryDoubleGather implements RowGather {

    private final int field;
    private final DoubleColumn target;

    BinaryDoubleGather(int field, DoubleColumn target) {
        this.field = field;
        this.target = target;
    }

    @Override
    public void accept(RowData row, int position) {
        // Typed access rather than the RowData interface: identical result, but keeps the call
        // monomorphic in the gather loop.
        target.set(position, ((BinaryRowData) row).getDouble(field));
    }

    @Override
    public String tier() {
        return "tier2-binary";
    }
}
