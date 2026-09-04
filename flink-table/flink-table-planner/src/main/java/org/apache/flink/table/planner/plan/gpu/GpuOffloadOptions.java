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

import org.apache.flink.configuration.ConfigOption;
import org.apache.flink.configuration.ConfigOptions;

/**
 * Configuration for GPU offload.
 *
 * <p>Declared here rather than in {@code ExecutionConfigOptions} so the fork touches a single
 * module. If this were upstreamed the options would move there, where the documentation generator
 * picks them up.
 */
public class GpuOffloadOptions {

    public static final ConfigOption<Boolean> ENABLED =
            ConfigOptions.key(GpuOffloadDecision.ENABLED_KEY)
                    .booleanType()
                    .defaultValue(false)
                    .withDescription(
                            "If true, subtrees of the plan whose per-row work is expressible as a "
                                    + "data-parallel kernel and exceeds "
                                    + GpuOffloadDecision.MIN_ROW_COST_KEY
                                    + " are executed on a GPU. Off by default.");

    public static final ConfigOption<Long> MIN_TOTAL_WORK =
            ConfigOptions.key(GpuOffloadDecision.MIN_TOTAL_WORK_KEY)
                    .longType()
                    .defaultValue(GpuOffloadDecision.DEFAULT_MIN_TOTAL_WORK)
                    .withDescription(
                            "Minimum total work, in weighted operations across all rows, before a "
                                    + "subtree is offloaded. The per-row floor asks whether the "
                                    + "device is faster on each row; this asks whether there are "
                                    + "enough rows to repay what offloading costs once per task -- "
                                    + "compiling the kernel with javac, compiling it again for the "
                                    + "device, and allocating the staging buffers. A heavy "
                                    + "expression over a short input clears the first gate and "
                                    + "still loses.");

    public static final ConfigOption<Integer> MIN_ROW_COST =
            ConfigOptions.key(GpuOffloadDecision.MIN_ROW_COST_KEY)
                    .intType()
                    .defaultValue(GpuOffloadDecision.DEFAULT_MIN_ROW_COST)
                    .withDescription(
                            "Minimum estimated per-row work, in weighted operations, before a "
                                    + "subtree is offloaded. Expressions below this threshold are "
                                    + "faster on the CPU than the staging and device transfer the "
                                    + "offload would add. Device-specific: calibrate with the "
                                    + "intensity sweep in the offload harness rather than assuming "
                                    + "the default transfers.");

    public static final ConfigOption<Integer> BATCH_SIZE =
            ConfigOptions.key("table.exec.gpu-offload.batch-size")
                    .intType()
                    .defaultValue(262_144)
                    .withDescription(
                            "Rows staged before a kernel launch. Batching is not optional -- a "
                                    + "device kernel needs a sized contiguous buffer -- so this "
                                    + "trades memory per subtask against dispatch overhead. "
                                    + "Measured from 64K to 4M, device work stays near a tenth of "
                                    + "the total throughout and larger batches mainly reduce "
                                    + "TornadoVM dispatch cost.");

    private GpuOffloadOptions() {}
}
