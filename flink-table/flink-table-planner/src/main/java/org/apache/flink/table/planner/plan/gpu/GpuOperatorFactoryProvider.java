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

import org.apache.flink.streaming.api.operators.StreamOperatorFactory;
import org.apache.flink.table.data.RowData;

import java.util.Iterator;
import java.util.Optional;
import java.util.ServiceLoader;

/**
 * Service interface through which the planner obtains a GPU operator without compiling against one.
 *
 * <p>Discovered with {@link ServiceLoader}, deliberately, so that {@code flink-table-planner} does
 * not depend on TornadoVM. That dependency would put a JVMCI-requiring, preview-API library on the
 * build path of a core Flink module for a feature that is off by default, and would make the fork
 * far harder to argue for upstream.
 *
 * <p>Absence is a supported state, not an error: with the flag on but no provider on the classpath,
 * every node falls back to normal code generation and says so in EXPLAIN. That is also what happens
 * on a cluster where only some TaskManagers have GPUs and the operator jar.
 */
public interface GpuOperatorFactoryProvider {

    /**
     * Builds an operator factory for one offloaded Calc, or empty if this provider cannot serve the
     * spec — for instance because the device is absent, or the kernel catalogue has no match.
     */
    Optional<StreamOperatorFactory<RowData>> createCalcOperatorFactory(GpuCalcSpec spec);

    /** Human-readable identity, for the EXPLAIN section and logs. */
    String describe();

    /**
     * Loads the single provider on the classpath.
     *
     * <p>Not cached: this runs once per offloaded node during planning, and caching a provider in a
     * static field would outlive the user classloader that supplied it.
     */
    static Optional<GpuOperatorFactoryProvider> find(ClassLoader classLoader) {
        Iterator<GpuOperatorFactoryProvider> it =
                ServiceLoader.load(GpuOperatorFactoryProvider.class, classLoader).iterator();
        return it.hasNext() ? Optional.of(it.next()) : Optional.empty();
    }
}
