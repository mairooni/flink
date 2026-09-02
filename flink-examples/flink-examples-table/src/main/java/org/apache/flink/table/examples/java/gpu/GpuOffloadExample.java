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

package org.apache.flink.table.examples.java.gpu;

import org.apache.flink.table.api.EnvironmentSettings;
import org.apache.flink.table.api.TableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Runs the same SQL query twice — once normally, once with GPU offload enabled — and verifies that
 * the results are identical.
 *
 * <p>The point of the example is what is <b>not</b> in it. There is no kernel, no device code, no
 * annotation, and no GPU dependency: this class compiles against the Table API alone. Offload is a
 * planner decision, so the only difference between the two runs is two configuration entries.
 *
 * <p>Contrast {@code org.apache.flink.streaming.examples.gpu.MatrixVectorMul}, which demonstrates
 * the other approach — the user writes JCuda calls inside a {@code RichMapFunction} and manages
 * device memory by hand.
 *
 * <p>This example shows how to:
 *
 * <ul>
 *   <li>enable transparent GPU offload for Flink SQL,
 *   <li>read the planner's decision, and the reason behind it, from {@code EXPLAIN},
 *   <li>confirm that offloading does not change query results.
 * </ul>
 *
 * <h2>Prerequisites</h2>
 *
 * <p>Offload is off by default and degrades silently: with the flag on but no GPU runtime on the
 * classpath, the planner falls back to normal code generation and says so in {@code EXPLAIN}. So
 * this example runs correctly either way — it simply does not use a GPU unless the following are
 * present.
 *
 * <ul>
 *   <li>A TornadoVM installation, and its JVM arguments applied to the TaskManagers via {@code
 *       env.java.opts} in {@code flink-conf.yaml} (or as VM options when running from an IDE).
 *   <li>{@code flink-gpu-runtime} on the classpath. It registers a {@code
 *       GpuOperatorFactoryProvider} through {@code META-INF/services}; the planner discovers it
 *       with {@link java.util.ServiceLoader} and holds no compile-time reference to it.
 * </ul>
 *
 * <h2>A note on the query</h2>
 *
 * <p>{@code val * 2.0 + 1.0} is two floating-point operations per row, which is far below the cost
 * floor at which moving data to a device pays for itself — measurements put it at roughly half the
 * speed of CPU execution. It is used here because it matches the one kernel shape the runtime
 * currently implements, and because a small query makes the mechanism easy to follow. The example
 * therefore lowers {@code min-row-cost} deliberately; a real workload should leave the calibrated
 * default in place and let the planner refuse queries like this one.
 */
public final class GpuOffloadExample {

    public static void main(String[] args) throws Exception {
        final int rows = args.length > 0 ? Integer.parseInt(args[0]) : 200_000;

        final List<Row> withoutGpu = run(rows, false);
        final List<Row> withGpu = run(rows, true);

        System.out.printf("%nrows without GPU: %,d%nrows with GPU:    %,d%n",
                withoutGpu.size(), withGpu.size());
        if (!withoutGpu.equals(withGpu)) {
            throw new IllegalStateException(
                    "GPU offload changed the query result, which is a bug");
        }
        System.out.println("Results are identical.");
    }

    private static List<Row> run(int rows, boolean gpu) throws Exception {
        final TableEnvironment env =
                TableEnvironment.create(EnvironmentSettings.newInstance().inBatchMode().build());

        // A deterministic source: datagen's random generator is not seeded, so random values would
        // differ between the two runs and make the comparison meaningless.
        env.executeSql(
                "CREATE TABLE Measurements (\n"
                        + "  id BIGINT,\n"
                        + "  val DOUBLE\n"
                        + ") WITH (\n"
                        + "  'connector' = 'datagen',\n"
                        + "  'number-of-rows' = '" + rows + "',\n"
                        + "  'fields.id.kind' = 'sequence',\n"
                        + "  'fields.id.start' = '0',\n"
                        + "  'fields.id.end' = '" + (rows - 1) + "',\n"
                        + "  'fields.val.kind' = 'sequence',\n"
                        + "  'fields.val.start' = '0',\n"
                        + "  'fields.val.end' = '" + (rows - 1) + "'\n"
                        + ")");

        if (gpu) {
            // The entire user-facing surface of the feature.
            env.getConfig().getConfiguration().setString("table.exec.gpu-offload.enabled", "true");
            // See the class comment: this query is well below the calibrated floor, and is only
            // offloaded here because it matches the kernel the runtime implements.
            env.getConfig().getConfiguration().setString("table.exec.gpu-offload.min-row-cost", "1");
        }

        final String query =
                "SELECT id, val * 2.0 + 1.0 AS scaled\n"
                        + "FROM Measurements\n"
                        + "WHERE val > " + (rows / 2) + ".0";

        System.out.println("========== " + (gpu ? "GPU offload enabled" : "default") + " ==========");
        // With offload enabled, EXPLAIN gains a "== GPU Offload ==" section reporting which
        // subtrees were selected and, for those that were not, why.
        System.out.println(env.explainSql(query));

        final List<Row> result = new ArrayList<>();
        try (CloseableIterator<Row> rows0 = env.sqlQuery(query).execute().collect()) {
            rows0.forEachRemaining(result::add);
        }
        result.sort(Comparator.comparingLong(row -> (Long) row.getField(0)));
        return result;
    }

    private GpuOffloadExample() {}
}
