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

package org.apache.flink.table.gpu.spike;

import javax.tools.JavaCompiler;
import javax.tools.ToolProvider;

import java.lang.reflect.Method;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Feasibility spike: generating a TornadoVM kernel at runtime for an ARBITRARY expression.
 *
 * <p>Not part of the offload path. It exists to answer, with a running program rather than an
 * argument, whether the hand-written kernel catalogue can be replaced by generated kernels.
 *
 * <p>The blocker has always been that TornadoVM resolves a kernel from its method reference's
 * SerializedLambda, which needs a writeReplace() only javac emits -- so Janino-generated classes
 * are unusable. But nothing says the compiler has to be Janino.
 *
 * <p>This generates BOTH the kernel and the class that builds the TaskGraph referencing it,
 * compiles them together with the real javac via javax.tools, and runs it. The method reference is
 * then an ordinary javac lambda with a proper writeReplace.
 */
public class RuntimeKernelSpike {

    /** An expression with no entry in the hand-written catalogue. */
    private static final String EXPRESSION =
            "TornadoMath.exp(v) * TornadoMath.log(v) + TornadoMath.sin(v) * TornadoMath.cos(v)";

    public static void main(String[] args) throws Exception {
        JavaCompiler javac = ToolProvider.getSystemJavaCompiler();
        System.out.println("javac available at runtime: " + (javac != null));
        if (javac == null) {
            throw new IllegalStateException("running on a JRE; a JDK is required");
        }

        Path dir = Files.createTempDirectory("gpukernel");
        Path src = dir.resolve("GeneratedKernel.java");
        Files.writeString(src, source(EXPRESSION));

        String cp = System.getProperty("java.class.path");
        List<String> opts =
                new ArrayList<>(
                        List.of(
                                "-classpath",
                                cp,
                                "-d",
                                dir.toString(),
                                "--enable-preview",
                                "-source",
                                "21",
                                "-target",
                                "21",
                                "-nowarn"));
        int rc = javac.run(null, null, System.err, concat(opts, src.toString()));
        System.out.println("javac exit: " + rc);
        if (rc != 0) {
            throw new IllegalStateException("generated source did not compile");
        }

        try (URLClassLoader loader =
                new URLClassLoader(
                        new URL[] {dir.toUri().toURL()},
                        RuntimeKernelSpike.class.getClassLoader())) {
            Class<?> generated = loader.loadClass("GeneratedKernel");
            Method run = generated.getMethod("runOnDevice", int.class);
            double[] out = (double[]) run.invoke(null, 1024);

            // Verify against the host computing the same expression.
            int wrong = 0;
            int nonFinite = 0;
            double maxRel = 0;
            for (int i = 0; i < out.length; i++) {
                double v = 0.1 + i * (10.0 / out.length);
                double want = Math.exp(v) * Math.log(v) + Math.sin(v) * Math.cos(v);
                if (!Double.isFinite(want) || !Double.isFinite(out[i])) {
                    // A NaN or infinity would make every relative comparison false and the check
                    // vacuous, so count these rather than silently passing them.
                    nonFinite++;
                    continue;
                }
                double rel = want == 0 ? Math.abs(out[i]) : Math.abs((out[i] - want) / want);
                maxRel = Math.max(maxRel, rel);
                if (rel > 1e-6) {
                    wrong++;
                }
            }
            System.out.printf(
                    "elements=%d  compared=%d  non_finite=%d  mismatches=%d  max_rel_err=%.3g%n",
                    out.length, out.length - nonFinite, nonFinite, wrong, maxRel);
            System.out.println(
                    wrong == 0 && nonFinite == 0
                            ? "ARBITRARY EXPRESSION RAN ON THE DEVICE, generated and compiled at runtime"
                            : "*** results wrong ***");
        }
    }

    private static String[] concat(List<String> opts, String file) {
        List<String> all = new ArrayList<>(opts);
        all.add(file);
        return all.toArray(new String[0]);
    }

    /**
     * The generated unit holds the kernel AND the TaskGraph that references it, so the method
     * reference is compiled by javac in the same pass and carries writeReplace().
     */
    private static String source(String expression) {
        return ""
                + "import uk.ac.manchester.tornado.api.*;\n"
                + "import uk.ac.manchester.tornado.api.annotations.Parallel;\n"
                + "import uk.ac.manchester.tornado.api.enums.DataTransferMode;\n"
                + "import uk.ac.manchester.tornado.api.math.TornadoMath;\n"
                + "import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;\n"
                + "\n"
                + "public final class GeneratedKernel {\n"
                + "\n"
                + "    public static void evaluate(DoubleArray in, DoubleArray out) {\n"
                + "        for (@Parallel int i = 0; i < in.getSize(); i++) {\n"
                + "            double v = in.get(i);\n"
                + "            out.set(i, "
                + expression
                + ");\n"
                + "        }\n"
                + "    }\n"
                + "\n"
                + "    public static double[] runOnDevice(int n) throws Exception {\n"
                + "        DoubleArray in = new DoubleArray(n);\n"
                + "        DoubleArray out = new DoubleArray(n);\n"
                + "        for (int i = 0; i < n; i++) { in.set(i, 0.1 + i * (10.0 / n)); }\n"
                + "        out.init(0.0);\n"
                + "        TaskGraph g = new TaskGraph(\"gen\")\n"
                + "                .transferToDevice(DataTransferMode.EVERY_EXECUTION, in)\n"
                + "                .task(\"t\", GeneratedKernel::evaluate, in, out)\n"
                + "                .transferToHost(DataTransferMode.EVERY_EXECUTION, out);\n"
                + "        try (TornadoExecutionPlan p = new TornadoExecutionPlan(g.snapshot())) {\n"
                + "            p.execute();\n"
                + "        }\n"
                + "        double[] r = new double[n];\n"
                + "        for (int i = 0; i < n; i++) { r[i] = out.get(i); }\n"
                + "        return r;\n"
                + "    }\n"
                + "}\n";
    }
}
