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

package org.apache.flink.table.gpu.operator;

import org.apache.flink.table.gpu.metrics.OffloadMetrics;
import org.apache.flink.table.runtime.gpu.GpuCalcSpec;
import org.apache.flink.table.runtime.gpu.GpuKernelSource;

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;
import uk.ac.manchester.tornado.api.TornadoProfilerResult;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.ProfilerMode;
import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

import javax.tools.JavaCompiler;
import javax.tools.ToolProvider;

import java.io.IOException;
import java.lang.reflect.Method;
import java.net.URISyntaxException;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.ProtectionDomain;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.stream.Stream;

/**
 * Compiles a generated kernel and runs it over batches of staged columns.
 *
 * <p>The kernel is not one of a few hand-written methods but Java source produced from the query's
 * own expressions, compiled once per operator instance. It replaced an earlier fixed catalogue that
 * could express 2 of the 13 shapes tested; generation covers 9.
 *
 * <h2>Why javac and a directory</h2>
 *
 * <p>TornadoVM reads a kernel's bytecode through {@code loader.getResourceAsStream(name +
 * ".class")}. A class defined only in memory is therefore unusable however it was produced, so the
 * compiled class is written to a temporary directory and loaded through a {@link URLClassLoader},
 * which serves it. {@code javax.tools} needs a JDK rather than a JRE, which TornadoVM already
 * requires.
 *
 * <p>Compilation happens once in {@link #open()}, alongside TornadoVM's own kernel compilation, not
 * per batch.
 */
public final class GeneratedKernelEngine implements AutoCloseable {

    private final GpuCalcSpec spec;
    private final boolean profile;

    private DoubleArray[] inputs;
    private DoubleArray[] outputs;
    private IntArray mask;
    private Path workDir;
    private URLClassLoader loader;
    private TornadoExecutionPlan plan;

    private final OffloadMetrics metrics = new OffloadMetrics();

    public GeneratedKernelEngine(GpuCalcSpec spec, boolean profile) {
        this.spec = spec;
        this.profile = profile;
    }

    public void open() throws Exception {
        final GpuKernelSource kernel = spec.kernel();
        final int batchSize = spec.batchSize();

        inputs = new DoubleArray[kernel.inputFieldIndexes().length];
        for (int i = 0; i < inputs.length; i++) {
            inputs[i] = new DoubleArray(batchSize);
            // TornadoVM's native arrays hold garbage on allocation, unlike Java arrays. The tail of
            // a partial batch is never read, but leaving it undefined would make device results
            // differ between runs.
            inputs[i].init(0.0);
        }
        outputs = new DoubleArray[kernel.outputCount()];
        for (int i = 0; i < outputs.length; i++) {
            outputs[i] = new DoubleArray(batchSize);
            outputs[i].init(0.0);
        }
        if (kernel.hasFilter()) {
            mask = new IntArray(batchSize);
            mask.init(0);
        }

        Method entry = compile(kernel);

        Object[] args = new Object[inputs.length + outputs.length + (mask == null ? 0 : 1)];
        int at = 0;
        for (DoubleArray in : inputs) {
            args[at++] = in;
        }
        for (DoubleArray out : outputs) {
            args[at++] = out;
        }
        if (mask != null) {
            args[at] = mask;
        }

        TaskGraph graph = new TaskGraph("calc");
        graph = graph.transferToDevice(DataTransferMode.EVERY_EXECUTION, (Object[]) inputs);
        // Naming the kernel by Method rather than by a method reference is what makes a generated
        // kernel possible at all: a method reference would have to exist in source.
        graph = graph.task("kernel", entry, args);
        Object[] results =
                mask == null
                        ? outputs
                        : Stream.concat(Arrays.stream(outputs), Stream.of(mask)).toArray();
        graph = graph.transferToHost(DataTransferMode.EVERY_EXECUTION, results);

        plan = new TornadoExecutionPlan(graph.snapshot());
        if (profile) {
            plan = plan.withProfiler(ProfilerMode.SILENT);
        }
    }

    private Method compile(GpuKernelSource kernel) throws Exception {
        JavaCompiler javac = ToolProvider.getSystemJavaCompiler();
        if (javac == null) {
            throw new IllegalStateException(
                    "No Java compiler available: GPU offload generates kernels at run time and so "
                            + "needs a JDK, not a JRE. TornadoVM requires one in any case.");
        }
        workDir = Files.createTempDirectory("flink-gpu-kernel");
        // The generated class name contains '$' to keep it distinct from anything hand-written;
        // the file must be named after the class for javac to accept it.
        String fileName = kernel.className() + ".java";
        Path source = workDir.resolve(fileName);
        Files.writeString(source, kernel.source());

        List<String> options =
                new ArrayList<>(
                        Arrays.asList(
                                "-classpath",
                                classpathFor(),
                                "-d",
                                workDir.toString(),
                                "--enable-preview",
                                "-source",
                                "21",
                                "-target",
                                "21",
                                // Debug info is not optional. @Parallel is a local-variable
                                // annotation, and TornadoVM associates it with the loop induction
                                // variable through the local variable table. Without -g javac emits
                                // no such table, the annotation is silently ignored, and the kernel
                                // is generated as a sequential loop that every GPU thread runs in
                                // full -- correct results, catastrophically slow, and no warning.
                                "-g",
                                "-nowarn"));
        options.add(source.toString());

        int rc = javac.run(null, null, System.err, options.toArray(new String[0]));
        if (rc != 0) {
            throw new IllegalStateException(
                    "Generated kernel did not compile. This is a bug in the generator; the source "
                            + "was:\n"
                            + kernel.source());
        }

        loader =
                new URLClassLoader(
                        new URL[] {workDir.toUri().toURL()},
                        GeneratedKernelEngine.class.getClassLoader());
        Class<?> generated = loader.loadClass(kernel.className());
        for (Method m : generated.getDeclaredMethods()) {
            if (m.getName().equals(kernel.methodName())) {
                return m;
            }
        }
        throw new IllegalStateException("generated class has no method " + kernel.methodName());
    }

    /**
     * Classpath for compiling the kernel.
     *
     * <p>Built from the code-source locations of the very classes the generated source imports,
     * rather than from {@code java.class.path}. Under TornadoVM's own launch configuration the API
     * is on the JVM's <em>module</em> path, so {@code java.class.path} does not mention it and a
     * classpath derived from it produces "package uk.ac.manchester.tornado.api does not exist". A
     * protection domain's code source is where the class actually came from, whether that was the
     * class path, the module path, or a user-code loader inside a TaskManager.
     */
    private static String classpathFor() {
        LinkedHashSet<String> entries = new LinkedHashSet<>();
        String systemPath = System.getProperty("java.class.path", "");
        if (!systemPath.isEmpty()) {
            entries.addAll(Arrays.asList(systemPath.split(java.io.File.pathSeparator)));
        }
        for (Class<?> referenced :
                new Class<?>[] {
                    uk.ac.manchester.tornado.api.annotations.Parallel.class,
                    uk.ac.manchester.tornado.api.math.TornadoMath.class,
                    DoubleArray.class,
                    IntArray.class
                }) {
            String location = codeSourceOf(referenced);
            if (location != null) {
                entries.add(location);
            }
        }
        return String.join(java.io.File.pathSeparator, entries);
    }

    private static String codeSourceOf(Class<?> type) {
        try {
            ProtectionDomain domain = type.getProtectionDomain();
            if (domain == null || domain.getCodeSource() == null) {
                return null;
            }
            URL location = domain.getCodeSource().getLocation();
            return location == null ? null : Paths.get(location.toURI()).toString();
        } catch (URISyntaxException | RuntimeException e) {
            return null;
        }
    }

    /** Write side of one staged input column. */
    public void setInput(int column, int position, double value) {
        inputs[column].set(position, value);
    }

    /** Off-heap buffer for one staged input column, for gathers that can bulk-copy into it. */
    public java.lang.foreign.MemorySegment inputSegment(int column) {
        return inputs[column].getSegment();
    }

    public double output(int column, int position) {
        return outputs[column].get(position);
    }

    public boolean selected(int position) {
        return mask == null || mask.get(position) != 0;
    }

    public OffloadMetrics metrics() {
        return metrics;
    }

    /** One batch's device timings, pending the caller's gather and drain numbers. */
    public static final class Execution {
        private final long wallNanos;
        private final TornadoProfilerResult profilerResult;

        Execution(long wallNanos, TornadoProfilerResult profilerResult) {
            this.wallNanos = wallNanos;
            this.profilerResult = profilerResult;
        }
    }

    public Execution execute() {
        long t0 = System.nanoTime();
        TornadoExecutionResult result = withKernelLoader(plan::execute);
        long wall = System.nanoTime() - t0;
        return new Execution(wall, profile ? result.getProfilerResult() : null);
    }

    /**
     * Runs {@code action} with the generated kernel's loader as the context class loader.
     *
     * <p>TornadoVM finds a kernel's {@code @Parallel} annotations by reading its class file as a
     * resource. A class compiled at run time lives in a loader of our making, so unless that loader
     * is reachable the annotation scan comes up empty -- and the failure is silent: the kernel is
     * emitted as a sequential loop which every device thread runs in full, giving correct results
     * about a thousand times slower than it should.
     */
    private <T> T withKernelLoader(java.util.function.Supplier<T> action) {
        Thread current = Thread.currentThread();
        ClassLoader previous = current.getContextClassLoader();
        current.setContextClassLoader(loader);
        try {
            return action.get();
        } finally {
            current.setContextClassLoader(previous);
        }
    }

    public void recordBatch(
            int count, long gatherNanos, Execution execution, int emitted, long drainNanos) {
        metrics.recordBatch(
                count,
                emitted,
                gatherNanos,
                execution.wallNanos,
                drainNanos,
                execution.profilerResult);
    }

    @Override
    public void close() throws Exception {
        if (plan != null) {
            plan.close();
            plan = null;
        }
        if (loader != null) {
            loader.close();
            loader = null;
        }
        if (workDir != null) {
            deleteRecursively(workDir);
            workDir = null;
        }
    }

    private static void deleteRecursively(Path dir) throws IOException {
        if (!Files.exists(dir)) {
            return;
        }
        try (Stream<Path> paths = Files.walk(dir)) {
            for (Path path : (Iterable<Path>) paths.sorted(Comparator.reverseOrder())::iterator) {
                Files.deleteIfExists(path);
            }
        }
    }
}
