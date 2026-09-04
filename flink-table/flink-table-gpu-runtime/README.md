# Flink Table GPU runtime

Query-independent runtime for transparent GPU offload of Flink SQL, executed through TornadoVM.

The planner decides *what* to offload (`flink-table-planner`, package `plan.gpu`); this module
does it. The two never meet at compile time: the planner discovers
`GpuOperatorFactoryProvider` through `META-INF/services`, so it carries no reference to TornadoVM
and a cluster without this jar simply runs everything on the CPU.

## Contents

| | |
|---|---|
| `gather/` | staging strategies per input layout — columnar bulk copy, binary row transpose, generic accessors |
| `operator/` | `GeneratedKernelEngine` (javac, buffers, task graph) and `GpuCalcOperator` (the Flink operator) |
| `provider/` | the `ServiceLoader` entry point |
| `metrics/` | the gather / copy-in / kernel / copy-out breakdown |
| `GeneratedKernelSweep` | standalone benchmark, no Flink job — arithmetic-intensity sweep against the cost floor |

## Building

Part of the default build. TornadoVM is not published to Maven Central, so its API jars must be
installed locally first or the reactor will fail to resolve them:

```bash
D=$TORNADO_SDK/share/java/tornado
for a in tornado-api tornado-annotation; do
  mvn install:install-file -Dfile=$D/$a-6.0.1-jdk21-dev.jar \
    -DgroupId=uk.ac.manchester.tornado -DartifactId=$a \
    -Dversion=6.0.1-jdk21-dev -Dpackaging=jar
done

mvn clean package -DskipTests
```

No profile or extra argument is needed. The module was originally behind a `-Pgpu` profile; that
was removed because it kept the module out of the reactor and therefore out of IntelliJ's project
structure, where it appeared as loose class files with no configurable source root.

Note that `activeByDefault` would not have been a fix: Maven deactivates such profiles as soon as
any `-P` is given on the command line, and the documented build uses `-Pjava21-target`.

Use `clean install` when changing the TornadoVM version. The `writeReplace()` that lets TornadoVM
resolve a kernel from its method reference is emitted at compile time against a specific
`tornado-api`; Maven does not recompile unchanged sources for a version bump, and the stale classes
fail at run time with `Kernel entry ... has no writeReplace()`.

This module compiles at Java 21 with `--enable-preview`, overriding the repository-wide source
level 11: TornadoVM's off-heap array types are built on `java.lang.foreign`, a preview API on 21.

## Running

TornadoVM's JVM arguments must reach every JVM that touches the plan -- the client builds it, the
JobManager holds it, the TaskManager runs the kernel -- via `env.java.opts` in `conf/config.yaml`
or as VM options in an IDE. The SDK's `tornado-argfile.template` carries them; expand
`${TORNADOVM_HOME}` in it rather than copying the flags by hand, since it is generated from
`tornado --printJavaFlags` and stays correct across TornadoVM versions.

From an IDE, see `flink-examples/flink-examples-table/.../java/gpu/GpuOffloadExample.java`. It is a
correctness demonstration, not a performance one: its expression is two flops and it forces the
cost floor down to admit it.

### Benchmark on a standalone cluster

`GpuOffloadExample` runs in a MiniCluster, which hides everything deployment does -- per-subtask
kernel compilation, device contention, the client round trip. For numbers that reflect what a user
sees, `scripts/` drives a real cluster:

```bash
export TORNADOVM_HOME=/path/to/tornadovm-sdk
DIST=build-target                       # or flink-dist/target/flink-<version>-bin/flink-<version>

scripts/gpu-cluster-setup.sh "$DIST"    # gpu-runtime opt/ -> lib/, TornadoVM flags into config.yaml
scripts/run-haversine.sh    "$DIST" 50000000 1 10
```

`run-haversine.sh` starts the cluster, writes the input to Parquet once if it is not already
there, then runs the query ten times with offload off and ten times with it on, through
`bin/flink run` exactly as a user would.

Two choices in there are not incidental. The source is **Parquet, not `datagen`**: generating rows
costs about 100 us each, which buries the operator whatever the kernel does, and Parquet also hands
the operator `ColumnarRowData`, the layout the columnar gather exists for. The expression is
**haversine**, which the estimator weighs well above the default floor -- unlike `val * 2.0 + 1.0`,
which is below it and has nothing for the GPU to win back.

`FINDINGS.md` has the measurements behind the cost floor.
