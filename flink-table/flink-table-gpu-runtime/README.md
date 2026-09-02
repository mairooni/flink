# Flink Table GPU runtime

Query-independent runtime for transparent GPU offload of Flink SQL, executed through TornadoVM.

The planner decides *what* to offload (`flink-table-planner`, package `plan.gpu`); this module
does it. The two never meet at compile time: the planner discovers
`GpuOperatorFactoryProvider` through `META-INF/services`, so it carries no reference to TornadoVM
and a cluster without this jar simply runs everything on the CPU.

## Contents

| | |
|---|---|
| `kernels/` | javac-compiled TornadoVM kernels. Must be javac-compiled: TornadoVM resolves a kernel from its method reference's `SerializedLambda`, so a runtime-generated class cannot be used |
| `gather/` | staging strategies per input layout — columnar bulk copy, binary row transpose, generic accessors |
| `operator/` | `FilterProjectEngine` (buffers plus task graph) and `GpuCalcOperator` (the Flink operator) |
| `provider/` | the `ServiceLoader` entry point |
| `metrics/` | the gather / copy-in / kernel / copy-out breakdown |
| `Harness` | standalone benchmark, no Flink job — used to calibrate the cost floor |

## Building

Not in the default build, because TornadoVM is not published to Maven Central. Install it first:

```bash
D=$TORNADO_SDK/share/java/tornado
for a in tornado-api tornado-annotation; do
  mvn install:install-file -Dfile=$D/$a-5.2.1-jdk21-dev.jar \
    -DgroupId=uk.ac.manchester.tornado -DartifactId=$a \
    -Dversion=5.2.1-jdk21-dev -Dpackaging=jar
done

mvn -Pgpu -pl flink-table/flink-table-gpu-runtime -am -DskipTests install
```

This module compiles at Java 21 with `--enable-preview`, overriding the repository-wide source
level 11: TornadoVM's off-heap array types are built on `java.lang.foreign`, a preview API on 21.

## Running

TornadoVM's JVM arguments must reach the TaskManagers, via `env.java.opts` in `flink-conf.yaml` or
as VM options in an IDE. The shipped `tornado-argfile` carries them.

See `flink-examples/flink-examples-table/.../java/gpu/GpuOffloadExample.java`, and `FINDINGS.md`
for the measurements behind the cost floor.
