"""Everything needed to start a PySpark session with the spark-vector plugin.

spark-vector runs on Spark 4.1 + JDK 25. Two things differ from a stock
`pip install pyspark` session:

1. The JVM must be JDK 25 with the Vector API incubator module enabled.
2. Spark 4.1.3 bundles Hadoop 3.4.2, whose client calls Subject.getSubject and
   fails on JDK 24+ (HADOOP-19212). The drop-in shaded 3.4.3 client jars are put
   on the driver classpath ahead of the bundled ones.

The jars are downloaded once into ./jars and checked against their .sha1 files.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path

from pyspark.sql import SparkSession

SPARK_VECTOR_VERSION = "0.0.1"
HADOOP_VERSION = "3.4.3"

_SPARK_VECTOR_REPO = "https://raw.githubusercontent.com/spark-vector/spark-vector/maven-repo"
_MAVEN_CENTRAL = "https://repo1.maven.org/maven2"

JARS_DIR = Path(__file__).resolve().parent / "jars"

# (file name, base URL of the artifact directory)
_JARS = {
    "plugin": (
        f"spark-vector-spark_2.13-{SPARK_VECTOR_VERSION}.jar",
        f"{_SPARK_VECTOR_REPO}/io/sparkvector/spark-vector-spark_2.13/{SPARK_VECTOR_VERSION}",
    ),
    "hadoop-api": (
        f"hadoop-client-api-{HADOOP_VERSION}.jar",
        f"{_MAVEN_CENTRAL}/org/apache/hadoop/hadoop-client-api/{HADOOP_VERSION}",
    ),
    "hadoop-runtime": (
        f"hadoop-client-runtime-{HADOOP_VERSION}.jar",
        f"{_MAVEN_CENTRAL}/org/apache/hadoop/hadoop-client-runtime/{HADOOP_VERSION}",
    ),
}


def _central(group: str, artifact: str, version: str) -> tuple[str, str]:
    return f"{artifact}-{version}.jar", f"{_MAVEN_CENTRAL}/{group.replace('.', '/')}/{artifact}/{version}"


# The columnar shuffle (Arrow IPC over Arrow Flight). The shuffle jar is not self-contained:
# it needs Arrow Flight and gRPC. This is the runtime closure of spark-vector-shuffle 0.0.1
# minus what PySpark 4.1.3 already ships (Arrow 18.3.0, Netty 4.2, Guava 33.4.8, gson, jsr305,
# zstd-jni...) -- the same set upstream's benchmarks/k8s/Dockerfile adds to Spark's jars.
_SHUFFLE_JARS = {
    "shuffle": (
        f"spark-vector-shuffle_2.13-{SPARK_VECTOR_VERSION}.jar",
        f"{_SPARK_VECTOR_REPO}/io/sparkvector/spark-vector-shuffle_2.13/{SPARK_VECTOR_VERSION}",
    ),
    **{
        artifact: _central(group, artifact, version)
        for group, artifact, version in [
            ("org.apache.arrow", "flight-core", "18.3.0"),
            ("io.grpc", "grpc-api", "1.71.0"),
            ("io.grpc", "grpc-core", "1.71.0"),
            ("io.grpc", "grpc-context", "1.71.0"),
            ("io.grpc", "grpc-netty", "1.71.0"),
            ("io.grpc", "grpc-protobuf", "1.71.0"),
            ("io.grpc", "grpc-protobuf-lite", "1.71.0"),
            ("io.grpc", "grpc-stub", "1.71.0"),
            ("io.grpc", "grpc-util", "1.71.0"),
            ("io.perfmark", "perfmark-api", "0.27.0"),
            ("com.google.api.grpc", "proto-google-common-protos", "2.51.0"),
            ("com.google.protobuf", "protobuf-java", "4.30.2"),
            ("com.google.protobuf", "protobuf-java-util", "4.30.2"),
            ("org.codehaus.mojo", "animal-sniffer-annotations", "1.24"),
            ("com.google.errorprone", "error_prone_annotations", "2.30.0"),
            ("com.google.android", "annotations", "4.1.1.4"),
            ("com.google.j2objc", "j2objc-annotations", "3.0.0"),
            ("org.jspecify", "jspecify", "1.0.0"),
        ]
    },
}

VECTOR_SHUFFLE_MANAGER = "org.apache.spark.sql.vector.shuffle.VectorShuffleManager"

# Needed on every JVM that runs spark-vector kernels (driver and executors).
JVM_OPTIONS = " ".join(
    [
        "--add-modules=jdk.incubator.vector",
        "--enable-native-access=ALL-UNNAMED",
        "--sun-misc-unsafe-memory-access=allow",  # silences Spark/Arrow Unsafe warnings
    ]
)


def _sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch(name: str, base_url: str) -> Path:
    target = JARS_DIR / name
    if target.exists():
        return target
    JARS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {name} ...")
    with urllib.request.urlopen(f"{base_url}/{name}.sha1", timeout=60) as r:
        expected = r.read().decode().split()[0].strip().lower()
    partial = target.with_suffix(".part")
    with urllib.request.urlopen(f"{base_url}/{name}", timeout=300) as r, partial.open("wb") as f:
        shutil.copyfileobj(r, f)
    actual = _sha1(partial)
    if actual != expected:
        partial.unlink()
        raise RuntimeError(f"Checksum mismatch for {name}: expected {expected}, got {actual}")
    partial.rename(target)
    return target


def ensure_jars(columnar_shuffle: bool = False) -> dict[str, Path]:
    wanted = {**_JARS, **(_SHUFFLE_JARS if columnar_shuffle else {})}
    return {key: _fetch(name, url) for key, (name, url) in wanted.items()}


def _java_major(java_home: Path) -> int | None:
    java = java_home / "bin" / "java"
    if not java.exists():
        return None
    out = subprocess.run([str(java), "-version"], capture_output=True, text=True).stderr
    # e.g. 'openjdk version "25.0.4.1" 2026-08-18'
    try:
        return int(out.split('"')[1].split(".")[0])
    except (IndexError, ValueError):
        return None


def ensure_java25() -> Path:
    """Point JAVA_HOME at a JDK 25 before PySpark launches the JVM."""
    candidates = [os.environ.get("JAVA_HOME")]
    try:
        candidates.append(
            subprocess.run(
                ["/usr/libexec/java_home", "-v", "25"], capture_output=True, text=True, check=True
            ).stdout.strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    candidates += ["/opt/homebrew/opt/openjdk@25", "/usr/lib/jvm/java-25-openjdk", "/usr/lib/jvm/java-25"]

    for candidate in filter(None, candidates):
        home = Path(candidate)
        if (home / "libexec/openjdk.jdk/Contents/Home").exists():  # Homebrew layout
            home = home / "libexec/openjdk.jdk/Contents/Home"
        if _java_major(home) == 25:
            os.environ["JAVA_HOME"] = str(home)
            # A stray JAVA_TOOL_OPTIONS (IDEs set one) leaks into Spark's JVM.
            os.environ.pop("JAVA_TOOL_OPTIONS", None)
            return home
    raise RuntimeError(
        "spark-vector needs JDK 25. Install it (e.g. `brew install openjdk@25`) "
        "and/or set JAVA_HOME to it."
    )


def build_session(app_name: str = "spark-vector-example", master: str = "local[*]",
                  driver_memory: str = "4g", columnar_shuffle: bool = False) -> SparkSession:
    """A local SparkSession with spark-vector enabled.

    The plugin reads its keys from the session's SQLConf, so
    `spark.conf.set("spark.vector.enabled", "false")` switches it off per query.

    columnar_shuffle=True also installs spark-vector's shuffle manager, so exchanges
    between spark-vector operators move Arrow record batches instead of rows. The
    manager is static (fixed for the session); it only handles spark-vector's own
    shuffles and leaves every other shuffle to Spark's sort shuffle, so plain-Spark
    queries in the same session (plugin disabled) are unaffected.
    """
    ensure_java25()
    jars = ensure_jars(columnar_shuffle)
    # spark.driver.extraClassPath is prepended by Spark's launcher, so:
    #  - Hadoop 3.4.3 wins over the bundled 3.4.2 (the JDK 24+ fix), and
    #  - the plugin sits on the application classpath, where the Spark UI looks up
    #    the Vector Acceleration tab's static resources (spark.jars is too late for it),
    #    and where SparkEnv instantiates the shuffle manager.
    # In local mode the executor lives in the driver JVM, so this is all it needs. On a
    # cluster, put the jars in the image (upstream's benchmarks/k8s/Dockerfile does).
    driver_cp = os.pathsep.join(str(path) for key, path in jars.items())

    builder = SparkSession.builder
    if columnar_shuffle:
        builder = (
            builder.config("spark.shuffle.manager", VECTOR_SHUFFLE_MANAGER)
            .config("spark.vector.shuffle.enabled", "true")
        )
        if master.startswith("local"):
            # Each executor runs an Arrow Flight server for remote shuffle fetches. It has no
            # TLS and, without spark.authenticate, no auth: in local mode there is no remote
            # executor, so keep it off the network interfaces.
            builder = builder.config("spark.vector.shuffle.flight.bindHost", "127.0.0.1")

    return (
        builder.appName(app_name)
        .master(master)
        .config("spark.driver.memory", driver_memory)
        # The plugin: registers the planner rule and the "Vector Acceleration" UI tab.
        .config("spark.plugins", "io.sparkvector.spark.VectorPlugin")
        .config("spark.driver.extraClassPath", driver_cp)
        .config("spark.driver.extraJavaOptions", JVM_OPTIONS)
        .config("spark.executor.extraJavaOptions", JVM_OPTIONS)
        # The operators consume the vectorized Parquet reader's batches; off-heap vectors
        # let the adapter wrap those lanes in place instead of copying them.
        .config("spark.sql.parquet.enableVectorizedReader", "true")
        .config("spark.sql.columnVector.offheap.enabled", "true")
        .config("spark.sql.shuffle.partitions", str(os.cpu_count() or 8))
        .getOrCreate()
    )
