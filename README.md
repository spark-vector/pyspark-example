# spark-vector PySpark example

Using [spark-vector](https://github.com/spark-vector/spark-vector) from PySpark, with
[uv](https://docs.astral.sh/uv/) managing the environment.

spark-vector is a Spark SQL plugin that replaces Filter, Project, HashAggregate, Sort, joins,
windows and other operators with columnar operators built on the Java Vector API (SIMD, no
native code). Your PySpark code stays the same: the plugin rewrites the physical plan, and
anything it can't convert falls back to Spark.

The example runs three DataFrame queries twice, once on plain Spark and once with spark-vector,
in the same session. For each query it prints the timings, the operators spark-vector took over,
and whether the two results match.

## Requirements

| | |
|---|---|
| uv | `brew install uv`, or see the [install guide](https://docs.astral.sh/uv/getting-started/installation/) |
| JDK 25 | `brew install openjdk@25`, or any JDK 25 set as `JAVA_HOME` |
| Python | 3.10 to 3.13 (uv installs it if missing) |

uv installs PySpark 4.1.3. spark-vector 0.0.1 supports only Spark 4.1 with Scala 2.13, on JDK 25.

## Quick start

```bash
git clone https://github.com/spark-vector/spark-vector-pyspark-example
cd spark-vector-pyspark-example
uv run example.py
```

On the first run:

- `spark_vector_setup.py` downloads the jars into `./jars` and checks each one against its
  published SHA-1.
- `example.py` writes a 20M-row `sales` table and a 1,000-row `stores` table as Parquet into
  `./data` (about 270 MB).

Later runs reuse both folders.

## Options

```bash
uv run example.py --vector-shuffle         # also use spark-vector's columnar shuffle
uv run example.py --keep-alive             # keep the Spark UI up until Enter / Ctrl+C
uv run example.py --rows 50000000 --iterations 5
uv run example.py --queries region_revenue --show-plan
```

| Flag | Default | Meaning |
|---|---|---|
| `--rows` | 20,000,000 | rows in the `sales` table (each size is generated once) |
| `--warmup` / `--iterations` | 1 / 3 | untimed runs, then timed runs; the median is reported |
| `--queries` | all | comma-separated subset of `pricing_summary`, `region_revenue`, `store_ranking` |
| `--show-plan` | off | print the final physical plan of each accelerated run |
| `--vector-shuffle` | off | switch to spark-vector's columnar shuffle (see below) |
| `--keep-alive` | off | leave the session and its UI at http://localhost:4040 running |

With `--keep-alive`, the Spark UI has a **Vector Acceleration** tab. It draws each query's plan
with every operator coloured by the engine that ran it, and lists the reason for every operator
that fell back to Spark.

## The queries

| Query | Shape | Operators spark-vector runs |
|---|---|---|
| `pricing_summary` | TPC-H Q1: selective filter, grouped aggregate, `ORDER BY` | Filter, Project, HashAggregate (partial and final), Sort* |
| `region_revenue` | filter, broadcast join to `stores`, aggregate with `countDistinct`, top 20 | Filter, Project, BroadcastHashJoin, HashAggregate, TakeOrderedAndProject |
| `store_ranking` | join, aggregate, `rank()` per region, filter on the rank | Filter, Project, BroadcastHashJoin, HashAggregate, Sort |

\* The final sort only runs on spark-vector with `--vector-shuffle`. Otherwise its input comes
through Spark's row shuffle, and spark-vector leaves such sorts to Spark.

## Results

Measured on an Apple M3 Pro (11 cores, 18 GB, `local[*]`, 20M rows), median of 3 runs. The
machine was running other jobs, so treat the speedups as indicative.

| Query | Spark | spark-vector | + columnar shuffle |
|---|---|---|---|
| `pricing_summary` | 0.44 s | 0.43 to 0.49 s | 0.37 s (1.19x) |
| `region_revenue` | 0.74 s | 0.82 s | 0.60 s (1.24x) |
| `store_ranking` | 0.45 to 0.54 s | 0.41 s | 0.38 s (1.17x) |

Without the columnar shuffle, every stage boundary converts batches to rows and back, which eats
much of the gain at this data size. Upstream's
[benchmark results](https://github.com/spark-vector/spark-vector/blob/main/docs/results.md) cover
TPC-H and TPC-DS up to 1 TB on a cluster. Every result in this example matched plain Spark's,
to a relative tolerance of 1e-9 for doubles.

## Using it in your own code

Copy `spark_vector_setup.py` into your project:

```python
from spark_vector_setup import build_session

spark = build_session()                         # local[*], plugin enabled
# spark = build_session(columnar_shuffle=True)  # + the columnar shuffle

df = spark.read.parquet("s3a://.../lineitem")   # plain PySpark from here on

spark.conf.set("spark.vector.enabled", "false") # per-query off switch (SQL conf)
```

`build_session` does the following:

| Step | Why |
|---|---|
| Points `JAVA_HOME` at a JDK 25 and clears `JAVA_TOOL_OPTIONS` | the kernels use the Java Vector API; IDEs often set tool options that leak into Spark's JVM |
| Adds `--add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED` to driver and executor JVMs | required by the kernels |
| Sets `spark.plugins=io.sparkvector.spark.VectorPlugin` | registers the planner rule and the UI tab |
| Puts the plugin jar on `spark.driver.extraClassPath`, not `spark.jars` | with `spark.jars` the plugin works, but the UI tab can't find its static resources |
| Puts the Hadoop 3.4.3 client jars on the same classpath | PySpark 4.1.3 bundles Hadoop 3.4.2, which fails on JDK 24+ ([HADOOP-19212](https://issues.apache.org/jira/browse/HADOOP-19212)); the classpath entries take precedence |
| Sets `spark.sql.columnVector.offheap.enabled=true` | spark-vector can read the Parquet reader's batches in place instead of copying them |

The jars come from spark-vector's Maven repository
(`https://raw.githubusercontent.com/spark-vector/spark-vector/maven-repo/`, coordinates
`io.sparkvector:spark-vector-spark_2.13:0.0.1`) and from Maven Central.

### The columnar shuffle

`--vector-shuffle` (or `columnar_shuffle=True`) does the following:

- Adds `spark-vector-shuffle_2.13` and the Arrow Flight and gRPC jars it needs.
  - It skips jars PySpark already ships, such as Arrow, Netty and Guava.
  - The set is the same one upstream's `benchmarks/k8s/Dockerfile` adds to a Spark image.
- Sets two properties:
  - `spark.shuffle.manager=org.apache.spark.sql.vector.shuffle.VectorShuffleManager`
  - `spark.vector.shuffle.enabled=true`

With it on, exchanges between spark-vector operators move compressed Arrow record batches and
never convert to rows. The shuffle manager is fixed for the session. It only handles
spark-vector's own exchanges and leaves the rest to Spark's sort shuffle, so the plain-Spark
baseline in the same session is unaffected.

Security: every executor runs an Arrow Flight server for shuffle fetches. It has no TLS, and it
has no authentication unless `spark.authenticate` is on. In local mode `build_session` binds it
to `127.0.0.1`. On a cluster it must be reachable between executors, so enable
`spark.authenticate`. Where RPC TLS (`spark.ssl.rpc.enabled`) is required, the server refuses to
start; use `spark.vector.shuffle.backend=block` there. See upstream
[`docs/flight-shuffle.md`](https://github.com/spark-vector/spark-vector/blob/main/docs/flight-shuffle.md).

## On a cluster

`build_session` targets `local[*]`, where the executor runs inside the driver JVM. For a cluster,
use the same settings with `spark-submit` and bake the jars into the image:

- Put JDK 25 on every node.
- Replace `hadoop-client-api` and `hadoop-client-runtime` in `$SPARK_HOME/jars` with 3.4.3.
- Put the spark-vector jars (and, for the shuffle, the Flight and gRPC jars) in `$SPARK_HOME/jars`.
- Set `spark.plugins` and both `extraJavaOptions` as above.

Upstream's
[`benchmarks/k8s/Dockerfile`](https://github.com/spark-vector/spark-vector/blob/main/benchmarks/k8s/Dockerfile)
builds such an image. Its README has a memory tuning section: the operators keep their tables on
the heap and their batches in Arrow direct memory, so set `-XX:MaxDirectMemorySize` explicitly.

## What doesn't get accelerated

spark-vector only takes columnar input, meaning Spark's vectorized Parquet reader, a Comet or
Iceberg vectorized scan, or another spark-vector operator. The following stay on Spark:

- `spark.range(...)`, `createDataFrame` from Python objects, and cached tables (`df.cache()`)
- Python UDFs and pandas UDFs, and writes (`df.write`)
- regular expressions, `collect_list`/`percentile`, and nested-type accessors

The full lists are in upstream's
[README](https://github.com/spark-vector/spark-vector#requirements-and-known-limitations) and
[`docs/expressions.md`](https://github.com/spark-vector/spark-vector/blob/main/docs/expressions.md).
To see why an operator fell back, use the UI tab or set
`spark.vector.explainFallback.enabled=true`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `spark-vector needs JDK 25` | install JDK 25 or set `JAVA_HOME` to one |
| `UnsupportedOperationException: getSubject is supported only if a security manager is allowed` | the Hadoop 3.4.2 jars are loaded first: use `build_session`, or swap the jars in `$SPARK_HOME/jars` |
| `could not attach the Vector Acceleration tab` | the plugin jar is only on `spark.jars`; put it on `spark.driver.extraClassPath` |
| Everything falls back to Spark | the input isn't columnar (see above); check `spark.sql.parquet.enableVectorizedReader` |
| Checksum mismatch while downloading | delete `./jars` and rerun |

## Layout

```
example.py              the three queries, data generator and Spark vs spark-vector harness
spark_vector_setup.py   JDK 25 lookup, jar download + SHA-1 check, build_session()
pyproject.toml, uv.lock pinned PySpark 4.1.3
jars/, data/            created on first run (git-ignored)
```

## License

Apache License 2.0, the same as spark-vector.
