"""PySpark + spark-vector: the same DataFrame queries with and without the plugin.

    uv run example.py                 # 20M-row fact table, 3 queries, Spark vs spark-vector
    uv run example.py --rows 50000000 --iterations 5
    uv run example.py --keep-alive    # leave the Spark UI (Vector Acceleration tab) up
    uv run example.py --vector-shuffle  # + spark-vector's columnar shuffle between stages

Nothing in the query code is spark-vector specific: the plugin rewrites the
physical plan (Filter, Project, HashAggregate, Sort, joins, windows) into SIMD
operators and leaves anything it cannot convert to Spark.
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
import threading
import time
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from spark_vector_setup import build_session

DATA_DIR = Path(__file__).resolve().parent / "data"


# --------------------------------------------------------------------------- data

def generate_data(spark: SparkSession, rows: int) -> None:
    """A small star schema in Parquet: a `sales` fact table and a `stores` dimension."""
    sales_path = DATA_DIR / f"sales_{rows}"
    stores_path = DATA_DIR / "stores"
    if sales_path.exists() and stores_path.exists():
        return
    print(f"Generating {rows:,} sales rows into {DATA_DIR} ...")
    n_stores = 1_000
    (
        spark.range(n_stores)
        .select(
            F.col("id").cast("int").alias("store_id"),
            F.concat(F.lit("store-"), F.col("id")).alias("store_name"),
            F.element_at(F.array(*[F.lit(r) for r in ("NORTH", "SOUTH", "EAST", "WEST", "CENTRAL")]),
                         (F.col("id") % 5 + 1).cast("int")).alias("region"),
        )
        .coalesce(1).write.mode("overwrite").parquet(str(stores_path))
    )
    status = F.element_at(F.array(F.lit("A"), F.lit("F"), F.lit("N"), F.lit("R")),
                          (F.col("id") % 4 + 1).cast("int"))
    (
        spark.range(rows)
        .select(
            F.col("id").alias("order_id"),
            (F.hash("id") % n_stores).cast("int").alias("raw_store"),
            status.alias("status"),
            F.element_at(F.array(F.lit("AIR"), F.lit("MAIL"), F.lit("RAIL"), F.lit("SHIP"), F.lit("TRUCK")),
                         (F.abs(F.hash("id", F.lit(7))) % 5 + 1).cast("int")).alias("ship_mode"),
            (F.abs(F.hash("id", F.lit(1))) % 50 + 1).cast("int").alias("quantity"),
            (F.abs(F.hash("id", F.lit(2))) % 100_000 / 100.0 + 1.0).alias("price"),
            (F.abs(F.hash("id", F.lit(3))) % 11 / 100.0).alias("discount"),
            (F.abs(F.hash("id", F.lit(4))) % 9 / 100.0).alias("tax"),
            F.date_add(F.lit("2020-01-01").cast("date"),
                       (F.abs(F.hash("id", F.lit(5))) % 2000).cast("int")).alias("ship_date"),
        )
        .withColumn("store_id", F.abs(F.col("raw_store")))
        .drop("raw_store")
        .write.mode("overwrite").parquet(str(sales_path))
    )


# ------------------------------------------------------------------------ queries

def q_pricing_summary(spark: SparkSession, rows: int) -> DataFrame:
    """TPC-H Q1 shape: a selective filter and a wide grouped aggregate."""
    s = spark.read.parquet(str(DATA_DIR / f"sales_{rows}"))
    disc_price = F.col("price") * (1 - F.col("discount"))
    return (
        s.where(F.col("ship_date") <= F.lit("2024-09-02").cast("date"))
        .groupBy("status", "ship_mode")
        .agg(
            F.sum("quantity").alias("sum_qty"),
            F.sum("price").alias("sum_base_price"),
            F.sum(disc_price).alias("sum_disc_price"),
            F.sum(disc_price * (1 + F.col("tax"))).alias("sum_charge"),
            F.avg("quantity").alias("avg_qty"),
            F.avg("discount").alias("avg_disc"),
            F.count(F.lit(1)).alias("count_order"),
        )
        .orderBy("status", "ship_mode")
    )


def q_region_revenue(spark: SparkSession, rows: int) -> DataFrame:
    """A broadcast hash join against the dimension, then aggregate + top-N."""
    s = spark.read.parquet(str(DATA_DIR / f"sales_{rows}"))
    st = spark.read.parquet(str(DATA_DIR / "stores"))
    return (
        s.where((F.col("quantity") > 10) & (F.col("discount").between(0.02, 0.08)))
        .join(F.broadcast(st), "store_id")
        .groupBy("region", "store_name", F.year("ship_date").alias("year"))
        .agg(F.sum(F.col("price") * F.col("quantity")).alias("revenue"),
             F.countDistinct("ship_mode").alias("modes"))
        .orderBy(F.desc("revenue"), "store_name", "year")
        .limit(20)
    )


def q_store_ranking(spark: SparkSession, rows: int) -> DataFrame:
    """Aggregate, then a ranking window per region and a filter on the rank."""
    s = spark.read.parquet(str(DATA_DIR / f"sales_{rows}"))
    st = spark.read.parquet(str(DATA_DIR / "stores"))
    per_store = (
        s.join(F.broadcast(st), "store_id")
        .groupBy("region", "store_id")
        .agg(F.sum(F.col("price") * (1 - F.col("discount"))).alias("net"),
             F.max("quantity").alias("max_qty"))
    )
    w = Window.partitionBy("region").orderBy(F.desc("net"), "store_id")
    return (
        per_store.withColumn("rank", F.rank().over(w))
        .where(F.col("rank") <= 3)
        .orderBy("region", "rank")
    )


QUERIES = {
    "pricing_summary": q_pricing_summary,
    "region_revenue": q_region_revenue,
    "store_ranking": q_store_ranking,
}


# ---------------------------------------------------------------------- harness

def _rows_equal(a: list, b: list, rel_tol: float = 1e-9) -> bool:
    """Spark merges partial aggregates in shuffle order, so doubles may differ in the last bits."""
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        for va, vb in zip(ra, rb):
            if isinstance(va, float) and isinstance(vb, float):
                if not math.isclose(va, vb, rel_tol=rel_tol):
                    return False
            elif va != vb:
                return False
    return True


def _final_plan(df: DataFrame) -> str:
    # After an action, AQE's executedPlan prints the final (isFinalPlan=true) plan.
    return df._jdf.queryExecution().executedPlan().toString()


def _operator_summary(plan: str) -> str:
    ops = re.findall(r"^[\s:+\-*()\d]*([A-Z][A-Za-z]+)", plan, flags=re.MULTILINE)
    vector = sorted({o for o in ops if o.startswith("Vector")})
    return ", ".join(vector) if vector else "none (everything fell back to Spark)"


def run(spark: SparkSession, name: str, build, rows: int, vector: bool,
        warmup: int, iterations: int) -> tuple[float, list, str]:
    spark.conf.set("spark.vector.enabled", str(vector).lower())
    result, df = None, None
    times = []
    for i in range(warmup + iterations):
        df = build(spark, rows)
        start = time.perf_counter()
        result = df.collect()
        elapsed = time.perf_counter() - start
        if i >= warmup:
            times.append(elapsed)
    return statistics.median(times), result, _final_plan(df)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=20_000_000, help="rows in the sales fact table")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--queries", default=",".join(QUERIES), help="comma-separated subset")
    parser.add_argument("--show-plan", action="store_true", help="print each accelerated physical plan")
    parser.add_argument("--keep-alive", action="store_true", help="keep the Spark UI up at the end")
    parser.add_argument("--vector-shuffle", action="store_true",
                        help="also use spark-vector's columnar shuffle (Arrow IPC / Flight)")
    args = parser.parse_args()

    spark = build_session(columnar_shuffle=args.vector_shuffle)
    if args.vector_shuffle:
        print(f"shuffle manager: {spark.sparkContext.getConf().get('spark.shuffle.manager')}")
    spark.sparkContext.setLogLevel("WARN")
    print(f"Spark {spark.version}, UI at {spark.sparkContext.uiWebUrl} (see the 'Vector Acceleration' tab)")

    spark.conf.set("spark.vector.enabled", "false")
    generate_data(spark, args.rows)

    summary = []
    for name in args.queries.split(","):
        build = QUERIES[name]
        t_spark, r_spark, _ = run(spark, name, build, args.rows, False, args.warmup, args.iterations)
        t_vec, r_vec, plan = run(spark, name, build, args.rows, True, args.warmup, args.iterations)
        same = _rows_equal(r_spark, r_vec)
        summary.append((name, t_spark, t_vec, same))
        print(f"\n== {name}: Spark {t_spark:.2f}s, spark-vector {t_vec:.2f}s "
              f"({t_spark / t_vec:.2f}x), results match: {same}")
        print(f"   accelerated operators: {_operator_summary(plan)}")
        if args.show_plan:
            print(plan)
        for row in r_vec[:5]:
            print("  ", row)

    print("\nquery              spark(s)  vector(s)  speedup  same")
    for name, ts, tv, same in summary:
        print(f"{name:<18} {ts:8.2f} {tv:10.2f} {ts / tv:8.2f}x  {same}")

    if args.keep_alive:
        print(f"\nSpark UI: {spark.sparkContext.uiWebUrl} -- press Enter (or Ctrl+C) to stop", flush=True)
        try:
            input()
        except EOFError:
            # No terminal attached (run in the background): wait until the process is stopped.
            try:
                threading.Event().wait()
            except KeyboardInterrupt:
                pass
        except KeyboardInterrupt:
            pass
    spark.stop()


if __name__ == "__main__":
    main()
