import argparse
import tempfile
import time
from pathlib import Path

import duckdb

from lambdatune import PG

# Không tạo khóa chính hay chỉ mục nào: kịch bản "No initial indexes" của bài báo
DDL = """
DROP TABLE IF EXISTS lineitem, orders, partsupp, part, customer, supplier, nation, region CASCADE;
CREATE TABLE region   (r_regionkey INT, r_name CHAR(25), r_comment VARCHAR(152));
CREATE TABLE nation   (n_nationkey INT, n_name CHAR(25), n_regionkey INT, n_comment VARCHAR(152));
CREATE TABLE part     (p_partkey INT, p_name VARCHAR(55), p_mfgr CHAR(25), p_brand CHAR(10),
                       p_type VARCHAR(25), p_size INT, p_container CHAR(10),
                       p_retailprice DECIMAL(15,2), p_comment VARCHAR(23));
CREATE TABLE supplier (s_suppkey INT, s_name CHAR(25), s_address VARCHAR(40), s_nationkey INT,
                       s_phone CHAR(15), s_acctbal DECIMAL(15,2), s_comment VARCHAR(101));
CREATE TABLE partsupp (ps_partkey INT, ps_suppkey INT, ps_availqty INT,
                       ps_supplycost DECIMAL(15,2), ps_comment VARCHAR(199));
CREATE TABLE customer (c_custkey INT, c_name VARCHAR(25), c_address VARCHAR(40), c_nationkey INT,
                       c_phone CHAR(15), c_acctbal DECIMAL(15,2), c_mktsegment CHAR(10),
                       c_comment VARCHAR(117));
CREATE TABLE orders   (o_orderkey INT, o_custkey INT, o_orderstatus CHAR(1),
                       o_totalprice DECIMAL(15,2), o_orderdate DATE, o_orderpriority CHAR(15),
                       o_clerk CHAR(15), o_shippriority INT, o_comment VARCHAR(79));
CREATE TABLE lineitem (l_orderkey INT, l_partkey INT, l_suppkey INT, l_linenumber INT,
                       l_quantity DECIMAL(15,2), l_extendedprice DECIMAL(15,2),
                       l_discount DECIMAL(15,2), l_tax DECIMAL(15,2), l_returnflag CHAR(1),
                       l_linestatus CHAR(1), l_shipdate DATE, l_commitdate DATE,
                       l_receiptdate DATE, l_shipinstruct CHAR(25), l_shipmode CHAR(10),
                       l_comment VARCHAR(44));
"""


TABLES = ["region", "nation", "part", "supplier", "partsupp", "customer", "orders", "lineitem"]


def copy_table(duck, db, table, folder):
    """Xuất một bảng từ DuckDB ra CSV trong `folder`, rồi nạp CSV đó vào PostgreSQL bằng COPY."""
    path = Path(folder) / f"{table}.csv"
    duck.execute(f"COPY {table} TO '{path}' (HEADER false, DELIMITER '|')")
    with db.conn.cursor() as cur:
        with open(path) as f:
            cur.copy_expert(f"COPY {table} FROM STDIN WITH (FORMAT csv, DELIMITER '|')", f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf", type=float, default=1.0)
    args = ap.parse_args()

    # Bước 1: sinh dữ liệu TPC-H trong DuckDB (trong bộ nhớ)
    duck = duckdb.connect()
    duck.execute("INSTALL tpch; LOAD tpch;")
    duck.execute(f"CALL dbgen(sf={args.sf})")

    # Bước 2: tạo lại các bảng trống trong PostgreSQL
    db = PG()
    db.sql(DDL)

    # Bước 3: chép từng bảng qua tệp CSV tạm
    with tempfile.TemporaryDirectory() as tmp:
        for table in TABLES:
            t0 = time.time()
            copy_table(duck, db, table, tmp)
            print(f"{table:<9} {time.time() - t0:6.1f} s")

    # Bước 4: cập nhật thống kê cho optimizer
    db.sql("VACUUM ANALYZE")


if __name__ == "__main__":
    main()
