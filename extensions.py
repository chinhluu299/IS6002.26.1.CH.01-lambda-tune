import math
import re

import psycopg2
import sqlglot
from sqlglot import exp

from lambdatune import TEMPLATE, compress, join_block, join_values, tokens, walk


# ============================================================================= D1: prompt mở rộng

# Phép so sánh của sqlglot → cách viết toán tử trong prompt
OPS = {
    exp.EQ: "=",
    exp.NEQ: "<>",
    exp.LT: "<",
    exp.LTE: "<=",
    exp.GT: ">",
    exp.GTE: ">=",
    exp.Between: "BETWEEN",
    exp.In: "IN",
    exp.Like: "LIKE",
    exp.ILike: "LIKE",
}

# Thứ tự in các toán tử trong một dòng snippet lọc
ORDER = ["=", "<>", "<", "<=", ">", ">=", "BETWEEN", "IN", "LIKE"]

# Hằng đứng bên trái ('1995-01-01' < c) thì đảo toán tử để cột đứng trước (c > '1995-01-01')
FLIP = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}


def constant(node):
    """True nếu `node` là hằng hoặc biểu thức không chứa cột và truy vấn con,
    vd. CAST('1994-01-01' AS date) + INTERVAL '1' year."""
    if node is None:
        return False
    if node.find(exp.Column) is not None:
        return False
    if node.find(exp.Select) is not None:
        return False
    return True


def tables_by_column(tables):
    """Trả về {cột: [các bảng có cột đó]}."""
    owner = {}
    for table, columns in tables.items():
        for column in columns:
            if column not in owner:
                owner[column] = []
            owner[column].append(table)
    return owner


def column_compared_with_constant(node):
    """Phép so sánh `node` có dạng "cột <op> hằng" (hoặc "hằng <op> cột") không.
    Có thì trả về (cột, toán tử); không thì trả về (None, None)."""
    op = OPS[type(node)]

    if isinstance(node, exp.In):
        # c IN (hằng, hằng, ...); IN (truy vấn con) thì danh sách giá trị rỗng
        column = node.this
        values = node.expressions
        if len(values) == 0:
            return None, None
        for value in values:
            if not constant(value):
                return None, None

    elif isinstance(node, exp.Between):
        # c BETWEEN hằng AND hằng
        column = node.this
        if not constant(node.args.get("low")):
            return None, None
        if not constant(node.args.get("high")):
            return None, None

    elif isinstance(node.expression, exp.Column) and constant(node.this):
        # hằng <op> c: đảo toán tử
        column = node.expression
        if op in FLIP:
            op = FLIP[op]

    else:
        # c <op> hằng
        column = node.this
        if not constant(node.expression):
            return None, None

    if not isinstance(column, exp.Column):
        return None, None  # vd. substring(c_phone, 1, 2) IN (...) của Q22
    return column, op


def table_of_column(column, alias, owner):
    """Bảng thật của một cột trong truy vấn; None nếu không xác định được."""
    if column.table != "":
        # Cột có ghi bảng hoặc alias, vd. n1.n_name → nation. Alias của CTE (vd. revenue) không phải bảng thật.
        return alias.get(column.table)
    # Cột không ghi bảng: chỉ nhận khi đúng một bảng có cột này (TPC-H đặt tên cột theo tiền tố bảng)
    candidates = owner.get(column.name, [])
    if len(candidates) == 1:
        return candidates[0]
    return None


def filter_predicates(sql, tables):
    """Vị từ cột–hằng của một truy vấn, tách bằng sqlglot. Trả về {"bảng.cột": {các toán tử}}.

    tables = {bảng: {cột}}. Vị từ cột–cột (join, l_commitdate < l_receiptdate) và so sánh với truy vấn con
    (Q2, Q15, Q17) bị bỏ qua.
    """
    owner = tables_by_column(tables)
    tree = sqlglot.parse_one(sql, read="postgres")

    # Bước 1: alias trong truy vấn → bảng thật (n1 → nation, l1 → lineitem)
    alias = {}
    for table_node in tree.find_all(exp.Table):
        if table_node.name in tables:
            alias[table_node.alias_or_name] = table_node.name

    # Bước 2: duyệt mọi phép so sánh, giữ các phép so sánh một cột với hằng
    found = {}
    for node in tree.find_all(*OPS):
        # Chỉ lấy vị từ trong WHERE / HAVING / JOIN ON, không lấy CASE WHEN của danh sách SELECT (Q12)
        if node.find_ancestor(exp.Where, exp.Having, exp.Join) is None:
            continue
        if node.find_ancestor(exp.Case) is not None:
            continue

        column, op = column_compared_with_constant(node)
        if column is None:
            continue

        # Bước 3: tìm bảng của cột
        table = table_of_column(column, alias, owner)
        if table is None:
            continue
        if column.name not in tables[table]:
            continue

        # Bước 4: ghi lại toán tử cho cột
        key = table + "." + column.name
        if key not in found:
            found[key] = set()
        found[key].add(op)
    return found


def filter_values(queries, plans, tables):
    """D1: V(f) = tổng "Total Cost" của các node quét bảng (Seq/Index/Bitmap Heap Scan) có điều kiện trên cột f, trong
    kế hoạch mặc định; cùng thước đo với V(p) của join (§3.2).

    queries và plans có cùng thứ tự. Trả về {"bảng.cột": (V(f), "bảng.cột: <, BETWEEN")}.
    """
    value = {}  # cột f → V(f)
    ops = {}    # cột f → các toán tử dùng với f

    for sql, plan in zip(queries, plans):
        predicates = filter_predicates(sql, tables)
        for node in walk(plan):
            relation = node.get("Relation Name")  # None nếu node không quét bảng
            condition = node.get("Filter", "") + " " + node.get("Index Cond", "") + " " + node.get("Recheck Cond", "")
            for f, f_ops in predicates.items():
                table, column = f.split(".")
                if table != relation:
                    continue
                if re.search(r"\b" + column + r"\b", condition) is None:
                    continue
                if f not in value:
                    value[f] = 0.0
                    ops[f] = set()
                value[f] = value[f] + node["Total Cost"]
                ops[f].update(f_ops)

    # Viết mỗi snippet lọc thành một dòng "bảng.cột: toán tử, toán tử"
    snippets = {}
    for f in value:
        ops_in_order = []
        for op in ORDER:
            if op in ops[f]:
                ops_in_order.append(op)
        line = f + ": " + ", ".join(ops_in_order)
        snippets[f] = (value[f], line)
    return snippets


def human(n):
    """6001215 → "6.0M": làm tròn để số dòng ước lượng (reltuples) thay đổi nhẹ không làm đổi prompt."""
    units = [(1e9, "B"), (1e6, "M"), (1e3, "K")]
    for divisor, suffix in units:
        if n >= divisor:
            scaled = n / divisor
            if n < 10 * divisor:
                return f"{scaled:.1f}{suffix}"
            else:
                return f"{scaled:.0f}{suffix}"
    return str(int(n))


def table_sizes(db):
    """D1: khối kích thước bảng (số dòng, MB), bảng lớn trước."""
    rows = db.sql("SELECT relname, reltuples, pg_table_size(oid) FROM pg_class WHERE relkind = 'r' "
                  "AND relnamespace = 'public'::regnamespace ORDER BY reltuples DESC", fetch=True)
    lines = ["Table sizes:"]
    for table, row_count, size_bytes in rows:
        if row_count < 0:
            row_count = 0  # bảng chưa được ANALYZE có reltuples = -1
        size_mb = size_bytes // 2**20
        lines.append(f"{table}: {human(row_count)} rows, {size_mb} MB")
    return "\n".join(lines)


def generate_prompt_ext(db, queries, memory_gb, cores, budget=math.inf, dbms="PostgreSQL"):
    """D1: như generate_prompt (§3.1), thêm snippet lọc vào cùng ILP và khối kích thước bảng."""
    # Bước 1: kế hoạch mặc định (EXPLAIN) của mọi truy vấn
    plans = []
    for sql in queries.values():
        plans.append(db.explain(sql))

    # Bước 2: giá trị snippet join V(p) và snippet lọc V(f)
    join_value = join_values(plans)
    filter_snippets = filter_values(queries.values(), plans, db.table_columns())

    # Bước 3: khối kích thước bảng luôn có mặt, nên số token của nó được trừ trước khỏi ngân sách B
    sizes = table_sizes(db)
    remaining_budget = budget - tokens(sizes)
    if remaining_budget < 0:
        remaining_budget = 0

    # Bước 4: ILP chọn snippet join và snippet lọc trong cùng ngân sách
    join_lines, kept_filters = compress(join_value, remaining_budget, filter_snippets)

    # Bước 5: ghép phần mô tả workload: join, rồi lọc (cột đắt nhất trước), rồi kích thước bảng
    workload = join_block(join_lines)
    if len(kept_filters) > 0:
        def filter_cost(f):
            return filter_snippets[f][0]

        kept_filters = sorted(kept_filters, key=filter_cost, reverse=True)
        filter_lines = []
        for f in kept_filters:
            filter_lines.append(filter_snippets[f][1])
        workload = workload + "\n\nEach row in the following list has the following format:\n"
        workload = workload + "{a column C}:{the comparison operators applied to C in filter predicates} (most expensive first)\n"
        workload = workload + "\n".join(filter_lines)
    workload = workload + "\n\n" + sizes

    return TEMPLATE.format(dbms=dbms, workload=workload, memory=f"{memory_gb}GB", cores=cores)


# ============================================================================= D2: vòng phản hồi

REFINE = """{prompt}

We evaluated the candidate configurations on this workload. The best one runs the whole workload in {best_time:.2f} s:
{best}
{problems}
The slowest queries under this configuration, with the most expensive operators of their actual plans (EXPLAIN ANALYZE):
{slow}

Propose an improved complete configuration that keeps what works and fixes these bottlenecks. Your response should strictly consist of a JSON object {{"commands": [list of SQL commands]}}, without any additional text."""


def node_time(node, processes):
    """Thời gian thực (giây) của một node EXPLAIN ANALYZE = Actual Total Time × Actual Loops / số tiến trình.
    Dưới Gather, mỗi tiến trình (các worker và leader) chạy song song một loop, nên phải chia cho số tiến trình."""
    total_ms = node.get("Actual Total Time", 0.0) * node.get("Actual Loops", 1)
    return total_ms / processes / 1000


def describe_node(node, self_time):
    """Một dòng mô tả node, vd. "Seq Scan on lineitem [Filter: ...] (rows=..., self time 0.84 s)"."""
    label = node["Node Type"]
    if "Relation Name" in node:
        label = label + " on " + node["Relation Name"]
    if "Index Name" in node:
        label = label + " using " + node["Index Name"]
    for key in ("Hash Cond", "Merge Cond", "Index Cond", "Join Filter", "Filter"):
        if key in node:
            label = label + f" [{key}: {node[key][:100]}]"
            break  # chỉ in điều kiện đầu tiên tìm thấy
    rows = node.get("Actual Rows", 0) * node.get("Actual Loops", 1)
    label = label + f" (rows={rows}, self time {max(self_time, 0):.2f} s)"
    return label


def collect_nodes(node, processes, result):
    """Duyệt kế hoạch theo chiều sâu; thêm (thời gian riêng, mô tả) của từng node vào danh sách `result`.
    Thời gian riêng của node = thời gian của nó trừ thời gian các node con."""
    children = node.get("Plans", [])

    # Các node con của Gather / Gather Merge chạy song song trên (số worker + 1 leader) tiến trình
    if node["Node Type"] == "Gather" or node["Node Type"] == "Gather Merge":
        child_processes = node.get("Workers Launched", 0) + 1
    else:
        child_processes = processes

    child_times = []
    for child in children:
        child_times.append(node_time(child, child_processes))
    self_time = node_time(node, processes) - sum(child_times)

    result.append((self_time, describe_node(node, self_time)))
    for child in children:
        collect_nodes(child, child_processes, result)


def summarize_plan(plan, limit=4):
    """`limit` toán tử có thời gian riêng lớn nhất của một kế hoạch EXPLAIN ANALYZE, nối bằng "; "."""
    nodes = []
    collect_nodes(plan, 1, nodes)
    nodes.sort(reverse=True)  # theo thời gian riêng giảm dần (bằng nhau thì theo mô tả)

    parts = []
    for self_time, label in nodes[:limit]:
        parts.append(label)
    return "; ".join(parts)


def explain_analyze(db, sql, timeout):
    """Kế hoạch thực tế (EXPLAIN ANALYZE, FORMAT JSON) của một truy vấn; None nếu chạy quá `timeout` giây."""
    db.sql(f"SET statement_timeout = {int(timeout * 1000)}")
    try:
        rows = db.sql("EXPLAIN (ANALYZE, FORMAT JSON) " + sql, fetch=True)
        return rows[0][0][0]
    except psycopg2.errors.QueryCanceled:
        return None
    finally:
        db.sql("RESET statement_timeout")


def slowest_queries(times, top):
    """Tên `top` truy vấn chậm nhất trong {truy vấn: giây}."""
    def time_of(name):
        return times[name]

    ordered = sorted(times, key=time_of, reverse=True)
    return ordered[:top]


def feedback_prompt(db, queries, prompt, config, best_time, rejected=(), top=3):
    """D2: chạy lại cấu hình tốt nhất b một lần với EXPLAIN ANALYZE, rồi ghép vào sau prompt gốc: các lệnh của b, các
    lệnh bị bộ kiểm định (D3) hoặc DBMS từ chối / kẹp, `top` truy vấn chậm nhất và các toán tử tốn thời gian nhất của
    chúng. Trả về (prompt phản hồi, {truy vấn: giây}).
    """
    settings, indexes = config

    # Bước 1: áp dụng tham số của b; ghi lại các lệnh bị D3 (từ trước) và DBMS (bây giờ) từ chối
    problems = list(rejected)
    errors = db.set_config(settings)
    problems.extend(errors)

    # Bước 2: tạo sẵn mọi chỉ mục của b
    for ix in indexes:
        try:
            db.sql(ix.sql)
        except psycopg2.Error as e:
            first_line = str(e).splitlines()[0]
            problems.append(ix.sql + ": " + first_line)

    # Bước 3: chạy mọi truy vấn với EXPLAIN ANALYZE. Dưới b cả workload mất C_b, nên một truy vấn không thể lâu hơn
    # nhiều; timeout chỉ để phòng trường hợp bất thường.
    timeout = 2 * best_time + 10
    plans = {}
    for name, sql in queries.items():
        plans[name] = explain_analyze(db, sql, timeout)

    # Bước 4: trả DBMS về cấu hình mặc định
    db.drop_indexes()
    db.set_config([])

    # Bước 5: thời gian từng truy vấn và các truy vấn chậm nhất
    times = {}
    for name, plan in plans.items():
        if plan is not None:
            times[name] = plan["Execution Time"] / 1000  # ms → s
    slow = slowest_queries(times, top)

    slow_report = []
    for name in slow:
        slow_report.append(f"{name} {times[name]:.2f} s")
    print("    chậm nhất: " + ", ".join(slow_report))

    # Bước 6: ghép prompt phản hồi
    best_lines = []
    for cmd in settings:
        best_lines.append(cmd)
    for ix in indexes:
        best_lines.append(ix.sql)

    if len(problems) > 0:
        problem_lines = ["Commands of this configuration that were rejected or adjusted before or while applying it:"]
        for problem in problems:
            problem_lines.append("- " + problem)
        problems_text = "\n".join(problem_lines) + "\n"
    else:
        problems_text = ""

    slow_lines = []
    for name in slow:
        summary = summarize_plan(plans[name]["Plan"])
        slow_lines.append(f"- {name} ({times[name]:.2f} s): {summary}")

    text = REFINE.format(prompt=prompt, best_time=best_time, best="\n".join(best_lines),
                         problems=problems_text, slow="\n".join(slow_lines))
    return text, times


# ============================================================================= D3: kiểm định cấu hình

# Tham số làm mất an toàn dữ liệu hoặc đổi hạ tầng: không cho LLM đổi
DENY = {
    "fsync", "full_page_writes", "synchronous_commit", "zero_damaged_pages", "ignore_checksum_failure",
    "data_directory", "config_file", "hba_file", "ident_file", "port", "listen_addresses", "unix_socket_directories",
    "shared_preload_libraries", "max_connections", "archive_mode", "archive_command", "wal_level",
    "allow_system_table_mods", "ssl", "password_encryption", "log_directory", "log_destination",
}

# Tham số số tiến trình: không vượt số nhân
WORKERS = {"max_worker_processes", "max_parallel_workers", "max_parallel_workers_per_gather",
           "max_parallel_maintenance_workers"}

# Đơn vị bộ nhớ PostgreSQL chấp nhận khi ghi kèm giá trị (phân biệt hoa/thường) → kB
KB = {"B": 1 / 1024, "kB": 1, "MB": 1024, "GB": 2**20, "TB": 2**30}

# Đơn vị mặc định của tham số (cột unit của pg_settings) → kB, dùng khi giá trị không ghi đơn vị
DEFAULT_UNIT_KB = {"8kB": 8, "kB": 1, "MB": 1024, "B": 1 / 1024}


def declared_specs(prompt):
    """(RAM GB, số nhân) khai báo trong prompt (template của §3.1); D3 kẹp bộ nhớ theo đó. None nếu không thấy."""
    if prompt is None:
        return None
    match = re.search(r"memory: (\d+)GB\s+cores: (\d+)", prompt)
    if match is None:
        return None
    memory_gb = int(match.group(1))
    cores = int(match.group(2))
    return memory_gb, cores


def to_kb(value, unit):
    """'8GB' → 8388608; '2048' với unit '8kB' → 16384. None nếu PostgreSQL không đọc được giá trị này."""
    match = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*([A-Za-z]*)", value.strip())
    if match is None:
        return None
    number = float(match.group(1))
    suffix = match.group(2)

    if suffix != "":
        if suffix in KB:
            return number * KB[suffix]
        return None  # vd. "8gb": PostgreSQL không nhận

    # Không ghi đơn vị: dùng đơn vị mặc định của tham số
    if unit in DEFAULT_UNIT_KB:
        return number * DEFAULT_UNIT_KB[unit]
    return number


def memory_caps_kb(memory_gb, cores):
    """Ngưỡng bộ nhớ (kB) của từng tham số, theo RAM khai báo trong prompt."""
    ram_kb = memory_gb * 2**20
    caps = {}
    caps["shared_buffers"] = 0.40 * ram_kb
    caps["effective_cache_size"] = 0.90 * ram_kb
    caps["maintenance_work_mem"] = 0.25 * ram_kb
    caps["work_mem"] = 0.25 * ram_kb / cores  # nhiều worker × nhiều node sort/hash cùng dùng work_mem
    caps["temp_buffers"] = 0.10 * ram_kb
    caps["wal_buffers"] = 2**20  # 1 GB
    return caps


def probe(db, name, value):
    """Thử giá trị bằng SET LOCAL trong một giao dịch rồi ROLLBACK; trả về thông báo lỗi nếu DBMS từ chối, None nếu
    hợp lệ. Bắt được cả lỗi phụ thuộc nền tảng, vd. effective_io_concurrency phải bằng 0 trên macOS.
    Chỉ dùng cho tham số đặt được theo phiên (context user / superuser)."""
    with db.conn.cursor() as cur:
        cur.execute("BEGIN")
        try:
            cur.execute(f"SET LOCAL {name} = %s", (value,))
        except psycopg2.Error as e:
            return str(e).splitlines()[0]
        finally:
            cur.execute("ROLLBACK")
    return None


def validate_settings(db, settings, memory_gb, cores):
    """D3 cho các lệnh ALTER SYSTEM SET. Trả về (các lệnh được giữ, [lý do loại / kẹp])."""
    rejected = []

    # Bước 1: thông tin của mọi tham số trong pg_settings
    parameters = {}
    for name, context, unit in db.sql("SELECT name, context, unit FROM pg_settings", fetch=True):
        parameters[name] = (context, unit)
    caps = memory_caps_kb(memory_gb, cores)

    kept = {}  # tên tham số → giá trị; lệnh SET trùng tên thì lệnh sau thắng, giống ALTER SYSTEM
    for cmd in settings:
        # Bước 2: tách tên và giá trị
        match = re.match(r"ALTER\s+SYSTEM\s+SET\s+([\w.]+)\s*(?:=|\bTO\b)\s*(.+)$", cmd, re.I | re.S)
        if match is None:
            rejected.append(f"unparsed command: {cmd}")
            continue
        name = match.group(1).lower()
        value = match.group(2).strip().strip("'\"")

        # Bước 3: tham số phải có thật và được phép đổi
        if name not in parameters:
            rejected.append(f"unknown parameter: {name}")
            continue
        context, unit = parameters[name]
        if name in DENY or context == "internal":
            rejected.append(f"parameter not allowed: {name} = {value}")
            continue

        # Bước 4: kẹp tham số bộ nhớ theo RAM khai báo
        if name in caps:
            kb = to_kb(value, unit)
            if kb is None:
                rejected.append(f"invalid value: {name} = {value}")
                continue
            if kb > caps[name]:
                new_value = f"{int(caps[name] // 1024)}MB"
                rejected.append(f"clamped {name}: {value} -> {new_value} ({memory_gb}GB RAM, {cores} cores)")
                value = new_value

        # Bước 5: kẹp số tiến trình theo số nhân
        if name in WORKERS and value.isdigit():
            if int(value) > cores:
                rejected.append(f"clamped {name}: {value} -> {cores} ({cores} cores)")
                value = str(cores)

        # Bước 6: thử giá trị trên DBMS (chỉ với tham số đặt được theo phiên)
        if context == "user" or context == "superuser":
            error = probe(db, name, value)
            if error is not None:
                rejected.append(f"invalid value: {name} = {value} ({error})")
                continue

        # Bước 7: giữ lại; báo nếu ghi đè một lệnh trước cùng tên
        if name in kept:
            rejected.append(f"duplicate parameter: {name} = {kept[name]} is overridden by {name} = {value}")
        kept[name] = value

    commands = []
    for name, value in kept.items():
        commands.append(f"ALTER SYSTEM SET {name} = '{value}'")
    return commands, rejected


def index_method(ix):
    """Kiểu chỉ mục (btree, brin, hash, ...) ghi trong lệnh CREATE INDEX; mặc định btree."""
    match = re.search(r"\bUSING\s+(\w+)", ix.sql, re.I)
    if match is None:
        return "btree"
    return match.group(1).lower()


def index_columns(ix):
    """Các cột của chỉ mục, viết thường, bỏ dấu nháy kép."""
    columns = []
    for column in ix.columns:
        columns.append(column.strip('"').lower())
    return tuple(columns)


def is_redundant(ix, kept):
    """True nếu ix là chỉ mục B-tree trùng hoặc là tiền tố của một chỉ mục B-tree đã giữ trên cùng bảng
    (B-tree trên (a, b) dùng được cho điều kiện trên a, nên chỉ mục (a) là thừa)."""
    if index_method(ix) != "btree":
        return False
    table = ix.table.lower()
    columns = index_columns(ix)
    for other in kept:
        if other.table.lower() != table:
            continue
        if index_method(other) != "btree":
            continue
        other_prefix = index_columns(other)[:len(columns)]
        if other_prefix == columns:
            return True
    return False


def validate_indexes(db, indexes):
    """D3 cho các lệnh CREATE INDEX. Trả về (các chỉ mục được giữ, theo thứ tự LLM viết, [lý do loại])."""
    tables = db.table_columns()
    rejected = []

    # Bước 1: xét chỉ mục nhiều cột trước, để nhận ra chỉ mục ít cột là tiền tố của nó
    def column_count(ix):
        return len(ix.columns)

    ordered = sorted(indexes, key=column_count, reverse=True)

    kept = []
    for ix in ordered:
        table = ix.table.lower()

        # Bước 2: bảng và mọi cột phải có thật
        exists = table in tables
        if exists:
            for column in index_columns(ix):
                if column not in tables[table]:
                    exists = False
                    break
        if not exists:
            rejected.append(f"index on a missing table or column: {ix.sql}")
            continue

        # Bước 3: bỏ chỉ mục trùng hoặc là tiền tố của chỉ mục đã giữ
        if is_redundant(ix, kept):
            rejected.append(f"redundant index (prefix of another index): {ix.sql}")
            continue

        kept.append(ix)

    # Bước 4: trả lại theo thứ tự ban đầu của LLM
    def original_position(ix):
        return indexes.index(ix)

    kept = sorted(kept, key=original_position)
    return kept, rejected


def validate_config(db, config, memory_gb, cores):
    """D3: config = (settings, indexes) của parse_config → ((settings, indexes) đã lọc, [lý do loại / kẹp]).

    Kẹp bộ nhớ theo RAM khai báo trong prompt: shared_buffers ≤ 40%, effective_cache_size ≤ 90%,
    maintenance_work_mem ≤ 25%, work_mem ≤ 25%·RAM / số nhân, temp_buffers ≤ 10%, wal_buffers ≤ 1 GB;
    số worker ≤ số nhân.
    """
    settings, indexes = config
    kept_settings, rejected_settings = validate_settings(db, settings, memory_gb, cores)
    kept_indexes, rejected_indexes = validate_indexes(db, indexes)
    rejected = rejected_settings + rejected_indexes
    return (kept_settings, kept_indexes), rejected
