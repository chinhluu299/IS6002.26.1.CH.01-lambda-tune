import json
import math
import os
import re
import subprocess
import time
from collections import namedtuple
from itertools import combinations
from pathlib import Path

import numpy as np
import psycopg2
from scipy.optimize import Bounds, LinearConstraint, milp

HERE = Path(__file__).resolve().parent


def load_env_file(path):
    """Đọc tệp .env (mỗi dòng KEY=VALUE) vào biến môi trường. Biến đã export trong shell được ưu tiên;
    dòng chú thích (#) và dòng không có giá trị bị bỏ qua."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator == "":
            continue
        key = key.strip()
        value = value.strip()
        if key.startswith("#"):
            continue
        if value == "":
            continue
        value = value.strip("'\"")  # bỏ dấu nháy bao quanh giá trị
        os.environ.setdefault(key, value)


load_env_file(HERE / ".env")
DSN = os.environ.get("LT_DSN", "host=localhost port=5433 user=postgres")

# Mặc định PostgreSQL chạy trong container của docker-compose.yml; đặt PGDATA để dùng cụm cục bộ (pg_ctl trên máy).
PGDATA = os.environ.get("PGDATA")
CONTAINER = "lambdatune-pg"
if PGDATA:
    EXEC = []                 # chạy pg_ctl trực tiếp trên máy
    DATA = PGDATA
else:
    EXEC = ["docker", "exec", "-u", "postgres", CONTAINER]  # chạy pg_ctl trong container
    DATA = "/var/lib/postgresql/data"


def machine():
    """RAM (GB) và số nhân của máy chạy PostgreSQL (với Docker Desktop: của máy ảo Docker), dùng cho prompt."""
    if PGDATA:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        memory_gb = round(page_size * page_count / 2**30)
        return memory_gb, os.cpu_count()

    command = ["docker", "info", "--format", "{{.MemTotal}} {{.NCPU}}"]
    output = subprocess.run(command, capture_output=True, text=True, check=True).stdout
    memory_bytes, cores = output.split()
    memory_gb = round(int(memory_bytes) / 2**30)
    return memory_gb, int(cores)


# ============================================================================= PostgreSQL
class PG:
    """Các thao tác λ-Tune cần: chạy truy vấn có timeout, đổi tham số hệ thống, tạo/xóa chỉ mục."""

    def __init__(self, dbname="tpch", dsn=DSN):
        self.dsn = f"{dsn} dbname={dbname}"
        self.connect()

    def connect(self):
        self.conn = psycopg2.connect(self.dsn)
        self.conn.autocommit = True

    def sql(self, cmd, fetch=False):
        """Chạy một lệnh SQL; fetch=True thì trả về mọi dòng kết quả."""
        with self.conn.cursor() as cur:
            cur.execute(cmd)
            if fetch:
                return cur.fetchall()
            return None

    def execute(self, query, timeout=math.inf):
        """Hàm Execute của Alg. 3: chạy truy vấn, tối đa `timeout` giây. Trả về (số giây, xong trước timeout?)."""
        if timeout < 0.1:
            return 0.0, False  # coi như hết giờ; timeout quá nhỏ còn làm chính lệnh RESET bên dưới bị ngắt

        if timeout == math.inf:
            timeout_ms = 0  # statement_timeout = 0 nghĩa là không giới hạn
        else:
            timeout_ms = int(timeout * 1000)
        self.sql(f"SET statement_timeout = {timeout_ms}")

        t0 = time.time()
        try:
            self.sql(query, fetch=True)
            return time.time() - t0, True
        except psycopg2.errors.QueryCanceled:
            return time.time() - t0, False
        finally:
            self.sql("RESET statement_timeout")

    def run_workload(self, queries, timeout=math.inf):
        """Chạy lần lượt mọi truy vấn, tổng thời gian tối đa `timeout`. Trả về (số giây, xong hết?)."""
        total = 0.0
        for sql in queries.values():
            sec, ok = self.execute(sql, timeout - total)
            total = total + sec
            if not ok:
                return total, False
        return total, True

    def explain(self, query):
        """Kế hoạch mặc định của optimizer (EXPLAIN, FORMAT JSON), không chạy truy vấn."""
        rows = self.sql("EXPLAIN (FORMAT JSON) " + query, fetch=True)
        return rows[0][0][0]["Plan"]

    def table_rows(self):
        """{bảng: số dòng ước lượng (reltuples)} của schema public."""
        rows = self.sql("SELECT relname, reltuples FROM pg_class "
                        "WHERE relkind = 'r' AND relnamespace = 'public'::regnamespace", fetch=True)
        counts = {}
        for table, count in rows:
            counts[table] = count
        return counts

    def table_columns(self):
        """{bảng: {các cột}} của schema public."""
        rows = self.sql("SELECT table_name, column_name FROM information_schema.columns "
                        "WHERE table_schema = 'public'", fetch=True)
        tables = {}
        for table, column in rows:
            if table not in tables:
                tables[table] = set()
            tables[table].add(column)
        return tables

    def set_config(self, settings):
        """Về cấu hình mặc định, chạy các lệnh ALTER SYSTEM SET rồi khởi động lại (vd. shared_buffers cần restart).
        Trả về các lệnh bị DBMS từ chối (D2 gửi lại chúng cho LLM)."""
        # Bước 1: xóa mọi tham số đã đặt trước đó, rồi đặt các tham số mới
        self.sql("ALTER SYSTEM RESET ALL")
        errors = []
        for cmd in settings:
            try:
                self.sql(cmd)
            except psycopg2.Error as e:
                first_line = str(e).splitlines()[0]
                errors.append(f"{cmd}: {first_line}")
                print(f"    [bỏ qua] {errors[-1]}")

        # Bước 2: khởi động lại PostgreSQL (không dùng "pg_ctl restart": nó lặp lại -D cũ)
        self.conn.close()
        pg_ctl = EXEC + ["pg_ctl", "-D", DATA, "-l", f"{DATA}/server.log", "-w"]
        subprocess.run(pg_ctl + ["stop", "-m", "fast"], capture_output=True)
        started = subprocess.run(pg_ctl + ["start"], capture_output=True)

        # Bước 3: cấu hình làm PostgreSQL không khởi động được thì xóa postgresql.auto.conf, quay về mặc định
        if started.returncode != 0:
            subprocess.run(EXEC + ["cp", "/dev/null", f"{DATA}/postgresql.auto.conf"])
            subprocess.run(pg_ctl + ["start"], capture_output=True, check=True)
            print("    [lỗi] PostgreSQL không khởi động được với cấu hình này, quay về mặc định")
            errors.append("PostgreSQL failed to restart with these settings; all of them were reverted to defaults")

        self.connect()
        return errors

    def drop_indexes(self):
        """Xóa mọi chỉ mục do tuning tạo ra (giữ chỉ mục của ràng buộc khóa, nếu có)."""
        rows = self.sql("SELECT indexname FROM pg_indexes WHERE schemaname = 'public' "
                        "AND indexname NOT IN (SELECT conname FROM pg_constraint)", fetch=True)
        for row in rows:
            name = row[0]
            self.sql(f'DROP INDEX IF EXISTS "{name}"')


# ============================================================================= §3 Sinh prompt
TEMPLATE = """Recommend some configuration parameters for {dbms} to optimize the system's performance. Parameters might include system-level configurations, like memory, query optimizer or physical design configurations, like index recommendations.

{workload}

The workload runs on a system with the following specs:
memory: {memory}
cores: {cores}

Your response should strictly consist of a JSON object {{"commands": [list of SQL commands]}}, without any additional text."""
# Listing 1 của bài báo; câu cuối (định dạng đầu ra) lấy từ repo gốc để tách được lệnh SQL.

COND = re.compile(r"(?:(\w+)\.)?(\w+) = (\w+)\.(\w+)")  # [bảng.]cột = bảng.cột; so sánh với hằng/tham số không khớp


def walk(node):
    """Danh sách mọi node của một kế hoạch EXPLAIN (FORMAT JSON): node cha trước, rồi lần lượt các cây con."""
    nodes = [node]
    for child in node.get("Plans", []):
        nodes.extend(walk(child))
    return nodes


def table_aliases(nodes):
    """{alias: bảng} của các node quét bảng trong một kế hoạch; node không có alias thì alias là tên bảng."""
    alias = {}
    for node in nodes:
        if "Relation Name" in node:
            table = node["Relation Name"]
            name = node.get("Alias", table)
            alias[name] = table
    return alias


def join_values(plans):
    """§3.2: V(p) = tổng "Total Cost" của các node có điều kiện join p, trên kế hoạch mặc định của optimizer.

    Trả về {("bảng1.cột1", "bảng2.cột2"): V}, mỗi cặp viết theo thứ tự tên tăng dần.
    """
    V = {}
    for plan in plans:
        nodes = walk(plan)
        alias = table_aliases(nodes)
        for node in nodes:
            for key in ("Hash Cond", "Merge Cond", "Join Filter", "Index Cond"):
                condition = node.get(key, "")
                for t1, c1, t2, c2 in COND.findall(condition):
                    # Vế trái: Index Cond không ghi bảng, vì đó là cột của chính bảng đang quét
                    if t1 != "":
                        table1 = alias.get(t1)
                    else:
                        table1 = node.get("Relation Name")
                    # Vế phải: None nếu là CTE / truy vấn con (bỏ qua)
                    table2 = alias.get(t2)
                    if table1 is None or table2 is None:
                        continue
                    if table1 == table2 and c1 == c2:
                        continue

                    pair = tuple(sorted([f"{table1}.{c1}", f"{table2}.{c2}"]))
                    if pair not in V:
                        V[pair] = 0.0
                    V[pair] = V[pair] + node["Total Cost"]
    return V


def tokens(text):
    """Ước lượng số token: khoảng 4 ký tự / token, ít nhất 1."""
    estimate = round(len(text) / 4)
    if estimate < 1:
        return 1
    return estimate


def compress(V, budget=math.inf, F=None):
    """§3.3, Bảng 1: chọn những cặp join đưa vào prompt, viết thành các dòng "a: b1, b2, ...".

    Biến 0/1:  R[a→b] = 1 nếu b đứng bên phải a (trên dòng của a);  L[a] = 1 nếu có dòng bắt đầu bằng a.
    max  Σ V(p)·R_p   sao cho   R[a→b] ≤ L[a]                     b chỉ đứng sau a khi có dòng của a
                                L[a] ≤ Σ_b R[a→b]                 dòng của a có ít nhất một cột bên phải
                                R[a→b] + R[b→a] ≤ 1               mỗi cặp join chỉ viết một lần
                                Σ H_b·R[a→b] + Σ H_a·L[a] ≤ B     tổng token không quá ngân sách B

    D1 (extensions.py): F = {cột f: (V(f), dòng snippet lọc "f: <, BETWEEN")}. Mỗi snippet lọc có thêm biến S[f]:
    mục tiêu cộng Σ V(f)·S[f], vế trái ràng buộc ngân sách cộng Σ H_f·S[f]. Snippet join và snippet lọc vì vậy cạnh
    tranh cùng một ngân sách B, theo cùng thước đo là chi phí EXPLAIN.
    Trả về ({a: [b1, b2, ...]}, [các cột f được chọn]).
    """
    if F is None:
        F = {}  # không có snippet lọc: đúng ILP của bài báo

    # Bước 1: mỗi cặp {a, b} có hai cách viết (arrow): "a: b" hoặc "b: a"
    pairs = list(V)
    arrows = []
    for a, b in pairs:
        arrows.append((a, b))
    for a, b in pairs:
        arrows.append((b, a))
    column_set = set()
    for a, b in pairs:
        column_set.add(a)
        column_set.add(b)
    cols = sorted(column_set)

    # Bước 2: đánh số các biến trong vector x = [R của từng arrow..., L của từng cột..., S của từng snippet lọc...]
    next_index = 0
    R = {}
    for arrow in arrows:
        R[arrow] = next_index
        next_index = next_index + 1
    L = {}
    for c in cols:
        L[c] = next_index
        next_index = next_index + 1
    S = {}
    for f in F:
        S[f] = next_index
        next_index = next_index + 1
    n = next_index

    # Bước 3: giá trị và số token của từng biến
    H = {}
    for c in cols:
        H[c] = tokens(c) + 1  # số token để viết cột c (+1 cho dấu ":" hoặc ",")
    value = np.zeros(n)
    token_cost = np.zeros(n)
    for (a, b), i in R.items():
        if (a, b) in V:
            value[i] = V[(a, b)]
        else:
            value[i] = V[(b, a)]
        token_cost[i] = H[b]
    for c, i in L.items():
        token_cost[i] = H[c]
    for f, i in S.items():
        filter_value, filter_line = F[f]
        value[i] = filter_value
        token_cost[i] = tokens(filter_line) + 1  # +1 cho dấu xuống dòng

    # Bước 4: các ràng buộc của Bảng 1, mỗi ràng buộc là một hàng "Σ hệ số · x[i] ≤ cận trên"
    rows = [token_cost]  # hàng đầu tiên: ràng buộc ngân sách token
    upper = [budget]

    def at_most(coeffs, bound):
        row = np.zeros(n)
        for i, coef in coeffs.items():
            row[i] = coef
        rows.append(row)
        upper.append(bound)

    for (a, b), i in R.items():
        at_most({i: 1, L[a]: -1}, 0)                  # R[a→b] − L[a] ≤ 0
    for c in cols:
        coeffs = {L[c]: 1}                            # L[c] − Σ_b R[c→b] ≤ 0
        for a, b in arrows:
            if a == c:
                coeffs[R[(a, b)]] = -1
        at_most(coeffs, 0)
    for a, b in pairs:
        at_most({R[(a, b)]: 1, R[(b, a)]: 1}, 1)     # R[a→b] + R[b→a] ≤ 1
    table1 = LinearConstraint(np.array(rows), -np.inf, upper)

    def solve(objective, extra_constraints):
        """milp tìm cực tiểu của objective · x, mọi biến nguyên trong [0, 1]."""
        constraints = [table1] + extra_constraints
        result = milp(objective, integrality=np.ones(n), bounds=Bounds(0, 1), constraints=constraints)
        return result.x.round()

    # Bước 5: cực đại Σ V(p)·R_p (+ Σ V(f)·S[f] với D1)
    x = solve(-value, [])

    # Bước 6: bài báo không nói chọn lời giải nào khi có nhiều lời giải tối ưu. Giữ nguyên tập cặp join (và snippet
    # lọc) đã chọn, nên giữ nguyên giá trị, rồi chọn cách viết ít token nhất, vd. "A: B, C, D" thay vì "B: A", "C: A".
    chosen = np.zeros((len(pairs) + len(S), n))  # hàng p: R[a→b] + R[b→a] của cặp p; hàng f: S[f]
    row = 0
    for a, b in pairs:
        chosen[row, R[(a, b)]] = 1
        chosen[row, R[(b, a)]] = 1
        row = row + 1
    for f, i in S.items():
        chosen[row, i] = 1
        row = row + 1
    keep_chosen = LinearConstraint(chosen, chosen @ x, np.inf)
    x = solve(token_cost, [keep_chosen])

    # Bước 7: đọc lời giải
    lines = {}
    for (a, b), i in R.items():
        if x[i] == 1:
            if a not in lines:
                lines[a] = []
            lines[a].append(b)
    kept_filters = []  # D1: các snippet lọc được chọn
    for f, i in S.items():
        if x[i] == 1:
            kept_filters.append(f)
    return lines, kept_filters


def join_block(lines):
    """Phần workload của Listing 1: mỗi dòng "A: B, C" là một khóa join và các cột join với nó."""
    text_lines = []
    for a in sorted(lines):
        right_side = sorted(lines[a])
        text_lines.append(f"{a}: {', '.join(right_side)}")
    header = ("Each row in the following list has the following format:\n"
              "{a join key A}:{all the joins with A in the workload}\n")
    return header + "\n".join(text_lines)


def generate_prompt(db, queries, memory_gb, cores, budget=math.inf, dbms="PostgreSQL"):
    """§3.1: EXPLAIN mọi truy vấn → V(p) → nén workload → điền vào template."""
    plans = []
    for sql in queries.values():
        plans.append(db.explain(sql))
    lines, kept_filters = compress(join_values(plans), budget)
    return TEMPLATE.format(dbms=dbms, workload=join_block(lines), memory=f"{memory_gb}GB", cores=cores)


# ============================================================================= Alg. 1: gọi LLM
def ask_llm(prompt, k=5, model="gpt-4", temperature=None):
    """Gọi LLM k lần với cùng một prompt; các cấu hình khác nhau nhờ lấy mẫu ngẫu nhiên của LLM (bài báo: GPT-4).
    Dùng API chuẩn OpenAI (/chat/completions); key và endpoint lấy từ OPENAI_API_KEY, OPENAI_BASE_URL."""
    from openai import OpenAI
    client = OpenAI()
    messages = [
        {"role": "system", "content": "You are a helpful Database Administrator."},
        {"role": "user", "content": prompt},
    ]
    responses = []
    for i in range(k):
        if temperature is None:
            r = client.chat.completions.create(model=model, messages=messages)
        else:
            r = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
        text = r.choices[0].message.content
        if text is None:
            text = ""
        print(f"[{i + 1}/{k}] {r.model}: {len(text)} ký tự, finish_reason={r.choices[0].finish_reason}")
        responses.append(text)
    return responses


Index = namedtuple("Index", "sql table columns")  # vd. ("CREATE INDEX ON lineitem (l_orderkey)", "lineitem", ("l_orderkey",))


def commands_from_json(text):
    """Danh sách lệnh trong JSON {"commands": [...]} của phản hồi; None nếu phản hồi không chứa JSON đúng dạng."""
    match = re.search(r"\{.*\}", text, re.S)
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
        return data["commands"]
    except (ValueError, KeyError, TypeError):
        return None


def parse_config(text):
    """Phản hồi LLM → cấu hình (settings, indexes): danh sách lệnh ALTER SYSTEM SET và danh sách Index."""
    # Bước 1: bỏ phần suy luận <think>...</think> mà một số mô hình (vd. Qwen3) trả kèm
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)

    # Bước 2: lấy các lệnh từ JSON; không phải JSON thì tìm lệnh SQL ngay trong văn bản
    commands = commands_from_json(text)
    if commands is None:
        commands = re.findall(r"(?:ALTER SYSTEM SET|CREATE INDEX)[^;]+", text, re.I)

    # Bước 3: phân loại từng lệnh
    settings = []
    indexes = []
    for c in commands:
        c = c.strip().rstrip(";")
        ix = re.match(r"CREATE\s+(?:UNIQUE\s+)?INDEX\b.*?\bON\s+(\w+)\s*(?:USING\s+\w+\s*)?\(([^)]*)\)", c, re.I | re.S)
        if ix is not None:
            table = ix.group(1)
            columns = []
            for part in ix.group(2).split(","):
                columns.append(part.split()[0])  # "l_orderkey DESC" → "l_orderkey"
            indexes.append(Index(c, table, tuple(columns)))
        elif re.match(r"(ALTER\s+SYSTEM\s+)?SET\s", c, re.I):
            # "SET x = y" cũng được đổi thành ALTER SYSTEM SET
            settings.append(re.sub(r"^(ALTER\s+SYSTEM\s+)?SET", "ALTER SYSTEM SET", c, flags=re.I))
    return settings, indexes


# ============================================================================= Alg. 4 + §5.4: thứ tự truy vấn
def compute_order_dp(needs, cost):
    """Algorithm 4: quy hoạch động trên tập con (giống thuật toán Selinger cho thứ tự join).

    needs[i] = tập chỉ mục mà truy vấn i cần, cost(ix) = chi phí tạo chỉ mục ix. Trả về thứ tự các i.
    Công thức (1): ngắt sau mỗi vị trí có xác suất 1/n, nên truy vấn ở vị trí j (đếm từ 1) chỉ được chạy với xác suất
    (n − j + 1)/n và chi phí kỳ vọng là Σ_j z_j·(n − j + 1)/n, với z_j = chi phí các chỉ mục nó cần mà chưa được tạo.
    Ví dụ 5.1 (chi phí 1 và 5): thứ tự q1, q2 tốn 1 + 5/2 = 3,5 < 5 + 1/2 = 5,5.
    """
    n = len(needs)
    empty = frozenset()
    dp_cost = {empty: 0.0}         # dp_cost[S]: chi phí kỳ vọng nhỏ nhất để chạy tập truy vấn S trước
    dp_order = {empty: []}         # dp_order[S]: thứ tự đạt chi phí đó
    created = {empty: frozenset()}  # created[S]: các chỉ mục đã có sau khi chạy S

    def query_cost(i, before):
        """computeCost: truy vấn i chạy ngay sau tập `before`, tức ở vị trí j = |before| + 1."""
        missing_costs = []  # chi phí các chỉ mục truy vấn i cần mà chưa được tạo
        for ix in needs[i] - created[before]:
            missing_costs.append(cost(ix))
        z = sum(missing_costs)
        return z * (n - len(before)) / n

    for size in range(1, n + 1):  # tập con theo kích thước tăng dần
        for combo in combinations(range(n), size):
            subset = frozenset(combo)
            any_query = min(subset)
            created[subset] = created[subset - {any_query}] | needs[any_query]
            dp_cost[subset] = math.inf
            for last in sorted(subset):  # thử từng truy vấn làm truy vấn cuối của S
                rest = subset - {last}
                c = dp_cost[rest] + query_cost(last, rest)
                if c < dp_cost[subset]:
                    dp_cost[subset] = c
                    dp_order[subset] = dp_order[rest] + [last]
    return dp_order[frozenset(range(n))]


def cluster_queries(Q, I, k):
    """§5.4: gom các truy vấn thành k nhóm bằng K-Means trên vector 0/1 "truy vấn dùng chỉ mục nào"."""
    from sklearn.cluster import KMeans

    used = set()
    for q in Q:
        used = used | I[q]
    all_indexes = sorted(used)

    X = []
    for q in Q:
        row = []
        for ix in all_indexes:
            row.append(ix in I[q])
        X.append(row)
    labels = KMeans(k, n_init=10, random_state=0).fit_predict(np.array(X, dtype=float))

    units = []
    for cluster in range(k):
        unit = []
        for q, label in zip(Q, labels):
            if label == cluster:
                unit.append(q)
        if len(unit) > 0:
            units.append(unit)
    return units


def find_optimal_order(Q, I, cost, max_units=13):
    """§5.4: gom các truy vấn có cùng tập chỉ mục thành một nhóm; nếu còn > 13 nhóm thì gom bằng K-Means.
    DP (Alg. 4) sắp thứ tự các nhóm; truy vấn trong một nhóm chạy liền nhau."""
    # Bước 1: các truy vấn có cùng tập chỉ mục vào chung một nhóm
    groups = {}
    for q in Q:
        key = frozenset(I[q])
        if key not in groups:
            groups[key] = []
        groups[key].append(q)
    units = list(groups.values())

    # Bước 2: còn quá nhiều nhóm thì gom lại bằng K-Means (DP có độ phức tạp 2^n)
    if len(units) > max_units:
        units = cluster_queries(Q, I, max_units)

    # Bước 3: tập chỉ mục mà mỗi nhóm cần
    needs = []
    for unit in units:
        index_sets = []
        for q in unit:
            index_sets.append(I[q])
        needs.append(frozenset().union(*index_sets))

    # Bước 4: DP sắp thứ tự các nhóm, rồi trải các nhóm ra thành danh sách truy vấn
    order = []
    for i in compute_order_dp(needs, cost):
        for q in units[i]:
            order.append(q)
    return order


# ============================================================================= Alg. 3: đánh giá một cấu hình
class ConfigMeta:
    """Bảng 2: thông tin của một cấu hình, giữ qua các vòng của Alg. 2."""

    def __init__(self):
        self.time = 0.0          # tổng thời gian các truy vấn đã chạy xong
        self.complete = False    # đã chạy xong cả workload chưa
        self.index_time = 0.0    # thời gian tạo chỉ mục trong lần đánh giá gần nhất
        self.completed = set()   # các truy vấn đã chạy xong


def mentions(sql, name):
    """True nếu tên `name` (bảng hoặc cột) xuất hiện như một từ riêng trong câu SQL."""
    return re.search(rf"\b{name}\b", sql) is not None


def query_index_map(queries, Q, indexes):
    """Alg. 3 dòng 6: chỉ mục có thể có ích cho q nếu bảng và ít nhất một cột của nó xuất hiện trong q."""
    I = {}
    for q in Q:
        sql = queries[q]
        useful = set()
        for ix in indexes:
            if not mentions(sql, ix.table):
                continue
            for col in ix.columns:
                if mentions(sql, col):
                    useful.add(ix)
                    break
        I[q] = useful
    return I


def evaluate(db, config, queries, t, meta, rows):
    """Algorithm 3: chạy các truy vấn chưa xong dưới cấu hình `config` trong tối đa t giây, cập nhật `meta`.

    rows = số dòng của từng bảng, dùng làm chi phí tạo chỉ mục khi sắp thứ tự truy vấn.
    """
    settings, indexes = config
    db.set_config(settings)

    # Alg. 2 dòng 20: chỉ chạy các truy vấn chưa xong ở các vòng trước
    Q = []
    for q in queries:
        if q not in meta.completed:
            Q.append(q)

    # Sắp thứ tự truy vấn để giảm chi phí tạo chỉ mục kỳ vọng (Alg. 4)
    I = query_index_map(queries, Q, indexes)

    def index_cost(ix):
        return rows.get(ix.table, 0)

    Q = find_optimal_order(Q, I, index_cost)

    remaining = t
    created = set()
    meta.complete = True
    meta.index_time = 0.0
    for q in Q:
        # Tạo lười: chỉ tạo những chỉ mục mà truy vấn sắp chạy có thể dùng và chưa có
        t0 = time.time()
        for ix in I[q] - created:
            try:
                db.sql(ix.sql)
            except psycopg2.Error as e:
                print(f"    [bỏ qua] {ix.sql}: {str(e).splitlines()[0]}")
        created = created | I[q]
        meta.index_time = meta.index_time + (time.time() - t0)

        # Chạy truy vấn với thời gian còn lại; hết giờ thì dừng lượt đánh giá này
        sec, ok = db.execute(queries[q], remaining)
        if not ok:
            meta.complete = False
            break
        remaining = remaining - sec
        meta.time = meta.time + sec
        meta.completed.add(q)
    db.drop_indexes()  # chỉ mục chỉ sống trong một lần đánh giá


# ============================================================================= Alg. 2: chọn cấu hình
def config_select(db, queries, configs, t=10.0, alpha=10.0, incumbent=None):
    """Algorithm 2. configs = {tên: (settings, indexes)}, t = timeout vòng đầu, α = hệ số tăng timeout.

    Trả về (tên cấu hình tốt nhất, thời gian workload của nó, trace). Thời gian tuning (không tính đổi cấu hình)
    là O(k·α·C_best) với α ≥ 2 (Định lý 4.3): cấu hình tệ không thể chiếm hết thời gian tuning.

    D2 (extensions.py): incumbent = (C_b, tên của b) là cấu hình tốt nhất đã biết. Khi đó vòng lặp dòng 6-13 bị bỏ
    qua và mỗi cấu hình mới đi thẳng tới dòng 14-15 với timeout C_b: chi phí thêm ≤ k₂·C_b.
    """
    rows = db.table_rows()
    meta = {}
    for c in configs:
        meta[c] = ConfigMeta()
    if incumbent is None:
        best = {"time": math.inf, "config": None}
    else:  # D2: bắt đầu với cấu hình tốt nhất đã biết
        incumbent_time, incumbent_name = incumbent
        best = {"time": incumbent_time, "config": incumbent_name}
    trace = []
    start = time.time()

    def completed_count(c):
        return len(meta[c].completed)

    def by_throughput(names):
        """Thông lượng = số truy vấn đã chạy xong; cấu hình có thông lượng cao được đánh giá trước."""
        return sorted(names, key=completed_count, reverse=True)

    def update(c, t):
        """Thủ tục Update (dòng 17-25)."""
        if best["time"] < math.inf:
            t = best["time"] - meta[c].time  # dòng 19: chạy lâu hơn mức này thì chắc chắn không tốt hơn best
        if t > 0:  # t ≤ 0 thì Evaluate ngắt ngay, nên bỏ qua để khỏi đổi cấu hình vô ích
            evaluate(db, configs[c], queries, t, meta[c], rows)
        if meta[c].complete and meta[c].time < best["time"]:
            best["time"] = meta[c].time
            best["config"] = c
        entry = {
            "elapsed": time.time() - start,
            "config": c,
            "timeout": t,
            "done": len(meta[c].completed),
            "complete": meta[c].complete,
            "index_time": meta[c].index_time,
            "best": best["time"],
        }
        trace.append(entry)
        print(f"[{entry['elapsed']:7.1f}s] {c:<12} timeout={t:8.2f}s  xong {len(meta[c].completed):2d}/{len(queries)}"
              f"  tạo chỉ mục {meta[c].index_time:5.1f}s  best={best['time']:.2f}s")

    # Chỉ dùng khi có incumbent; không thì được gán lại ngay khi vòng lặp dưới kết thúc
    candidates = list(configs)

    # Dòng 6-13: chạy theo vòng đến khi có cấu hình đầu tiên chạy xong cả workload
    while best["time"] == math.inf:
        for c in by_throughput(configs):
            update(c, t)
            if meta[c].complete:
                candidates = []
                for other in configs:
                    if other != c:
                        candidates.append(other)
                break
            # Dòng 12: timeout không nhỏ hơn thời gian tạo chỉ mục lớn nhất đã đo
            for m in meta.values():
                if m.index_time > t:
                    t = m.index_time
        t = t * alpha  # dòng 13

    # Dòng 14-15: cho các cấu hình còn lại cơ hội vượt best, mỗi cấu hình với timeout riêng
    for c in by_throughput(candidates):
        update(c, t)
    db.set_config([])
    return best["config"], best["time"], trace
