import itertools
import json
import math
import random
import re
import subprocess
import time

import numpy as np
import psycopg2

from lambdatune import DATA, DSN, EXEC, HERE, walk


def evaluate_once(db, queries, settings, indexes, start):
    """Áp dụng một cấu hình (tham số + chỉ mục) rồi chạy cả workload một lần."""
    db.set_config(settings)
    for ix in indexes:
        db.sql(ix)
    t, ok = db.run_workload(queries)
    db.drop_indexes()
    db.set_config([])

    if ok:
        best = t
    else:
        best = math.inf
    elapsed = time.time() - start
    print(f"=> workload {t:.2f} s, thời gian tuning {elapsed:.0f} s, {len(indexes)} chỉ mục")
    return {
        "best_config": {"settings": settings, "indexes": indexes},
        "best_time": best,
        "tuning_time": elapsed,
        "trace": [{"elapsed": elapsed, "best": best}],
    }


# ============================================================================= Chỉ mục giả định (hypopg)
def hypopg_ready(db):
    """Bật extension hypopg (chỉ mục giả định), cần cho Dexter và DB2 Advisor. Container của docker-compose.yml đã
    cài sẵn gói postgresql-17-hypopg; cụm PostgreSQL cục bộ (PGDATA) thì phải tự cài."""
    rows = db.sql("SELECT 1 FROM pg_available_extensions WHERE name = 'hypopg'", fetch=True)
    if len(rows) == 0:
        raise SystemExit("PostgreSQL chưa cài hypopg: dùng container của docker-compose.yml "
                         "(docker compose up -d --build) hoặc cài hypopg cho cụm cục bộ")
    db.sql("CREATE EXTENSION IF NOT EXISTS hypopg")
    db.sql("SELECT hypopg_reset()")


def index_sql(table, columns):
    """("lineitem", ("l_partkey", "l_suppkey")) → CREATE INDEX ON lineitem (l_partkey, l_suppkey)"""
    return f"CREATE INDEX ON {table} ({', '.join(columns)})"


# ============================================================================= Dexter
DEXTER_IMAGE = "ankane/dexter:v0.6.3"
FOUND = re.compile(r"Index found: (?:\w+\.)?(\w+) \(([^)]*)\)")  # dòng kết quả của Dexter: [schema.]bảng (cột, ...)


def dexter_connection():
    """host, port, user của PostgreSQL (lấy từ DSN) cho container Dexter. Từ trong container, máy chạy Docker có tên
    host.docker.internal, nên "localhost" được đổi thành tên đó."""
    parts = {"host": "localhost", "port": "5432", "user": "postgres"}
    for item in DSN.split():
        key, separator, value = item.partition("=")
        if separator != "":
            parts[key] = value
    if parts["host"] == "localhost" or parts["host"] == "127.0.0.1":
        parts["host"] = "host.docker.internal"
    return parts


def dexter(db, queries, min_savings_pct=50):
    """Công cụ Dexter thật. Với mỗi truy vấn, Dexter tạo mọi ứng viên (chỉ mục 1 cột và 2 cột trên các cột mà truy
    vấn lọc / join / sắp xếp) thành chỉ mục giả định bằng hypopg, và gợi ý chỉ mục mà kế hoạch dùng nếu chi phí EXPLAIN
    giảm ≥ min_savings_pct % (mặc định của Dexter: 50). Tham số hệ thống để mặc định."""
    start = time.time()
    hypopg_ready(db)

    # Bước 1: gọi Dexter trong Docker, mỗi truy vấn là một tham số -s
    conn = dexter_connection()
    command = ["docker", "run", "--rm", "--add-host=host.docker.internal:host-gateway", DEXTER_IMAGE,
               "-h", conn["host"], "-p", conn["port"], "-U", conn["user"], "-d", "tpch",
               "--min-cost-savings-pct", str(min_savings_pct), "--log-level", "debug"]
    for sql in queries.values():
        command.append("-s")
        command.append(sql.strip().rstrip(";"))
    output = subprocess.run(command, capture_output=True, text=True)
    print(output.stdout)
    if output.returncode != 0:
        raise SystemExit(f"Dexter lỗi (mã {output.returncode}): {output.stderr}")

    # Bước 2: đọc các dòng "Index found: bảng (cột, ...)"
    indexes = []
    for line in output.stdout.splitlines():
        match = FOUND.search(line)
        if match is None:
            continue
        table, columns = match.groups()
        indexes.append(f"CREATE INDEX ON {table} ({columns})")
    print(f"    Dexter gợi ý {len(indexes)} chỉ mục ({time.time() - start:.0f} s phân tích)")

    # Bước 3: đo workload với các chỉ mục này
    return evaluate_once(db, queries, [], indexes, start)


# ============================================================================= DB2 Advisor
class WhatIf:
    """Chỉ mục giả định (hypopg) và chi phí EXPLAIN có bộ nhớ đệm, như lớp CostEvaluation của Kossmann et al.
    Mỗi chỉ mục là một cặp (bảng, (cột, ...))."""

    def __init__(self, db, queries, query_tables):
        self.db = db
        self.queries = queries            # {truy vấn: câu SQL}
        self.query_tables = query_tables  # {truy vấn: các bảng truy vấn dùng}
        self.current = {}                 # chỉ mục giả định đang có → (oid, tên hypopg)
        self.sizes = {}                   # chỉ mục → kích thước ước lượng (byte), theo hypopg_relation_size
        self.cache = {}                   # (truy vấn, chỉ mục trên các bảng của nó) → chi phí

    def use(self, indexes):
        """Đổi tập chỉ mục giả định thành đúng `indexes`."""
        for ix in list(self.current.keys()):
            if ix not in indexes:
                oid = self.current[ix][0]
                self.db.sql(f"SELECT hypopg_drop_index({oid})")
                del self.current[ix]
        for ix in sorted(indexes):
            if ix in self.current:
                continue
            create = index_sql(ix[0], ix[1])
            rows = self.db.sql(f"SELECT indexrelid, indexname FROM hypopg_create_index('{create}')", fetch=True)
            oid, name = rows[0]
            self.current[ix] = (oid, name)
            if ix not in self.sizes:
                self.sizes[ix] = self.db.sql(f"SELECT hypopg_relation_size({oid})", fetch=True)[0][0]

    def relevant(self, q, indexes):
        """Các chỉ mục trong `indexes` nằm trên bảng mà truy vấn q dùng (chỉ chúng mới đổi được kế hoạch của q)."""
        result = set()
        for ix in indexes:
            if ix[0] in self.query_tables[q]:
                result.add(ix)
        return result

    def plan(self, q, indexes):
        """EXPLAIN truy vấn q khi có đúng các chỉ mục giả định `indexes`. Trả về (chi phí, các chỉ mục kế hoạch dùng)."""
        self.use(self.relevant(q, indexes))
        plan = self.db.explain(self.queries[q])
        names = set()
        for node in walk(plan):
            name = node.get("Index Name")
            if name is not None:
                names.add(name)
        used = set()
        for ix, (oid, name) in self.current.items():
            if name in names:
                used.add(ix)
        return plan["Total Cost"], used

    def cost(self, q, indexes):
        """Chi phí EXPLAIN của q với `indexes`, có bộ nhớ đệm."""
        key = (q, frozenset(self.relevant(q, indexes)))
        if key not in self.cache:
            self.cache[key] = self.plan(q, indexes)[0]
        return self.cache[key]

    def workload_cost(self, indexes):
        """Tổng chi phí EXPLAIN của cả workload với `indexes`."""
        total = 0.0
        for q in self.queries:
            total = total + self.cost(q, indexes)
        return total


def query_columns(sql, tables):
    """Các cột "đánh chỉ mục được" của một truy vấn: mọi cột có tên xuất hiện trong câu SQL (cách của Kossmann et al.;
    ở đây so khớp nguyên từ). Trả về {bảng: [cột, ...]}."""
    found = {}
    for table in sorted(tables):
        for column in sorted(tables[table]):
            if re.search(rf"\b{column}\b", sql) is None:
                continue
            if table not in found:
                found[table] = []
            found[table].append(column)
    return found


def try_variations(whatif, selected, ranked, budget, seconds, max_removals, rng):
    """Bước TryVariations của DB2 Advisor: đổi ngẫu nhiên k chỉ mục đã chọn lấy k chỉ mục chưa chọn (trong ngân sách),
    giữ phương án mới nếu tổng chi phí workload giảm. Chạy trong `seconds` giây."""
    not_used = []
    for ix in ranked:
        if ix not in selected:
            not_used.append(ix)
    max_removals = min(max_removals, len(selected), len(not_used))
    if seconds <= 0 or max_removals == 0:
        print("    TryVariations: không có chỉ mục nào ngoài ngân sách để đổi, bỏ qua")
        return selected

    current = set(selected)
    current_cost = whatif.workload_cost(current)
    first_cost = current_cost
    tries = 0
    stop = time.time() + seconds
    while time.time() < stop:
        # Bước 1: số chỉ mục đổi, như bản gốc: ngẫu nhiên trong 1 .. max_removals − 1
        if max_removals > 1:
            k = rng.randrange(1, max_removals)
        else:
            k = 1
        k = min(k, len(current))
        if k == 0:
            break

        # Bước 2: bỏ k chỉ mục đã chọn, thêm k chỉ mục chưa chọn nếu còn ngân sách
        variation = current - set(rng.sample(sorted(current), k))
        size = 0
        for ix in variation:
            size = size + whatif.sizes[ix]
        for ix in rng.sample(not_used, k):
            if size + whatif.sizes[ix] > budget:
                continue
            variation.add(ix)
            size = size + whatif.sizes[ix]

        # Bước 3: giữ phương án mới nếu chi phí workload giảm
        cost = whatif.workload_cost(variation)
        tries = tries + 1
        if cost < current_cost:
            current = variation
            current_cost = cost
    print(f"    TryVariations: {tries} phương án, chi phí workload {first_cost:.0f} → {current_cost:.0f}")
    return sorted(current)


def db2advis(db, queries, budget_mb=500, max_width=2, variation_seconds=10, max_removals=4, seed=0):
    """DB2 Advisor theo bản cài đặt của Kossmann et al., dùng chỉ mục giả định của hypopg:
      1. Ứng viên của mỗi truy vấn: mọi chỉ mục 1 .. max_width cột (có thứ tự) trên các cột của truy vấn, cùng bảng.
      2. Lợi ích của một chỉ mục: tổng (chi phí khi không có chỉ mục − chi phí khi có mọi ứng viên) của các truy vấn
         mà kế hoạch dùng chỉ mục đó.
      3. Gộp chỉ mục là tiền tố của một chỉ mục có tỉ lệ lợi ích / kích thước cao hơn vào chỉ mục đó.
      4. Chọn tham lam theo tỉ lệ lợi ích / kích thước cho tới hết ngân sách budget_mb.
      5. TryVariations trong variation_seconds giây.
    Tham số mặc định lấy theo bản của Kossmann et al.: 500 MB, tối đa 2 cột, 10 s, đổi tối đa 4 chỉ mục mỗi lần."""
    start = time.time()
    hypopg_ready(db)
    rng = random.Random(seed)

    # Bước 1: ứng viên của từng truy vấn (chỉ xét bảng thật, không xét view của hypopg)
    all_columns = db.table_columns()
    tables = {}
    for table in db.table_rows():
        tables[table] = all_columns[table]
    query_tables = {}
    candidates = {}
    for q, sql in queries.items():
        columns = query_columns(sql, tables)
        query_tables[q] = set(columns.keys())
        candidates[q] = set()
        for table, names in columns.items():
            for width in range(1, max_width + 1):
                for combo in itertools.permutations(names, width):
                    candidates[q].add((table, combo))
    whatif = WhatIf(db, queries, query_tables)

    # Bước 2: chi phí mỗi truy vấn khi không có chỉ mục và khi có mọi ứng viên của nó → lợi ích của từng chỉ mục
    benefit = {}  # chỉ mục → lợi ích
    candidate_count = 0
    for q in queries:
        candidate_count = candidate_count + len(candidates[q])
        cost_without = whatif.cost(q, set())
        cost_with, used = whatif.plan(q, candidates[q])
        for ix in used:
            if ix not in benefit:
                benefit[ix] = 0.0
            benefit[ix] = benefit[ix] + (cost_without - cost_with)

    def by_ratio(ix):
        """Khóa sắp xếp: tỉ lệ lợi ích / kích thước, hòa thì so tên chỉ mục (như bản gốc)."""
        return (benefit[ix] / whatif.sizes[ix], ix)

    # Bước 3: gộp chỉ mục bị bao (là tiền tố) vào chỉ mục có tỉ lệ cao hơn
    ranked = sorted(benefit, key=by_ratio, reverse=True)
    removed = set()
    for i in range(len(ranked)):
        high = ranked[i]
        if high in removed:
            continue
        for low in ranked[i + 1:]:
            if low in removed:
                continue
            if high[0] == low[0] and high[1][:len(low[1])] == low[1]:
                benefit[high] = benefit[high] + benefit[low]
                removed.add(low)
    remaining = []
    for ix in ranked:
        if ix not in removed:
            remaining.append(ix)
    remaining.sort(key=by_ratio, reverse=True)

    # Bước 4: chọn tham lam theo tỉ lệ trong ngân sách (bản gốc tính 1 MB = 10⁶ byte)
    budget = budget_mb * 1000 * 1000
    selected = []
    used_bytes = 0
    for ix in remaining:
        if used_bytes + whatif.sizes[ix] <= budget:
            selected.append(ix)
            used_bytes = used_bytes + whatif.sizes[ix]

    # Bước 5: TryVariations
    selected = try_variations(whatif, selected, remaining, budget, variation_seconds, max_removals, rng)
    whatif.use(set())

    indexes = []
    total = 0
    for ix in sorted(selected):
        indexes.append(index_sql(ix[0], ix[1]))
        total = total + whatif.sizes[ix]
    print(f"    {candidate_count} ứng viên, kế hoạch dùng {len(benefit)}, còn {len(remaining)} sau khi gộp → chọn "
          f"{len(indexes)} chỉ mục, ~{total / 1e6:.0f} / {budget_mb} MB ({time.time() - start:.0f} s phân tích)")

    # Bước 6: đo workload với các chỉ mục được chọn
    return evaluate_once(db, queries, [], indexes, start)


# ============================================================================= LlamaTune (MLOS)
# Không gian tham số của LlamaTune cho PostgreSQL 13: 112 tham số, khoảng giá trị theo đơn vị gốc của pg_settings
# (vd. shared_buffers tính theo trang 8 kB). Repo LlamaTune không ghi giấy phép nên tệp này không được chép vào thư mục;
# lần chạy đầu tải nó (ghim theo commit) về .cache/.
LLAMATUNE_KNOBS_URL = ("https://raw.githubusercontent.com/uw-mad-dash/llamatune/"
                       "288f9fb09cef480c3de53e14d883c871aad512fe/spaces/definitions/postgres-13.json")

# Giá trị đặc biệt mà LlamaTune lấy mẫu với xác suất 20% (adapters/bias_sampling.py của repo gốc): 0 = tắt tính năng,
# −1 = lấy theo một tham số khác. Bỏ old_snapshot_threshold vì PostgreSQL 17 không còn tham số này.
SPECIAL_VALUES = {
    "autovacuum_vacuum_cost_delay": -1,
    "autovacuum_vacuum_cost_limit": -1,
    "autovacuum_work_mem": -1,
    "backend_flush_after": 0,
    "bgwriter_flush_after": 0,
    "bgwriter_lru_maxpages": 0,
    "checkpoint_flush_after": 0,
    "effective_io_concurrency": 0,
    "geqo_generations": 0,
    "geqo_pool_size": 0,
    "max_parallel_workers_per_gather": 0,
    "max_prepared_transactions": 0,
    "temp_file_limit": -1,
    "vacuum_cost_delay": 0,
    "wal_buffers": -1,
    "wal_writer_flush_after": 0,
}


def llamatune_space(db, seed):
    """ConfigSpace gồm các tham số của LlamaTune mà PostgreSQL đang chạy có."""
    import ConfigSpace

    # Bước 1: đọc định nghĩa tham số (tải về .cache/ nếu chưa có)
    path = HERE / ".cache" / "llamatune_postgres-13.json"
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        subprocess.run(["curl", "-sSfL", LLAMATUNE_KNOBS_URL, "-o", str(path)], check=True)
    knobs = json.loads(path.read_text())

    # Bước 2: mỗi tham số thành một chiều: on/off → categorical, số nguyên / số thực → khoảng đều
    existing = set()
    for row in db.sql("SELECT name FROM pg_settings", fetch=True):
        existing.add(row[0])
    space = ConfigSpace.ConfigurationSpace(seed=seed)
    for knob in knobs:
        name = knob["name"]
        if name not in existing:
            print(f"    [bỏ qua] {name}: PostgreSQL này không có tham số này")
            continue
        if knob["type"] == "enum":
            dim = ConfigSpace.CategoricalHyperparameter(name, knob["choices"], default_value=knob["default"])
        elif knob["type"] == "integer":
            dim = ConfigSpace.UniformIntegerHyperparameter(name, knob["min"], knob["max"],
                                                            default_value=knob["default"])
        else:
            dim = ConfigSpace.UniformFloatHyperparameter(name, knob["min"], knob["max"], default_value=knob["default"])
        space.add(dim)
    return space


def llamatune_settings(config):
    """Một cấu hình (pandas Series: tham số → giá trị) → các lệnh ALTER SYSTEM SET. Số thực làm tròn 2 chữ số như hàm
    finalize_conf của LlamaTune; số nguyên không kèm đơn vị, nên PostgreSQL hiểu theo đơn vị gốc của tham số."""
    settings = []
    for name, value in config.items():
        if isinstance(value, str):
            text = f"'{value}'"
        elif isinstance(value, float):
            text = str(round(value, 2))
        else:
            text = str(int(value))
        settings.append(f"ALTER SYSTEM SET {name} = {text}")
    return settings


def fix_reverse_lookup(optimizer, space):
    """Vá lỗi của mlos_core 0.6.2 với numpy 2. Adapter LlamaTune nhớ mỗi cấu hình đã gợi ý (giá trị kiểu np.int64) để
    lúc báo điểm tìm lại điểm 16 chiều của nó, nhưng lúc tìm lại thì dùng giá trị kiểu int. ConfigSpace băm cấu hình
    theo repr, mà repr(np.int64(5)) khác repr(5), nên không tìm thấy. Ở đây lưu lại các khóa với giá trị kiểu Python."""
    import ConfigSpace
    adapter = optimizer.space_adapter
    fixed = {}
    for config, low_dim in adapter._suggested_configs.items():
        values = {}
        for name, value in dict(config).items():
            if isinstance(value, np.generic):
                value = value.item()  # np.int64(5) → 5
            values[name] = value
        fixed[ConfigSpace.Configuration(space, values=values)] = low_dim
    adapter._suggested_configs = fixed


def special_param_value_scaler(adapter, param, input_value):
    """Bản sửa của LlamaTuneAdapter._special_param_value_scaler (mlos_core 0.6.2), cho các tham số có giá trị đặc biệt.
    input_value ∈ [0, 1] là giá trị chuẩn hóa của tham số sau phép chiếu HeSBO. Đoạn đầu [0, tổng tỉ lệ) ứng với các
    giá trị đặc biệt; phần còn lại được giãn đều ra cả khoảng giá trị ("Scale input value uniformly to non-special
    values", theo chú thích của chính hàm gốc).
    Hàm gốc đưa giá trị đã chuẩn hóa này qua param.to_vector, tức coi nó là giá trị thật (vd. temp_file_limit = 0,4 kB),
    nên mọi giá trị không đặc biệt rơi về 0 hoặc 1: temp_file_limit ≈ 1 kB làm truy vấn cần file tạm báo lỗi,
    max_parallel_workers_per_gather ≤ 1, effective_io_concurrency ≤ 1."""
    perc_sum = 0.0
    for special_value, biasing_perc in adapter._special_param_values_dict[param.name]:
        perc_sum = perc_sum + biasing_perc
        if input_value < perc_sum:
            return float(param.to_vector(special_value))  # vùng của giá trị đặc biệt (giống hàm gốc)

    # Phần không đặc biệt: giãn [perc_sum, 1] thành [0, 1], giữ ở dạng chuẩn hóa (hàm gốc gọi thêm to_vector ở đây)
    return float((input_value - perc_sum) / (1 - perc_sum))


def fix_special_values():
    """Thay hàm lỗi của mlos_core 0.6.2 bằng special_param_value_scaler ở trên, cho cả lớp LlamaTuneAdapter."""
    from mlos_core.spaces.adapters.llamatune import LlamaTuneAdapter
    LlamaTuneAdapter._special_param_value_scaler = special_param_value_scaler


def reconnect(db, attempts=30):
    """Kết nối lại sau khi PostgreSQL bị sập giữa chừng (vd. hết bộ nhớ); cần thì khởi động lại bằng pg_ctl."""
    for attempt in range(attempts):
        try:
            db.connect()
            return
        except psycopg2.OperationalError:
            time.sleep(1)
        if attempt == 10:  # sau 10 s vẫn không kết nối được: tiến trình postmaster đã dừng
            subprocess.run(EXEC + ["pg_ctl", "-D", DATA, "-l", f"{DATA}/server.log", "-w", "start"],
                           capture_output=True)
    db.connect()  # lần cuối: lỗi thì để lỗi hiện ra


def llamatune(db, queries, indexes, budget, trial_timeout=None, seed=0):
    """LlamaTune qua MLOS, với các thiết lập của cấu hình "llama" trong repo gốc:
      - SMAC (random forest) tìm trong không gian 16 chiều; phép chiếu ngẫu nhiên HeSBO đưa mỗi điểm về giá trị của
        111 tham số PostgreSQL;
      - giá trị đặc biệt (SPECIAL_VALUES) được chọn với xác suất 20%; mỗi tham số chia tối đa 10 000 mức;
      - 10 điểm khởi tạo ngẫu nhiên, sau đó 10% số gợi ý là ngẫu nhiên.
    Mỗi lần thử đặt mọi tham số, khởi động lại PostgreSQL và chạy cả workload. Chỉ mục của Dexter được tạo trước và
    giữ suốt quá trình ("LlamaTune + Dexter" trong Bảng 3).
    trial_timeout (giây) giới hạn từng lần thử; None thì một lần thử chỉ bị ngắt khi hết ngân sách `budget`, như bài
    báo (bài báo chỉ đặt timeout cho UDO và GPTuner). run.py mặc định 24 s, vì không có giới hạn này một cấu hình ngẫu
    nhiên tắt gần hết phép join (enable_* = off) chiếm trọn 2 giờ và không lần thử nào chạy xong. Lần thử bị
    ngắt báo cho SMAC thời gian đến lúc ngắt. Cấu hình làm PostgreSQL không khởi động được, bị sập hoặc báo lỗi khi chạy
    nhận điểm phạt = 2 × thời gian chạy xong tệ nhất đã gặp."""
    import pandas as pd
    from mlos_core.optimizers import OptimizerFactory, OptimizerType, SpaceAdapterType

    fix_special_values()  # vá lỗi lấy mẫu giá trị đặc biệt của mlos_core 0.6.2 (xem special_param_value_scaler)
    start = time.time()
    for ix in indexes:
        db.sql(ix)  # chỉ mục của Dexter được tạo trước, giữ suốt quá trình tuning

    space = llamatune_space(db, seed)
    optimizer = OptimizerFactory.create(
        parameter_space=space,
        optimization_targets=["time"],
        optimizer_type=OptimizerType.SMAC,
        optimizer_kwargs={"seed": seed, "max_trials": 100000, "n_random_init": 10, "n_random_probability": 0.1},
        space_adapter_type=SpaceAdapterType.LLAMATUNE,
        space_adapter_kwargs={"num_low_dims": 16, "special_param_values": SPECIAL_VALUES,
                              "max_unique_values_per_param": 10000},
    )
    print(f"    {len(space)} tham số → 16 chiều, {len(indexes)} chỉ mục của Dexter ({time.time() - start:.0f} s)")

    trace = []
    best_time = math.inf
    best_settings = []
    worst_time = 0.0  # thời gian của lần thử chạy xong chậm nhất, dùng cho điểm phạt
    while time.time() - start < budget:
        # Bước 1: SMAC gợi ý một điểm 16 chiều, LlamaTune đổi nó thành giá trị của từng tham số
        suggestion = optimizer.suggest()
        fix_reverse_lookup(optimizer, space)
        settings = llamatune_settings(suggestion.config)

        # Bước 2: áp dụng cấu hình (khởi động lại PostgreSQL) và chạy cả workload
        errors = db.set_config(settings)
        failed = False
        for error in errors:
            if error.startswith("PostgreSQL failed to restart"):
                failed = True
        timeout = budget - (time.time() - start)
        if trial_timeout:
            timeout = min(timeout, trial_timeout)
        t = 0.0
        ok = False
        if not failed:
            try:
                t, ok = db.run_workload(queries, timeout)
            except psycopg2.Error as e:  # vd. vượt temp_file_limit, hoặc PostgreSQL bị sập
                print(f"    [lỗi khi chạy] {str(e).splitlines()[0]}")
                failed = True
                reconnect(db)

        # Bước 3: điểm cho SMAC
        out_of_budget = False
        if ok:
            score = t
            worst_time = max(worst_time, t)
            if t < best_time:
                best_time = t
                best_settings = settings
        elif failed:
            if worst_time > 0:
                score = 2 * worst_time
            else:
                score = budget
        elif time.time() - start >= budget:
            out_of_budget = True  # bị ngắt vì hết ngân sách: không còn lần thử nào sau đó, không cần báo điểm
        else:
            score = t  # bị ngắt ở trial_timeout: thời gian thật còn lớn hơn
        if not out_of_budget:
            optimizer.register(observations=suggestion.complete(pd.Series({"time": score})))

        # Bước 4: ghi lại
        entry = {"elapsed": time.time() - start, "time": t, "complete": ok, "failed": failed, "best": best_time}
        trace.append(entry)
        if ok:
            note = ""
        elif failed:
            note = "(cấu hình lỗi)"
        else:
            note = "(ngắt)"
        print(f"[{entry['elapsed']:7.1f}s] lần thử {len(trace):3d}: {t:7.2f} s {note}  best={best_time:.2f} s")

    db.drop_indexes()
    db.set_config([])
    return {
        "best_config": {"settings": best_settings, "indexes": indexes},
        "best_time": best_time,
        "tuning_time": time.time() - start,
        "trace": trace,
    }
