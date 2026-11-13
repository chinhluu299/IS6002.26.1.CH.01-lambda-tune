"""Demo chạy thật λ-Tune và D1-D3 trên PostgreSQL, hiện log từng bước trên trình duyệt.

    .venv/bin/python demo/server.py            # rồi mở http://127.0.0.1:8765
        --port 8765
        --open        tự mở trình duyệt

Demo gọi đúng các hàm của lambdatune.py, extensions.py và run.py, với cùng tham số như run_all.py (k = 5, t = 10 s,
α = 10, k₂ = 3). Nó không sửa các tệp đó và không ghi vào results/ hay configs*/.

Để thấy từng bước bên trong Alg. 2-4, demo bọc (monkeypatch) vài hàm khi chạy:
    PG.execute / PG.sql / PG.set_config / PG.drop_indexes   chạy truy vấn, tạo chỉ mục, đổi tham số (lớp con LoggedPG)
    lambdatune.evaluate                                      một lượt đánh giá của Alg. 3
    lambdatune.find_optimal_order                            thứ tự truy vấn của Alg. 4
    extensions.explain_analyze                               từng truy vấn được đo lại ở D2
    print của lambdatune, extensions, run                    các dòng log sẵn có (dòng trace của Alg. 2, D3, ...)
Hàm bọc chỉ ghi sự kiện rồi gọi hàm gốc với nguyên tham số, nên kết quả giống khi chạy run.py.

Chế độ LLM: "đã lưu" đọc phản hồi của lần chạy chính thức trong configs*/ (không gọi API); "gọi thật" gọi ask_llm như
run.py, cần OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL trong .env.

Mỗi lần chạy ghi vào demo/runs/<thời điểm>_<biến thể>/: log.txt (toàn bộ log), events.jsonl (sự kiện cho trang web, dùng
để xem lại), prompt.txt, prompt_feedback.txt (D2), result.json, summary.json, và configs*/ nếu gọi LLM thật.
"""
import argparse
import builtins
import json
import math
import os
import re
import sys
import threading
import time
import traceback
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import psycopg2

DEMO = Path(__file__).resolve().parent
ROOT = DEMO.parent
sys.path.insert(0, str(ROOT))

import extensions  # noqa: E402
import lambdatune  # noqa: E402
import run  # noqa: E402
from extensions import (feedback_prompt, filter_values, generate_prompt_ext, slowest_queries,  # noqa: E402
                        summarize_plan, table_sizes)
from lambdatune import (PG, ask_llm, compress, generate_prompt, join_values, machine, parse_config,  # noqa: E402
                        tokens)

RUNS = DEMO / "runs"
RESULTS = ROOT / "results"
QUERIES = run.QUERIES
T0 = 10.0     # timeout vòng đầu (run.py --t0, bài báo: 10 s)
ALPHA = 10.0  # hệ số tăng timeout (run.py --alpha, bài báo: 10)
K = 5         # số cấu hình xin LLM ở Alg. 1 (run.py --k)
K2 = 3        # số cấu hình xin thêm ở vòng phản hồi D2 (run.py --k2)

# câu SQL → tên truy vấn, để biết PG.execute đang chạy truy vấn nào
QUERY_NAME = {}
for query_name, query_sql in QUERIES.items():
    QUERY_NAME[query_sql] = query_name

# Biến thể (D1, D2, D3) → kết quả của lần chạy chính thức (đóng băng) để so sánh
OFFICIAL = {
    (False, False, False): "lambda_tune.json",
    (True, False, True): "d1_d3.json",
    (False, True, False): "d2.json",
    (True, True, True): "d1_d2_d3.json",
}

CREATE_INDEX = re.compile(r"\s*CREATE\s+(UNIQUE\s+)?INDEX\b", re.I)

# Dòng log của config_select (lambdatune.py): một dòng sau mỗi lần gọi Update của Alg. 2
UPDATE_LINE = re.compile(r"^\[\s*(?P<elapsed>[\d.]+)s\] (?P<config>\S+)\s+timeout=\s*(?P<timeout>-?[\d.]+)s"
                         r"\s+xong\s+(?P<done>\d+)/(?P<total>\d+)\s+tạo chỉ mục\s+(?P<index>[\d.]+)s"
                         r"\s+best=(?P<best>[\d.]+|inf)s$")
# Dòng log của ask_llm khi gọi với k = 1: "[1/1] model: N ký tự, finish_reason=stop"
LLM_LINE = re.compile(r"^\[1/1\] (.*)$")

CURRENT = None  # lần chạy hiện tại (hoặc vừa xong)
START_LOCK = threading.Lock()


class Stopped(Exception):
    """Người trình bày bấm Dừng."""


def finite(x):
    """Số thực → làm tròn 4 chữ số; vô cùng / NaN → None (JSON không có inf)."""
    if math.isinf(x) or math.isnan(x):
        return None
    return round(x, 4)


def jsonable(value):
    """Đổi giá trị Python sang dạng ghi được ra JSON: tuple / set → list, inf → None."""
    if isinstance(value, float):
        return finite(value)
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            out[str(key)] = jsonable(item)
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.append(jsonable(item))
        return out
    if isinstance(value, set):
        out = []
        for item in sorted(value):
            out.append(jsonable(item))
        return out
    return value


def first_line(error):
    return str(error).strip().splitlines()[0]


def variant_name(options):
    name = "λ-Tune"
    for key, label in (("d1", "D1"), ("d2", "D2"), ("d3", "D3")):
        if options[key]:
            name = name + " + " + label
    return name


def variant_slug(options):
    slug = "lambda_tune"
    for key in ("d1", "d2", "d3"):
        if options[key]:
            slug = slug + "_" + key
    if options["live_llm"]:
        slug = slug + "_llm"
    return slug


def official(d1, d2, d3):
    """Kết quả của lần chạy chính thức cho biến thể này (results/*.json, chỉ đọc); None nếu không có."""
    name = OFFICIAL.get((d1, d2, d3))
    if name is None:
        return None
    path = RESULTS / name
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return {"file": "results/" + name, "best": data["best"], "best_time": data["best_time"],
            "tuning_time": data["tuning_time"]}


def saved_folder(d1, refine):
    """Thư mục chứa phản hồi LLM đã lưu của lần chạy chính thức."""
    if refine:
        if d1:
            return ROOT / "configs_d1_d2_d3"
        return ROOT / "configs_d2"
    if d1:
        return ROOT / "configs_ext"
    return ROOT / "configs"


def plan_stages(options):
    """Các bước của biến thể này, theo thứ tự chạy."""
    stages = []
    tags = []
    if options["d1"]:
        tags = ["D1"]
    stages.append({"id": "prompt", "kind": "prompt", "title": "Sinh prompt nén", "ref": "§3", "tags": tags})
    stages.append({"id": "llm", "kind": "llm", "title": f"Gọi LLM k = {K} lần", "ref": "Alg. 1", "tags": []})
    if options["d3"]:
        stages.append({"id": "d3", "kind": "d3", "title": "Kiểm định cấu hình", "ref": "D3", "tags": ["D3"]})
    stages.append({"id": "select", "kind": "select", "title": "Chọn cấu hình tốt nhất", "ref": "Alg. 2–4",
                   "tags": []})
    if options["d2"]:
        stages.append({"id": "feedback", "kind": "feedback", "title": "Đo lại cấu hình tốt nhất bằng EXPLAIN ANALYZE",
                       "ref": "D2", "tags": ["D2"]})
        stages.append({"id": "llm2", "kind": "llm", "title": f"Gọi LLM k₂ = {K2} lần với prompt phản hồi",
                       "ref": "D2", "tags": ["D2"]})
        if options["d3"]:
            stages.append({"id": "d3_2", "kind": "d3", "title": "Kiểm định cấu hình mới", "ref": "D3",
                           "tags": ["D3"]})
        stages.append({"id": "select2", "kind": "select", "title": "Chấm cấu hình mới với incumbent",
                       "ref": "Alg. 2 + D2", "tags": ["D2"]})
    stages.append({"id": "result", "kind": "result", "title": "Kết quả", "ref": "", "tags": []})
    return stages


# ============================================================================= một lần chạy
class Job:
    """Một lần chạy demo: danh sách sự kiện (trang web đọc qua SSE), log.txt và events.jsonl trong demo/runs/<id>/."""

    def __init__(self, options):
        self.options = options
        self.variant = variant_name(options)
        self.start = time.time()
        stamp = datetime.fromtimestamp(self.start).strftime("%Y%m%d-%H%M%S")
        self.id = stamp + "_" + variant_slug(options)
        self.folder = RUNS / self.id
        self.folder.mkdir(parents=True)
        self.events = []
        self.cond = threading.Condition()
        self.finished = False
        self.status = "running"
        self.stop_requested = False
        self.db = None
        self.thread = None
        self.current_stage = None
        self.config_names = {}    # id(cấu hình) → tên, để biết evaluate() đang đánh giá cấu hình nào
        self.llm_progress = None  # (i, k) của lần gọi LLM đang chạy
        self.llm_time = 0.0
        self.log_file = open(self.folder / "log.txt", "w", encoding="utf-8")
        self.events_file = open(self.folder / "events.jsonl", "w", encoding="utf-8")

    def emit(self, kind, text=None, **data):
        """Ghi một sự kiện; `text` (nếu có) là một dòng log: hiện trên trang, ghi vào log.txt và in ra terminal."""
        if self.finished:
            if text is not None:
                builtins.print(text, flush=True)
            return
        event = {"type": kind, "ts": round(time.time() - self.start, 3), "stage": self.current_stage}
        if text is not None:
            event["text"] = text
        for key, value in data.items():
            event[key] = jsonable(value)
        with self.cond:
            event["i"] = len(self.events)
            self.events.append(event)
            self.events_file.write(json.dumps(event, ensure_ascii=False) + "\n")
            self.events_file.flush()
            if text is not None:
                line = f"[{event['ts']:8.2f}s] {text}"
                self.log_file.write(line + "\n")
                self.log_file.flush()
                builtins.print(line, flush=True)
            self.cond.notify_all()

    def stage(self, stage_id, status, text=None):
        if status == "start":
            self.current_stage = stage_id
        self.emit("stage", text, id=stage_id, status=status)

    def check_stop(self):
        if self.stop_requested:
            raise Stopped()

    def request_stop(self):
        """Đánh dấu dừng và hủy câu lệnh PostgreSQL đang chạy (nếu có)."""
        self.stop_requested = True
        db = self.db
        if db is None:
            return
        try:
            db.conn.cancel()
        except Exception:
            pass  # kết nối đang đóng (PostgreSQL đang khởi động lại): lệnh tiếp theo sẽ gặp check_stop

    def finish(self, status, result):
        self.status = status
        summary = {
            "id": self.id,
            "variant": self.variant,
            "live_llm": self.options["live_llm"],
            "status": status,
            "started": datetime.fromtimestamp(self.start).strftime("%d/%m/%Y %H:%M"),
            "duration": round(time.time() - self.start, 1),
            "best": result.get("best"),
            "best_time": jsonable(result.get("best_time")),
        }
        (self.folder / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        labels = {"done": "xong", "stopped": "đã dừng", "error": "lỗi"}
        self.emit("run_end", f"===== {labels[status]} sau {summary['duration']:.0f} s · log: demo/runs/{self.id}/log.txt",
                  status=status)
        with self.cond:
            self.finished = True
            self.cond.notify_all()
        self.log_file.close()
        self.events_file.close()


# ============================================================================= móc ghi log quanh code đã đóng băng
class LoggedPG(PG):
    """PG của lambdatune.py, thêm sự kiện cho các thao tác mà Alg. 2-3 và D2 dùng; mỗi phương thức gọi lại bản gốc."""

    def __init__(self, job):
        self.job = job
        super().__init__()

    def execute(self, query, timeout=math.inf):
        name = QUERY_NAME.get(query, "?")
        self.job.check_stop()
        self.job.emit("query_start", None, q=name, timeout=timeout)
        sec, ok = super().execute(query, timeout)
        self.job.check_stop()
        if ok:
            text = f"    {name} xong sau {sec:.2f} s"
        elif timeout < 0.1:
            text = f"    {name}: hết thời gian, không chạy"
        else:
            text = f"    {name} bị ngắt sau {sec:.2f} s (hết {timeout:.2f} s còn lại)"
        self.job.emit("query_end", text, q=name, sec=sec, ok=ok, timeout=timeout)
        return sec, ok

    def sql(self, cmd, fetch=False):
        if CREATE_INDEX.match(cmd) is None:
            return super().sql(cmd, fetch)
        self.job.check_stop()
        self.job.emit("index_start", None, sql=cmd)
        start = time.time()
        try:
            result = super().sql(cmd, fetch)
        except psycopg2.Error as e:
            self.job.emit("index_end", f"    lỗi khi tạo chỉ mục {cmd}: {first_line(e)}", sql=cmd,
                          sec=time.time() - start, ok=False)
            raise
        sec = time.time() - start
        self.job.emit("index_end", f"    {cmd} ({sec:.2f} s)", sql=cmd, sec=sec, ok=True)
        return result

    def set_config(self, settings):
        self.job.check_stop()
        if len(settings) == 0:
            text = "  ALTER SYSTEM RESET ALL, khởi động lại PostgreSQL: về cấu hình mặc định"
        else:
            text = f"  ALTER SYSTEM RESET ALL + {len(settings)} lệnh ALTER SYSTEM SET, khởi động lại PostgreSQL"
        self.job.emit("config_apply", text, settings=settings)
        start = time.time()
        errors = super().set_config(settings)
        sec = time.time() - start
        text = f"  PostgreSQL đã chạy lại sau {sec:.1f} s"
        if len(errors) > 0:
            text = text + f", {len(errors)} lệnh bị từ chối"
        self.job.emit("config_applied", text, sec=sec, errors=errors)
        return errors

    def drop_indexes(self):
        super().drop_indexes()
        self.job.emit("drop_indexes", "  DROP INDEX: xóa mọi chỉ mục do tuning tạo ra")


ORIGINAL_EVALUATE = lambdatune.evaluate
ORIGINAL_FIND_ORDER = lambdatune.find_optimal_order
ORIGINAL_EXPLAIN_ANALYZE = extensions.explain_analyze


def logged_evaluate(db, config, queries, t, meta, rows):
    """Bọc Alg. 3 (lambdatune.evaluate): ghi đầu và cuối mỗi lượt đánh giá."""
    job = CURRENT
    if job is None or job.finished:
        return ORIGINAL_EVALUATE(db, config, queries, t, meta, rows)
    name = job.config_names.get(id(config), "?")
    before = sorted(meta.completed)
    remaining = len(queries) - len(before)
    job.emit("eval_start", f"[Alg. 3] đánh giá {name}: timeout t = {t:.2f} s, {remaining} truy vấn chưa xong",
             config=name, t=t, completed=before, time_before=meta.time)
    ORIGINAL_EVALUATE(db, config, queries, t, meta, rows)
    job.emit("eval_end", None, config=name, done=len(meta.completed), complete=meta.complete, time=meta.time,
             index_time=meta.index_time)


def logged_find_order(Q, I, cost, max_units=13):
    """Bọc Alg. 4 + §5.4 (lambdatune.find_optimal_order): ghi thứ tự truy vấn và chỉ mục mỗi truy vấn cần."""
    order = ORIGINAL_FIND_ORDER(Q, I, cost, max_units)
    job = CURRENT
    if job is None or job.finished:
        return order
    needs = {}
    indexes = {}
    groups = set()
    for q in order:
        sqls = []
        for ix in I[q]:
            sqls.append(ix.sql)
            if ix.sql not in indexes:
                label = ix.table + "(" + ", ".join(ix.columns) + ")"
                indexes[ix.sql] = {"label": label, "table": ix.table, "cost": cost(ix)}
        needs[q] = sorted(sqls)
        groups.add(frozenset(I[q]))
    job.emit("order", f"[Alg. 4] thứ tự DP ({len(groups)} nhóm theo tập chỉ mục): {' '.join(order)}",
             order=order, needs=needs, indexes=indexes, groups=len(groups))
    return order


def logged_explain_analyze(db, sql, timeout):
    """Bọc extensions.explain_analyze (D2): ghi thời gian và toán tử tốn nhất của từng truy vấn."""
    job = CURRENT
    if job is None or job.finished:
        return ORIGINAL_EXPLAIN_ANALYZE(db, sql, timeout)
    name = QUERY_NAME.get(sql, "?")
    job.check_stop()
    job.emit("analyze_start", None, q=name)
    plan = ORIGINAL_EXPLAIN_ANALYZE(db, sql, timeout)
    job.check_stop()
    if plan is None:
        job.emit("analyze_end", f"  EXPLAIN ANALYZE {name}: quá {timeout:.0f} s", q=name, sec=None, ok=False)
        return plan
    sec = plan["Execution Time"] / 1000
    top = summarize_plan(plan["Plan"], 1)
    job.emit("analyze_end", f"  EXPLAIN ANALYZE {name}: {sec:.2f} s; toán tử tốn nhất: {top}", q=name, sec=sec,
             ok=True, top=top)
    return plan


def logged_print(*args, sep=" ", end="\n", file=None, flush=False):
    """Thay print trong lambdatune / extensions / run: mỗi dòng thành một sự kiện log."""
    job = CURRENT
    parts = []
    for arg in args:
        parts.append(str(arg))
    text = sep.join(parts)
    if job is None or job.finished:
        builtins.print(text, end=end, file=file, flush=flush)
        return
    for line in text.split("\n"):
        if line.strip() == "":
            continue
        handle_printed_line(job, line)


def handle_printed_line(job, line):
    match = UPDATE_LINE.match(line)
    if match is not None:
        best = float(match.group("best"))
        job.emit("update", line, config=match.group("config"), elapsed=float(match.group("elapsed")),
                 timeout=float(match.group("timeout")), done=int(match.group("done")),
                 total=int(match.group("total")), index_time=float(match.group("index")), best=best)
        return
    match = LLM_LINE.match(line)
    if match is not None and job.llm_progress is not None:
        i, k = job.llm_progress
        line = f"  [{i}/{k}] {match.group(1)}"
    job.emit("log", line)


def install_hooks():
    lambdatune.evaluate = logged_evaluate
    lambdatune.find_optimal_order = logged_find_order
    extensions.explain_analyze = logged_explain_analyze
    lambdatune.print = logged_print
    extensions.print = logged_print
    run.print = logged_print


# ============================================================================= các bước của pipeline
def pair_value(item):
    return item[1]


def step_prompt(job, db, d1, budget):
    """§3 (và D1): EXPLAIN → V(p) → ILP nén → prompt. Gọi generate_prompt / generate_prompt_ext như run.py prompt;
    các bước trung gian được tính lại bằng chính các hàm đó để hiển thị."""
    job.stage("prompt", "start", "── Sinh prompt nén (§3) ──")
    memory_gb, cores = machine()
    job.emit("machine", f"PostgreSQL chạy trên máy {memory_gb}GB RAM, {cores} nhân (hai số này được điền vào prompt)",
             memory=memory_gb, cores=cores)

    # §3.2: kế hoạch mặc định của optimizer cho từng truy vấn (EXPLAIN, không chạy truy vấn)
    plans = []
    for name, sql in QUERIES.items():
        job.check_stop()
        plan = db.explain(sql)
        plans.append(plan)
        job.emit("explain", f"  EXPLAIN {name}: chi phí ước lượng {plan['Total Cost']:,.0f}", q=name,
                 cost=plan["Total Cost"])
    V = join_values(plans)
    pairs = []
    for (a, b), value in sorted(V.items(), key=pair_value, reverse=True):
        pairs.append({"a": a, "b": b, "v": value})
    job.emit("join_values", f"[§3.2] V(p) = tổng chi phí các toán tử join dùng cặp cột p: {len(pairs)} cặp",
             pairs=pairs)

    # §3.3: ILP chọn những gì vào prompt trong ngân sách B (D1: snippet lọc cạnh tranh cùng ngân sách)
    snippets = None
    ilp_budget = budget
    if d1:
        snippets = filter_values(QUERIES.values(), plans, db.table_columns())
        sizes = table_sizes(db)
        ilp_budget = budget - tokens(sizes)
        if ilp_budget < 0:
            ilp_budget = 0
        filters = []
        for f, (value, line) in sorted(snippets.items(), key=snippet_value, reverse=True):
            filters.append({"f": f, "v": value, "line": line})
        job.emit("filters", f"[D1] V(f) của {len(filters)} cột có vị từ lọc; khối kích thước bảng ~{tokens(sizes)} token",
                 filters=filters, sizes=sizes, sizes_tokens=tokens(sizes))
    start = time.time()
    lines, kept_filters = compress(V, ilp_budget, snippets)
    sec = time.time() - start

    kept_pairs = []
    value = 0.0
    ilp_tokens = 0  # cách ILP đếm token: H_a + Σ H_b, H = tokens(cột) + 1
    for a in sorted(lines):
        ilp_tokens = ilp_tokens + tokens(a) + 1
        for b in sorted(lines[a]):
            ilp_tokens = ilp_tokens + tokens(b) + 1
            pair = sorted([a, b])
            kept_pairs.append(pair)
            value = value + V[(pair[0], pair[1])]
    total = sum(V.values())
    if math.isinf(budget):
        budget_text = "B = ∞"
    else:
        budget_text = f"B = {budget:g}"
    text = (f"[§3.3] ILP (Bảng 1) chọn {len(kept_pairs)}/{len(V)} cặp join, giữ {value / total:.0%} tổng V(p), "
            f"{ilp_tokens} token, {budget_text}")
    if d1:
        text = text + f"; thêm {len(kept_filters)}/{len(snippets)} snippet lọc"
    line_list = []
    for a in sorted(lines):
        line_list.append({"a": a, "b": sorted(lines[a])})
    job.emit("compress", text, lines=line_list, kept_pairs=kept_pairs, kept_filters=kept_filters, value=value,
             total=total, tokens=ilp_tokens, budget=budget, sec=sec)

    # §3.1: prompt hoàn chỉnh, sinh bằng đúng hàm mà run.py dùng
    if d1:
        prompt = generate_prompt_ext(db, QUERIES, memory_gb, cores, budget)
        official_file = RESULTS / "prompt_ext.txt"
    else:
        prompt = generate_prompt(db, QUERIES, memory_gb, cores, budget)
        official_file = RESULTS / "prompt.txt"
    same = official_file.exists() and prompt == official_file.read_text()
    (job.folder / "prompt.txt").write_text(prompt)
    all_sql = "".join(QUERIES.values())
    text = f"[§3.1] prompt ~{tokens(prompt)} token; nguyên văn 22 truy vấn SQL ~{tokens(all_sql)} token"
    if same:
        text = text + f"; trùng results/{official_file.name} của lần chạy chính thức"
    job.emit("prompt", text, prompt=prompt, tokens=tokens(prompt), sql_tokens=tokens(all_sql), same_as_official=same,
             official_file="results/" + official_file.name, extended=d1)
    job.emit("log", prompt)
    job.stage("prompt", "done")
    return prompt


def snippet_value(item):
    return item[1][0]


def emit_response(job, name, text, model, source, sec, prompt_match, path):
    settings, indexes = parse_config(text)
    index_sqls = []
    for ix in indexes:
        index_sqls.append(ix.sql)
    unique = len(set(settings))
    detail = f"{len(settings)} lệnh ALTER SYSTEM SET ({unique} khác nhau), {len(indexes)} CREATE INDEX"
    job.emit("llm_response", f"  {name}: {len(text)} ký tự → {detail}", name=name, model=model, source=source,
             sec=sec, chars=len(text), settings=settings, indexes=index_sqls, unique_settings=unique, raw=text,
             prompt_match=prompt_match, path=path)


def step_llm(job, stage_id, prompt, k, live, folder, prefix):
    """Alg. 1: k phản hồi LLM cho cùng một prompt. live = False thì đọc phản hồi đã lưu trong `folder`."""
    job.stage(stage_id, "start", f"── Gọi LLM k = {k} lần (Alg. 1) ──")
    if not live:
        files = sorted(folder.glob("*.json"))
        job.emit("llm_saved", f"[Alg. 1] dùng {len(files)} phản hồi LLM đã lưu trong {folder.name}/ (không gọi API)",
                 folder=folder.name, k=len(files))
        for path in files:
            job.check_stop()
            data = json.loads(path.read_text())
            emit_response(job, path.stem, data["response"], data.get("model"), "saved", None,
                          data.get("prompt") == prompt, folder.name + "/" + path.name)
        job.stage(stage_id, "done")
        return

    model = os.environ.get("LLM_MODEL", "gpt-4")  # như run.py --model
    responses = []
    for i in range(1, k + 1):
        job.check_stop()
        job.llm_progress = (i, k)
        job.emit("llm_call", f"[Alg. 1] gọi LLM lần {i}/{k} ({model}) với cùng prompt", i=i, k=k, model=model)
        start = time.time()
        text = ask_llm(prompt, 1, model, None)[0]  # ask_llm(prompt, k) là k lần gọi như thế này
        sec = time.time() - start
        job.llm_time = job.llm_time + sec
        responses.append(text)
        emit_response(job, f"{prefix}_{i}", text, model, "live", sec, True, None)
    job.llm_progress = None
    run.save_responses(folder, prompt, responses, model, prefix)
    job.emit("log", f"  đã lưu {k} phản hồi vào demo/runs/{job.id}/{folder.name}/")
    job.stage(stage_id, "done")


def step_load(job, db, folder, validate, stage_id):
    """Đọc cấu hình bằng run.load_configs; validate = True là D3."""
    if validate:
        job.stage(stage_id, "start", "── D3: kiểm định cấu hình trước khi chạy ──")
    configs, rejected = run.load_configs(db, folder, validate)
    data = {}
    kept_settings = 0
    kept_indexes = 0
    problems = 0
    for name, (settings, indexes) in configs.items():
        index_sqls = []
        for ix in indexes:
            index_sqls.append(ix.sql)
        data[name] = {"settings": settings, "indexes": index_sqls, "rejected": rejected.get(name, [])}
        kept_settings = kept_settings + len(settings)
        kept_indexes = kept_indexes + len(indexes)
        problems = problems + len(rejected.get(name, []))
    if validate:
        text = (f"[D3] giữ {kept_settings} lệnh tham số và {kept_indexes} chỉ mục; loại / kẹp {problems} lệnh "
                f"trên {len(configs)} cấu hình")
    else:
        text = f"{len(configs)} cấu hình ứng viên, không kiểm định (D3 tắt)"
    job.emit("configs", text, configs=data, validated=validate)
    if validate:
        job.stage(stage_id, "done")
    return configs, rejected


def step_select(job, db, configs, stage_id, incumbent, official_info):
    """Alg. 2-4: lambdatune.config_select, như run.py tune (incumbent = None) hoặc run.py refine (D2)."""
    if incumbent is None:
        title = "── Chọn cấu hình tốt nhất (Alg. 2–4) ──"
    else:
        title = "── D2: chấm cấu hình mới bằng Alg. 2 với incumbent ──"
    job.stage(stage_id, "start", title)
    job.config_names = {}
    summary = {}
    for name, config in configs.items():
        job.config_names[id(config)] = name
        summary[name] = {"settings": len(config[0]), "indexes": len(config[1])}
    if incumbent is None:
        text = f"[Alg. 2] chọn trong {len(configs)} cấu hình: t = {T0:g} s, α = {ALPHA:g}"
        incumbent_info = None
    else:
        text = (f"[Alg. 2 + D2] incumbent b = {incumbent[1]} (C_b = {incumbent[0]:.2f} s): bỏ qua các vòng ở dòng "
                f"6–13, mỗi cấu hình mới chạy một lượt với timeout C_b")
        incumbent_info = {"time": incumbent[0], "config": incumbent[1]}
    job.emit("select_start", text, configs=summary, order=list(configs), t0=T0, alpha=ALPHA,
             incumbent=incumbent_info, total=len(QUERIES), official=official_info)
    start = time.time()
    best, best_time, trace = lambdatune.config_select(db, QUERIES, configs, T0, ALPHA, incumbent=incumbent)
    sec = time.time() - start
    job.emit("select_end", f"=> cấu hình tốt nhất: {best} ({best_time:.2f} s), thời gian chọn {sec:.0f} s", best=best,
             best_time=best_time, sec=sec, trace=trace)
    job.stage(stage_id, "done")
    return best, best_time, sec


def step_feedback(job, db, base_prompt, configs, rejected, best, best_time):
    """D2 bước 1-2 của run.refine: dựng lại b như refine (qua parse_config), chạy feedback_prompt."""
    job.stage("feedback", "start", "── D2: chạy lại cấu hình tốt nhất với EXPLAIN ANALYZE ──")
    settings, indexes = configs[best]
    index_sqls = []
    for ix in indexes:
        index_sqls.append(ix.sql)
    b_config = parse_config(json.dumps({"commands": settings + index_sqls}))
    b_rejected = rejected.get(best, [])
    job.emit("feedback_start", f"[D2] b = {best} (C_b = {best_time:.2f} s): {len(settings)} tham số, "
             f"{len(index_sqls)} chỉ mục", best=best, best_time=best_time, settings=settings, indexes=index_sqls,
             rejected=b_rejected)
    start = time.time()
    prompt, times = feedback_prompt(db, QUERIES, base_prompt, b_config, best_time, b_rejected)
    sec = time.time() - start
    slow = slowest_queries(times, 3)
    (job.folder / "prompt_feedback.txt").write_text(prompt)
    appended = prompt[len(base_prompt):].strip()
    slow_text = []
    for name in slow:
        slow_text.append(f"{name} {times[name]:.2f} s")
    job.emit("feedback", f"[D2] 3 truy vấn chậm nhất: {', '.join(slow_text)}; phần thêm vào prompt:", times=times,
             slow=slow, appended=appended, sec=sec)
    job.emit("log", appended)
    job.stage("feedback", "done")
    return prompt, sec


def pipeline(job, db):
    options = job.options
    d1 = options["d1"]
    d2 = options["d2"]
    d3 = options["d3"]
    live = options["live_llm"]
    budget = math.inf
    if options["budget"] is not None:
        budget = float(options["budget"])

    # Bước 1-4: như run.py prompt → llm → tune (D1: --extended, D3: --validate)
    prompt = step_prompt(job, db, d1, budget)
    if live:
        folder = job.folder / "configs"
    else:
        folder = saved_folder(d1, False)
    step_llm(job, "llm", prompt, K, live, folder, "config")
    configs, rejected = step_load(job, db, folder, d3, "d3")
    best, best_time, select_time = step_select(job, db, configs, "select", None, official(d1, False, d3))

    final_best = best
    final_time = best_time
    settings, indexes = configs[best]
    tuning_time = select_time  # như run.py: không tính thời gian gọi LLM
    refine = None

    # Bước 5-8 (D2): như run.py refine
    if d2:
        feedback, feedback_time = step_feedback(job, db, prompt, configs, rejected, best, best_time)
        if live:
            folder2 = job.folder / "configs_refine"
        else:
            folder2 = saved_folder(d1, True)
        step_llm(job, "llm2", feedback, K2, live, folder2, "refine")
        configs2, rejected2 = step_load(job, db, folder2, d3, "d3_2")
        best2, best_time2, select_time2 = step_select(job, db, configs2, "select2", (best_time, best), None)
        if best2 in configs2:
            settings, indexes = configs2[best2]
        final_best = best2
        final_time = best_time2
        tuning_time = tuning_time + feedback_time + select_time2
        refine = {"feedback_time": feedback_time, "select_time": select_time2, "best": best2,
                  "best_time": best_time2, "improved": best2 in configs2}

    # Kết quả
    job.stage("result", "start", "── Kết quả ──")
    index_sqls = []
    for ix in indexes:
        index_sqls.append(ix.sql)
    result = {
        "variant": job.variant,
        "options": options,
        "best": final_best,
        "best_time": final_time,
        "tuning_time": tuning_time,
        "llm_time": job.llm_time,
        "base": {"best": best, "best_time": best_time, "select_time": select_time},
        "refine": refine,
        "best_config": {"settings": settings, "indexes": index_sqls},
        "official": official(d1, d2, d3),
        "folder": f"demo/runs/{job.id}",
    }
    (job.folder / "result.json").write_text(json.dumps(jsonable(result), indent=2, ensure_ascii=False))
    job.emit("result", f"=> {job.variant}: {final_best} ({final_time:.2f} s), thời gian tuning {tuning_time:.0f} s "
             f"(không tính gọi LLM)", **result)
    job.stage("result", "done")
    return result


def cleanup(job, db):
    """Sau khi dừng / lỗi: xóa chỉ mục và trả PostgreSQL về mặc định, gọi thẳng PG (không qua móc ghi log)."""
    try:
        if db.conn.closed:
            PG.connect(db)
        PG.drop_indexes(db)
        PG.set_config(db, [])
        job.emit("log", "đã xóa chỉ mục và trả PostgreSQL về cấu hình mặc định")
    except Exception as e:
        job.emit("log", f"[lỗi] dọn dẹp chưa xong ({first_line(e)}); chạy lại một lần demo hoặc run.py để dọn")


def run_job(job):
    db = None
    status = "done"
    result = {}
    try:
        mode = "LLM gọi thật" if job.options["live_llm"] else "phản hồi LLM đã lưu"
        job.emit("run_start", f"===== {job.variant} · {mode} · demo/runs/{job.id}/ =====", id=job.id,
                 variant=job.variant, options=job.options, stages=plan_stages(job.options), epoch=job.start,
                 folder=f"demo/runs/{job.id}", queries=list(QUERIES))
        try:
            db = LoggedPG(job)
        except psycopg2.Error as e:
            raise RuntimeError(f"không kết nối được PostgreSQL: {first_line(e)}. Container đã chạy chưa "
                               f"(docker compose up -d)?") from e
        job.db = db
        job.emit("log", "chuẩn bị như run.py: xóa chỉ mục còn sót, đưa PostgreSQL về cấu hình mặc định")
        db.drop_indexes()
        db.set_config([])
        result = pipeline(job, db)
    except Stopped:
        status = "stopped"
    except Exception as e:
        if job.stop_requested:
            status = "stopped"
        else:
            status = "error"
            job.emit("error", f"[lỗi] {first_line(e)}", message=str(e), trace=traceback.format_exc())
    if status != "done" and db is not None:
        cleanup(job, db)
    if db is not None:
        try:
            db.conn.close()
        except Exception:
            pass
    job.finish(status, result)


def start_job(body):
    """POST /api/run: kiểm tra tham số rồi chạy pipeline trong một luồng riêng."""
    global CURRENT
    options = {}
    for key in ("d1", "d2", "d3", "live_llm"):
        options[key] = bool(body.get(key, False))
    budget = body.get("budget")
    if budget in (None, ""):
        options["budget"] = None
    else:
        try:
            options["budget"] = float(budget)
        except (TypeError, ValueError):
            return 400, {"error": "Ngân sách B phải là số token"}
        if options["budget"] <= 0:
            return 400, {"error": "Ngân sách B phải lớn hơn 0"}
    if options["live_llm"] and os.environ.get("OPENAI_API_KEY", "") == "":
        return 400, {"error": "Chưa có OPENAI_API_KEY trong .env: chỉ chạy được với phản hồi đã lưu"}

    with START_LOCK:
        if CURRENT is not None and not CURRENT.finished:
            return 409, {"error": "Đang có một lần chạy khác", "id": CURRENT.id}
        job = Job(options)
        CURRENT = job
        job.thread = threading.Thread(target=run_job, args=(job,), name="demo-job")
        job.thread.start()
    return 200, {"id": job.id}


# ============================================================================= HTTP
def check_db():
    try:
        conn = psycopg2.connect(lambdatune.DSN + " dbname=tpch", connect_timeout=3)
    except psycopg2.Error as e:
        return {"ok": False, "text": first_line(e)}
    try:
        with conn.cursor() as cur:
            cur.execute("SHOW server_version")
            version = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM pg_indexes WHERE schemaname = 'public' "
                        "AND indexname NOT IN (SELECT conname FROM pg_constraint)")
            leftover = cur.fetchone()[0]
    finally:
        conn.close()
    return {"ok": True, "text": f"PostgreSQL {version.split()[0]}", "leftover_indexes": leftover}


def status():
    job = CURRENT
    info = {"running": False, "current": None}
    if job is not None:
        info["running"] = not job.finished
        info["current"] = {"id": job.id, "variant": job.variant, "finished": job.finished, "status": job.status}
    if info["running"]:
        info["db"] = {"ok": True, "text": "đang chạy demo"}
    else:
        info["db"] = check_db()
    key = os.environ.get("OPENAI_API_KEY", "")
    info["llm"] = {"configured": key != "", "model": os.environ.get("LLM_MODEL", "gpt-4")}
    return info


def list_runs():
    runs = []
    if not RUNS.exists():
        return runs
    for folder in sorted(RUNS.iterdir(), reverse=True):
        summary = folder / "summary.json"
        if summary.exists():
            runs.append(json.loads(summary.read_text()))
        elif (folder / "events.jsonl").exists():
            # đang chạy, hoặc server bị tắt giữa chừng: lấy tên biến thể từ sự kiện đầu tiên (run_start)
            state = "interrupted"
            if CURRENT is not None and CURRENT.id == folder.name and not CURRENT.finished:
                state = "running"
            info = {"id": folder.name, "variant": folder.name, "status": state}
            with open(folder / "events.jsonl", encoding="utf-8") as f:
                first = f.readline()
            if first.strip() != "":
                start = json.loads(first)
                info["variant"] = start.get("variant", folder.name)
                info["live_llm"] = start.get("options", {}).get("live_llm", False)
                if "epoch" in start:
                    info["started"] = datetime.fromtimestamp(start["epoch"]).strftime("%d/%m/%Y %H:%M")
            runs.append(info)
    return runs


class Handler(BaseHTTPRequestHandler):
    server_version = "lambdatune-demo"

    def log_message(self, format, *args):
        pass  # không in mỗi request ra terminal: terminal dành cho log của lần chạy

    def send_body(self, code, body, content_type):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, obj, code=200):
        body = json.dumps(jsonable(obj), ensure_ascii=False).encode("utf-8")
        self.send_body(code, body, "application/json; charset=utf-8")

    def do_GET(self):
        url = urlparse(self.path)
        params = parse_qs(url.query)
        if url.path == "/" or url.path == "/index.html":
            self.send_body(200, (DEMO / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif url.path == "/api/status":
            self.send_json(status())
        elif url.path == "/api/runs":
            self.send_json(list_runs())
        elif url.path.startswith("/api/runs/"):
            self.send_saved(url.path[len("/api/runs/"):])
        elif url.path == "/api/events":
            self.stream(params)
        else:
            self.send_json({"error": "không có trang này"}, 404)

    def do_POST(self):
        # Chỉ nhận JSON: trang web khác không gửi được kiểu này tới localhost nếu không qua preflight CORS
        content_type = self.headers.get("Content-Type", "").split(";")[0].strip()
        if content_type != "application/json":
            self.send_json({"error": "cần Content-Type: application/json"}, 415)
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw or b"{}")
        except ValueError:
            self.send_json({"error": "JSON không hợp lệ"}, 400)
            return
        if self.path == "/api/run":
            code, obj = start_job(body)
            self.send_json(obj, code)
        elif self.path == "/api/stop":
            job = CURRENT
            if job is None or job.finished:
                self.send_json({"error": "không có lần chạy nào đang chạy"}, 409)
                return
            job.emit("log", "[dừng] người dùng bấm Dừng: hủy câu lệnh đang chạy rồi dọn dẹp")
            job.request_stop()
            self.send_json({"ok": True})
        else:
            self.send_json({"error": "không có API này"}, 404)

    def send_saved(self, run_id):
        """Sự kiện của một lần chạy đã ghi (demo/runs/<id>/events.jsonl), để trang web dựng lại."""
        if re.fullmatch(r"[\w.-]+", run_id) is None:
            self.send_json({"error": "id không hợp lệ"}, 400)
            return
        path = RUNS / run_id / "events.jsonl"
        if not path.exists():
            self.send_json({"error": "không thấy lần chạy này"}, 404)
            return
        events = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() != "":
                events.append(json.loads(line))
        self.send_json(events)

    def stream(self, params):
        """Server-sent events của lần chạy hiện tại, bắt đầu từ sự kiện thứ `from` (hoặc sau Last-Event-ID)."""
        job = CURRENT
        run_id = params.get("run", [""])[0]
        if job is None or job.id != run_id:
            self.send_json({"error": "lần chạy này không còn chạy; mở lại từ danh sách đã ghi"}, 404)
            return
        index = int(params.get("from", ["0"])[0])
        last_id = self.headers.get("Last-Event-ID")
        if last_id is not None and last_id.isdigit():
            index = int(last_id) + 1
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            while True:
                with job.cond:
                    if index >= len(job.events) and not job.finished:
                        job.cond.wait(15)
                    new = job.events[index:]
                    finished = job.finished
                chunks = []
                for event in new:
                    chunks.append(f"id: {event['i']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n")
                index = index + len(new)
                done = finished and index >= len(job.events)
                if len(chunks) == 0 and not done:
                    chunks.append(": keepalive\n\n")
                if done:
                    chunks.append("event: end\ndata: {}\n\n")
                self.wfile.write("".join(chunks).encode("utf-8"))
                self.wfile.flush()
                if done:
                    return
        except (BrokenPipeError, ConnectionResetError):
            return


def main():
    ap = argparse.ArgumentParser(description="Demo chạy thật λ-Tune + D1-D3, log từng bước trên trình duyệt")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true", help="tự mở trình duyệt")
    args = ap.parse_args()

    install_hooks()
    RUNS.mkdir(exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    url = f"http://127.0.0.1:{args.port}"
    builtins.print(f"Demo λ-Tune: {url}   (Ctrl+C để tắt; log của mỗi lần chạy cũng in ở đây)", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        job = CURRENT
        if job is not None and not job.finished:
            builtins.print("đang dừng lần chạy và dọn dẹp PostgreSQL ...", flush=True)
            job.request_stop()
            job.thread.join(120)
    server.server_close()


if __name__ == "__main__":
    main()
