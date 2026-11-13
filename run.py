import argparse
import json
import math
import os
import time
from pathlib import Path

from baselines import db2advis, dexter, llamatune
from extensions import declared_specs, feedback_prompt, generate_prompt_ext, validate_config
from lambdatune import (HERE, PG, ConfigMeta, ask_llm, config_select, evaluate, generate_prompt, machine,
                        parse_config, tokens)

def load_queries():
    """{tên truy vấn: câu SQL} của 22 truy vấn TPC-H trong queries/, theo thứ tự q01 … q22."""
    queries = {}
    for path in sorted((HERE / "queries").glob("*.sql")):
        queries[path.stem] = path.read_text()
    return queries


QUERIES = load_queries()
RESULTS = HERE / "results"

# Tên tệp kết quả (results/<tên>.json) → tên hiển thị, theo nhãn của bài báo, cộng ba cải tiến D1-D3. `measure` đo lại
# cấu hình của các phương pháp này.
METHODS = {
    "lambda_tune": "λ-Tune",
    "llamatune": "LlamaTune",  # Bảng 3 ghi "LlamaTune"; chỉ mục của Dexter được tạo trước (§6.2, "LlamaTune + Dexter" ở Hình 4)
    "dexter": "Dexter",  # Hình 8: công cụ gợi ý chỉ mục, tham số mặc định
    "db2advis": "DB2 Advisor",
    "d1_d3": "λ-Tune + D1 + D3",
    "d2": "λ-Tune + D2",
    "d1_d2_d3": "λ-Tune + D1 + D2 + D3",
}


def save_result(name, obj):
    """Ghi kết quả ra results/<name> (JSON)."""
    text = json.dumps(obj, indent=2, ensure_ascii=False)
    (RESULTS / name).write_text(text)


def load_result(name):
    """Đọc kết quả results/<name> (JSON)."""
    text = (RESULTS / name).read_text()
    return json.loads(text)


def save_responses(folder, prompt, responses, model, prefix):
    """Ghi mỗi phản hồi LLM ra folder/<prefix>_<i>.json, kèm mô hình và prompt đã dùng."""
    folder.mkdir(exist_ok=True)
    number = 1
    for text in responses:
        data = {"model": model, "prompt": prompt, "response": text}
        path = folder / f"{prefix}_{number}.json"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        number = number + 1


def load_configs(db, folder, validate):
    """Phản hồi LLM trong `folder` → {tên: (settings, indexes)}; validate = lọc qua bộ kiểm định D3.
    Trả về (configs, {tên: [lệnh bị loại / kẹp]})."""
    # Bước 1: đọc và tách từng phản hồi thành cấu hình
    configs = {}
    prompts = {}
    for path in sorted(folder.glob("*.json")):
        data = json.loads(path.read_text())
        configs[path.stem] = parse_config(data["response"])
        prompts[path.stem] = data.get("prompt")

    # Bước 2 (D3): kiểm định từng cấu hình theo RAM và số nhân khai báo trong prompt đã sinh ra nó
    rejected = {}
    if not validate:
        return configs, rejected
    for name in list(configs.keys()):
        specs = declared_specs(prompts[name])
        if specs is None:
            specs = machine()  # phản hồi không kèm prompt: dùng thông số của máy
        memory_gb, cores = specs
        validated, problems = validate_config(db, configs[name], memory_gb, cores)
        configs[name] = validated
        rejected[name] = problems
        print(f"[D3] {name}: loại / kẹp {len(problems)} lệnh")
        for problem in problems:
            print(f"    {problem}")
    return configs, rejected


def refine(db, args):
    """D2: vòng phản hồi, xuất phát từ cấu hình tốt nhất b của một lần tune (results/<args.base>)."""
    # Bước 1: đọc kết quả của lần tune làm điểm xuất phát
    base = load_result(args.base)
    if not args.out:
        name = "refine"
    else:
        name = Path(args.out).stem
    args.extended = base["args"].get("extended", False)  # phản hồi phải nối tiếp đúng prompt đã sinh ra b
    if args.extended:
        base_prompt = (RESULTS / "prompt_ext.txt").read_text()
    else:
        base_prompt = (RESULTS / "prompt.txt").read_text()
    b_settings = base["best_config"]["settings"]
    b_indexes = base["best_config"]["indexes"]
    b_config = parse_config(json.dumps({"commands": b_settings + b_indexes}))
    base_rejected = base.get("rejected", {})
    b_rejected = base_rejected.get(base["best"], [])  # các lệnh của b đã bị D3 loại / kẹp (nếu có)

    # Bước 2: chạy lại b với EXPLAIN ANALYZE và tạo prompt phản hồi
    start = time.time()
    prompt, query_times = feedback_prompt(db, QUERIES, base_prompt, b_config, base["best_time"], b_rejected)
    feedback_time = time.time() - start
    (RESULTS / f"prompt_{name}.txt").write_text(prompt)
    print(prompt[len(base_prompt):].strip())

    # Bước 3: xin LLM k₂ cấu hình mới (bỏ qua với --no-llm: dùng phản hồi đã lưu)
    folder = HERE / f"configs_{name}"
    llm_time = 0.0
    if not args.no_llm:
        start = time.time()
        responses = ask_llm(prompt, args.k2, args.model, args.temperature)
        save_responses(folder, prompt, responses, args.model, "refine")
        llm_time = time.time() - start

    # Bước 4: đọc (và kiểm định, nếu có --validate) các cấu hình mới
    configs, rejected = load_configs(db, folder, args.validate)

    # Bước 5: Algorithm 2 với incumbent = (C_b, b): mỗi cấu hình mới bị cắt khi vượt C_b
    start = time.time()
    best, best_time, trace = config_select(db, QUERIES, configs, args.t0, args.alpha,
                                           incumbent=(base["best_time"], base["best"]))
    select_time = time.time() - start

    # Bước 6: cấu hình tốt nhất sau vòng phản hồi; không cấu hình mới nào vượt C_b thì giữ b
    if best in configs:
        settings, indexes = configs[best]
    else:
        settings, indexes = b_config
    index_sqls = []
    for ix in indexes:
        index_sqls.append(ix.sql)

    # Bước 7: trace nối tiếp trace của lần tune gốc (thời điểm cộng thêm thời gian đã dùng trước đó)
    offset = base["tuning_time"] + feedback_time
    full_trace = list(base["trace"])
    for entry in trace:
        shifted = dict(entry)
        shifted["elapsed"] = entry["elapsed"] + offset
        full_trace.append(shifted)

    all_rejected = dict(base_rejected)
    all_rejected.update(rejected)

    print(f"\n=> cấu hình tốt nhất: {best} ({best_time:.2f} s; trước vòng phản hồi {base['best_time']:.2f} s), "
          f"thêm {feedback_time:.0f} s đo EXPLAIN ANALYZE + {select_time:.0f} s đánh giá (gọi LLM {llm_time:.0f} s)")
    result = {
        "args": vars(args),
        "best": best,
        "best_time": best_time,
        "tuning_time": offset + select_time,  # không tính thời gian gọi LLM, giống tune
        "feedback_time": feedback_time,
        "select_time": select_time,
        "llm_time": llm_time,
        "best_config": {"settings": settings, "indexes": index_sqls},
        "rejected": all_rejected,
        "query_times": query_times,
        "trace": full_trace,
    }
    save_result(f"{name}.json", result)


def sequential(db, configs, cap):
    """Đối chứng cho Alg. 2: chạy hết cấu hình này rồi mới sang cấu hình khác (tối đa `cap` giây mỗi cấu hình)."""
    rows = db.table_rows()
    best_name = None
    best_time = math.inf
    trace = []
    start = time.time()
    for name, config in configs.items():
        m = ConfigMeta()
        evaluate(db, config, QUERIES, cap, m, rows)
        if m.complete and m.time < best_time:
            best_name = name
            best_time = m.time
        entry = {"elapsed": time.time() - start, "config": name, "time": m.time, "complete": m.complete,
                 "best": best_time}
        trace.append(entry)
        print(f"[{entry['elapsed']:7.1f}s] {name:<12} xong {len(m.completed):2d}/{len(QUERIES)}  {m.time:7.2f} s  "
              f"best={best_time:.2f} s")
    db.set_config([])
    return best_name, best_time, trace


def measure(db, settings, indexes, repeat, cap):
    """Chạy cả workload dưới một cấu hình (tạo sẵn mọi chỉ mục), lấy trung vị `repeat` lần cho mỗi truy vấn.
    Truy vấn có lần chạy bị ngắt ở `cap` giây thì ghi `cap`, nghĩa là ≥ cap."""
    db.set_config(settings)
    for ix in indexes:
        try:
            db.sql(ix)
        except Exception as e:
            print(f"    [bỏ qua] {ix}: {str(e).splitlines()[0]}")

    times = {}
    for q, sql in QUERIES.items():
        runs = []  # các cặp (số giây, xong trước cap?)
        for attempt in range(repeat):
            runs.append(db.execute(sql, cap))
        runs.sort()

        all_finished = True
        for sec, ok in runs:
            if not ok:
                all_finished = False
        if all_finished:
            median_run = runs[len(runs) // 2]
            times[q] = median_run[0]
        else:
            times[q] = cap

    db.drop_indexes()
    db.set_config([])
    print(f"    tổng {sum(times.values()):.2f} s")
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prompt", "llm", "tune", "refine", "dexter", "db2advis", "llamatune", "measure"])
    ap.add_argument("--budget", type=float, default=math.inf, help="ngân sách token B cho workload trong prompt")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL", "gpt-4"), help="tên mô hình (bài báo: GPT-4)")
    ap.add_argument("--temperature", type=float, default=None, help="mặc định: để API tự chọn")
    ap.add_argument("--t0", type=float, default=10.0, help="timeout vòng đầu (bài báo: 10 s)")
    ap.add_argument("--alpha", type=float, default=10.0, help="hệ số tăng timeout (bài báo: 10)")
    ap.add_argument("--outlier", action="store_true", help="tune: thêm một cấu hình tệ vào đầu danh sách (§4)")
    ap.add_argument("--sequential", action="store_true", help="tune: đánh giá tuần tự thay cho Alg. 2 (đối chứng)")
    ap.add_argument("--cap", type=float, default=120.0, help="giới hạn mỗi truy vấn khi đo cấu hình mặc định")
    ap.add_argument("--db2-budget-mb", type=float, default=500.0, help="ngân sách bộ nhớ cho chỉ mục của DB2 Advisor")
    ap.add_argument("--llamatune-budget", type=float, default=7200.0, help="ngân sách thời gian cho LlamaTune (giây)")
    ap.add_argument("--llamatune-timeout", type=float, default=24.0,
                    help="giới hạn mỗi lần thử của LlamaTune (giây); 0 = không giới hạn, như bài báo")
    ap.add_argument("--out", default=None)
    ap.add_argument("--extended", action="store_true", help="D1: prompt mở rộng (prompt_ext.txt, configs_ext/)")
    ap.add_argument("--validate", action="store_true", help="D3: kiểm định cấu hình LLM trước khi đánh giá")
    ap.add_argument("--base", default="lambda_tune.json", help="refine: kết quả tune làm điểm xuất phát (D2)")
    ap.add_argument("--k2", type=int, default=3, help="refine: số cấu hình xin thêm ở vòng phản hồi (D2)")
    ap.add_argument("--no-llm", action="store_true", help="refine: không gọi LLM, dùng phản hồi đã lưu")
    args = ap.parse_args()

    # D1: prompt mở rộng và các cấu hình sinh từ nó được để riêng
    if args.extended:
        cfg_dir = HERE / "configs_ext"
        prompt_file = RESULTS / "prompt_ext.txt"
    else:
        cfg_dir = HERE / "configs"
        prompt_file = RESULTS / "prompt.txt"
    RESULTS.mkdir(exist_ok=True)

    if args.cmd == "llm":  # chỉ gọi API, không cần DB
        prompt = prompt_file.read_text()
        responses = ask_llm(prompt, args.k, args.model, args.temperature)
        save_responses(cfg_dir, prompt, responses, args.model, "config")
        return

    db = PG()
    db.drop_indexes()  # bắt đầu từ trạng thái sạch, kể cả khi lần chạy trước bị dừng giữa chừng
    db.set_config([])

    if args.cmd == "prompt":
        memory_gb, cores = machine()
        if args.extended:
            prompt = generate_prompt_ext(db, QUERIES, memory_gb, cores, args.budget)  # D1
        else:
            prompt = generate_prompt(db, QUERIES, memory_gb, cores, args.budget)
        prompt_file.write_text(prompt)
        print(prompt)
        all_sql = "".join(QUERIES.values())
        print(f"\n-- cả prompt: ~{tokens(prompt)} token; nguyên văn {len(QUERIES)} truy vấn SQL: "
              f"~{tokens(all_sql)} token")

    elif args.cmd == "tune":
        configs, rejected = load_configs(db, cfg_dir, args.validate)
        if args.outlier:
            # Thêm một cấu hình tệ (mặc định, không chỉ mục) vào ĐẦU danh sách
            with_outlier = {"outlier": ([], [])}
            for name, config in configs.items():
                with_outlier[name] = config
            configs = with_outlier
        start = time.time()
        if args.sequential:
            best, best_time, trace = sequential(db, configs, cap=600)
        else:
            best, best_time, trace = config_select(db, QUERIES, configs, args.t0, args.alpha)
        tuning_time = time.time() - start
        print(f"\n=> cấu hình tốt nhất: {best} ({best_time:.2f} s), thời gian tuning {tuning_time:.0f} s")
        settings, indexes = configs[best]
        index_sqls = []
        for ix in indexes:
            index_sqls.append(ix.sql)
        result = {
            "args": vars(args),
            "best": best,
            "best_time": best_time,
            "tuning_time": tuning_time,
            "best_config": {"settings": settings, "indexes": index_sqls},
            "rejected": rejected,  # D3: {cấu hình: [lệnh bị loại / kẹp]}, rỗng nếu không có --validate
            "trace": trace,
        }
        if not args.out:
            save_result("tune.json", result)
        else:
            save_result(args.out, result)

    elif args.cmd == "refine":  # D2
        refine(db, args)

    elif args.cmd == "dexter":
        save_result("dexter.json", dexter(db, QUERIES))

    elif args.cmd == "db2advis":
        save_result("db2advis.json", db2advis(db, QUERIES, budget_mb=args.db2_budget_mb))

    elif args.cmd == "llamatune":
        # Như "LlamaTune + Dexter" trong Bảng 3: chỉ mục của Dexter được tạo trước khi tuning
        indexes = load_result("dexter.json")["best_config"]["indexes"]
        trial_timeout = None
        if args.llamatune_timeout > 0:
            trial_timeout = args.llamatune_timeout
        result = llamatune(db, QUERIES, indexes, args.llamatune_budget, trial_timeout=trial_timeout)
        save_result("llamatune.json", result)

    elif args.cmd == "measure":
        # Cấu hình mặc định chạy 1 lần (rất chậm), các phương pháp khác chạy 3 lần lấy trung vị
        print("default")
        res = {"default": measure(db, [], [], repeat=1, cap=args.cap)}
        for m in METHODS:
            if not (RESULTS / f"{m}.json").exists():
                continue  # phương pháp chưa chạy
            print(m)
            config = load_result(f"{m}.json")["best_config"]
            res[m] = measure(db, config["settings"], config["indexes"], repeat=3, cap=args.cap)
        if (RESULTS / "lambda_tune.json").exists():
            # Hình 8: chỉ dùng chỉ mục λ-Tune chọn, tham số mặc định, để so với Dexter và DB2 Advisor
            print("lambda_tune_indexes")
            config = load_result("lambda_tune.json")["best_config"]
            res["lambda_tune_indexes"] = measure(db, [], config["indexes"], repeat=3, cap=args.cap)
        save_result("measure.json", res)


if __name__ == "__main__":
    main()
