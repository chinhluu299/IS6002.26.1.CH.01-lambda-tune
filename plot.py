import json
import math

import matplotlib.pyplot as plt

from run import METHODS, RESULTS as R

INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
plt.rcParams.update({
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
})

# Các cột của Bảng 3 và Bảng 4: hai phương pháp của bài báo đã cài đặt, rồi ba cải tiến
TABLE_COLUMNS = ["lambda_tune", "llamatune", "d1_d3", "d2", "d1_d2_d3"]


def load(name):
    """Đọc results/<name> (JSON); None nếu chưa có tệp này."""
    path = R / name
    if not path.exists():
        return None
    return json.loads(path.read_text())


def method_runs(methods):
    """[(tên hiển thị, tên tệp JSON)] cho các phương pháp trong METHODS."""
    runs = []
    for m in methods:
        runs.append((METHODS[m], f"{m}.json"))
    return runs


def trajectory(name, runs, title):
    """Hình 4: workload tốt nhất tìm được theo thời gian tuning, mỗi đường một phương pháp (đọc trace trong JSON).
    runs = [(nhãn, tên tệp JSON)]; tệp chưa có thì bỏ qua đường đó. Như bài báo, phương pháp không có cấu hình nào
    chạy xong trong thời gian tuning được ghi bằng nét đứt trong chú thích.
    Đường đứng trước được vẽ đè lên đường đứng sau: trace của D2 nối tiếp trace của lần tune gốc nên trùng với nó ở
    đoạn đầu, vẽ đè thì vẫn thấy đường gốc, còn đường D2 hiện ra ở đoạn nối thêm."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    layer = len(runs) + 2  # zorder: đường đầu tiên nằm trên cùng
    for color, (label, file) in zip(COLORS, runs):
        layer = layer - 1
        res = load(file)
        if res is None:
            continue

        # Các điểm (thời điểm, workload tốt nhất) từ lúc đã có cấu hình chạy xong cả workload
        xs = []
        ys = []
        for entry in res.get("trace", []):
            if entry["best"] < math.inf:
                xs.append(entry["elapsed"])
                ys.append(entry["best"])
        if len(xs) == 0:
            ax.plot([], [], color=color, lw=2, ls="--",
                    label=f"{label} ({res['tuning_time']:.0f} s, không có cấu hình chạy xong)")
            continue
        xs.append(res["tuning_time"])  # kéo đường ngang tới lúc kết thúc tuning
        ys.append(ys[-1])

        ax.step(xs, ys, where="post", color=color, lw=2, zorder=layer, label=f"{label} ({res['tuning_time']:.0f} s)")
        ax.plot(xs[0], ys[0], "o", color=color, ms=6, zorder=layer)  # thời điểm có cấu hình đầu tiên chạy xong

    ax.set(title=title, xlabel="Thời gian tuning (s)", ylabel="Workload tốt nhất tìm được (s)", xlim=(0, None))
    ax.set_ylim(0, ax.get_ylim()[1] * 1.6)  # chừa chỗ cho legend
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(R / name, dpi=150)


def index_tools(m):
    """Hình 8: tổng thời gian workload khi chỉ dùng chỉ mục, tham số mặc định. m = nội dung measure.json.
    Như bài báo có cả cột không chỉ mục; trục tung theo thang log vì cột này lớn hơn các cột khác hàng chục lần."""
    bars = [("default", "Không chỉ mục"), ("dexter", "Dexter"), ("db2advis", "DB2 Advisor"),
            ("lambda_tune_indexes", "λ-Tune")]
    names = []
    labels = []
    for name, label in bars:
        if name in m:
            names.append(name)
            labels.append(label)
    if len(names) < 2:
        return

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for i in range(len(names)):
        total = sum(m[names[i]].values())
        if names[i] == "default":
            color = COLORS[1]
            text = f"≥ {total:.0f} s"  # có truy vấn bị cắt ở --cap, nên số đo là cận dưới
        else:
            color = COLORS[0]
            text = f"{total:.1f} s"
        ax.bar(i, total, 0.5, color=color)
        ax.annotate(text, (i, total), ha="center", va="bottom", color=INK, xytext=(0, 3), textcoords="offset points")

    ax.set_yscale("log")
    low, high = ax.get_ylim()
    ax.set_ylim(low, high * 2)  # chừa chỗ cho số trên cột cao nhất
    ax.set(xticks=list(range(len(names))), xticklabels=labels, ylabel="Workload (s, thang log)",
           title="TPC-H 1GB, PostgreSQL: chỉ dùng chỉ mục, tham số mặc định\n(như Hình 8)")
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(R / "fig_indexes.png", dpi=150)


def per_query(m):
    """Hình 5: thời gian từng truy vấn, mặc định và λ-Tune. m = nội dung measure.json: {phương pháp: {truy vấn: giây}}."""
    qs = list(m["default"])
    cap = max(m["default"].values())  # mức bị cắt của cấu hình mặc định (--cap)
    left = []
    right = []
    default_times = []
    lambda_times = []
    labels = []
    for i in range(len(qs)):
        q = qs[i]
        left.append(i - 0.2)
        right.append(i + 0.2)
        default_times.append(m["default"][q])
        lambda_times.append(m["lambda_tune"][q])
        labels.append(q[1:])  # "q01" → "01"

    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.bar(left, default_times, 0.38, color=COLORS[1], label="Mặc định, không chỉ mục")
    ax.bar(right, lambda_times, 0.38, color=COLORS[0], label="λ-Tune")
    for i in range(len(qs)):
        if m["default"][qs[i]] >= cap:
            ax.annotate(f"≥{cap:.0f}", (i - 0.2, cap), ha="center", va="bottom", color=MUTED, fontsize=8)
    ax.set_yscale("log")
    title = f"Tổng: mặc định ≥ {sum(m['default'].values()):.0f} s, λ-Tune {sum(m['lambda_tune'].values()):.1f} s"
    ax.set(xticks=list(range(len(qs))), xticklabels=labels, xlabel="Truy vấn TPC-H", ylabel="Thời gian (s, thang log)",
           title=title)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(R / "fig_per_query.png", dpi=150)


def scaled(value, best):
    """Cách ghi số của Bảng 3: cột tốt nhất ghi "1", các cột khác 2 chữ số thập phân (kể cả khi làm tròn thành 1.00)."""
    if value == best:
        return "1"
    return f"{value / best:.2f}"


def configs_evaluated(res):
    """Số cấu hình đã đánh giá (Bảng 4). λ-Tune và các cải tiến: số cấu hình khác nhau trong trace (một cấu hình có thể
    được chạy ở nhiều vòng); LlamaTune: mỗi lần thử là một cấu hình."""
    names = set()
    for entry in res.get("trace", []):
        if "config" in entry:
            names.add(entry["config"])
    if len(names) > 0:
        return len(names)
    return len(res.get("trace", []))


def markdown_row(cells):
    """Một dòng bảng Markdown."""
    return "| " + " | ".join(cells) + " |"


def tables(m):
    """Bảng 3 và Bảng 4 của bài báo cho các cột TABLE_COLUMNS đã có kết quả, cùng số liệu gốc.
    m = nội dung measure.json. Bảng 3 chia cho cột tốt nhất trong bảng (kể cả các cải tiến)."""
    # Bước 1: các cột đã có kết quả đo lại
    columns = []
    header = []
    for method in TABLE_COLUMNS:
        if method in m:
            columns.append(method)
            header.append(METHODS[method])
    if len(columns) == 0:
        return

    # Bước 2: tổng thời gian workload đo lại của mỗi cột, và cột tốt nhất
    total = {}
    for method in columns:
        total[method] = sum(m[method].values())
    best = min(total.values())

    # Bước 3: Bảng 3, chi phí chia cho cột tốt nhất
    print("Bảng 3: chi phí cấu hình tốt nhất của mỗi phương pháp, chia cho chi phí tốt nhất trong bảng")
    print(markdown_row(["Benchmark", "DBMS", "Initial Indexes"] + header))
    print(markdown_row(["---"] * (3 + len(columns))))
    cells = ["TPC-H 1GB", "PG", "No"]
    for method in columns:
        cells.append(scaled(total[method], best))
    print(markdown_row(cells))

    # Bước 4: Bảng 4, số cấu hình đã đánh giá
    print("\nBảng 4: số cấu hình đã đánh giá")
    print(markdown_row(["Scenario", "Initial Indexes"] + header))
    print(markdown_row(["---"] * (2 + len(columns))))
    cells = ["TPC-H 1GB", "No"]
    results = {}
    for method in columns:
        results[method] = load(f"{method}.json")
        if results[method] is None:
            cells.append("–")
        else:
            cells.append(str(configs_evaluated(results[method])))
    print(markdown_row(cells))

    # Bước 5: số liệu gốc: thời gian workload và thời gian tuning
    print("\nSố liệu gốc: workload đo lại (s, trung vị 3 lần mỗi truy vấn) và thời gian tuning (s)")
    print(markdown_row([""] + header))
    print(markdown_row(["---"] * (1 + len(columns))))
    workload_cells = ["Workload (s)"]
    tuning_cells = ["Tuning (s)"]
    for method in columns:
        workload_cells.append(f"{total[method]:.2f}")
        if results[method] is None:
            tuning_cells.append("–")
        else:
            tuning_cells.append(f"{results[method]['tuning_time']:.0f}")
    print(markdown_row(workload_cells))
    print(markdown_row(tuning_cells))
    if "default" in m:
        print(f"Mặc định, không chỉ mục: ≥ {sum(m['default'].values()):.2f} s (có truy vấn bị cắt ở --cap)")


def main():
    trajectory("fig_methods.png", [("λ-Tune", "lambda_tune.json"), ("LlamaTune + Dexter", "llamatune.json")],
               "Postgres - TPC-H 1GB, không có chỉ mục ban đầu (như Hình 4)")
    trajectory("fig_outlier.png", [("λ-Tune", "outlier.json"), ("Đánh giá tuần tự", "outlier_sequential.json")],
               "TPC-H SF1: có một cấu hình tệ trong k ứng viên")
    trajectory("fig_dev.png", method_runs(["lambda_tune", "d1_d3", "d2", "d1_d2_d3"]),
               "TPC-H SF1: λ-Tune và các cải tiến D1-D3")

    measured = load("measure.json")
    if measured is not None:
        per_query(measured)
        index_tools(measured)
        tables(measured)

    names = []
    for path in R.glob("fig_*.png"):
        names.append(path.name)
    names.sort()
    print("\nđã ghi", " ".join(names))


if __name__ == "__main__":
    main()
