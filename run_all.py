import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV = HERE / ".venv"
STEPS = [
    ("prompt", ["prompt"]),                                                  # §3: prompt nén
    ("llm", ["llm"]),                                                        # Alg. 1: gọi LLM k = 5 lần → configs/ (ghi đè)
    ("lambda_tune", ["tune", "--out", "lambda_tune.json"]),                  # λ-Tune (t = 10 s, α = 10 như bài báo)
    ("dexter", ["dexter"]),                                                  # baseline của bài báo: gợi ý chỉ mục
    ("db2advis", ["db2advis"]),
    ("llamatune", ["llamatune"]),                                            # LlamaTune + Dexter, 2 giờ, 24 s mỗi lần thử
    ("outlier", ["tune", "--outlier", "--out", "outlier.json"]),             # §4: thêm một cấu hình tệ vào các ứng viên
    ("outlier_sequential", ["tune", "--outlier", "--sequential", "--out", "outlier_sequential.json"]),
    ("prompt_ext", ["prompt", "--extended"]),                                # D1: prompt mở rộng
    ("llm_ext", ["llm", "--extended"]),                                      # → configs_ext/ (ghi đè)
    ("d1_d3", ["tune", "--extended", "--validate", "--out", "d1_d3.json"]),  # D1 + D3
    ("d2", ["refine", "--base", "lambda_tune.json", "--out", "d2.json"]),    # λ-Tune gốc + D2 → configs_d2/
    ("d1_d2_d3", ["refine", "--base", "d1_d3.json", "--validate", "--out", "d1_d2_d3.json"]),  # → configs_d1_d2_d3/
    ("measure", ["measure"]),                                                # đo lại cấu hình mỗi phương pháp chọn
]


def ensure_venv():
    """Chạy lại chính script này bằng Python của .venv (tạo .venv nếu chưa có)."""
    py = VENV / "bin" / "python"
    if Path(sys.prefix).resolve() == VENV.resolve():
        return  # đang chạy bằng Python của .venv rồi

    if not py.exists():
        print(">> tạo .venv")
        subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
        requirements = str(HERE / "requirements.txt")
        subprocess.run([str(py), "-m", "pip", "install", "-q", "-r", requirements], check=True)

    # Thay tiến trình hiện tại bằng: .venv/bin/python run_all.py <các tham số cũ>
    new_argv = [str(py), __file__] + sys.argv[1:]
    os.execv(py, new_argv)


def connect_with_retry(attempts=60, wait_seconds=2):
    """Kết nối PostgreSQL, thử lại vì lần đầu container còn chạy initdb. None nếu không kết nối được."""
    from lambdatune import PG
    for attempt in range(attempts):
        try:
            return PG()
        except Exception:
            time.sleep(wait_seconds)
    return None


def ensure_postgres():
    """Khởi động PostgreSQL (cụm cục bộ nếu có PGDATA, không thì container Docker) và nạp TPC-H SF1 nếu DB còn trống."""
    from lambdatune import PGDATA
    print(">> khởi động PostgreSQL")
    if PGDATA:  # cụm cục bộ
        status = subprocess.run(["pg_ctl", "-D", PGDATA, "status"], capture_output=True)
        if status.returncode != 0:
            subprocess.run(["pg_ctl", "-D", PGDATA, "-l", f"{PGDATA}/server.log", "-w", "start"], check=True)
    else:
        subprocess.run(["docker", "compose", "up", "-d"], check=True)

    db = connect_with_retry()
    if db is None:
        sys.exit("không kết nối được PostgreSQL")

    rows = db.sql("SELECT 1 FROM pg_tables WHERE tablename = 'lineitem'", fetch=True)
    if len(rows) == 0:
        print(">> nạp TPC-H SF1 (vài phút)")
        subprocess.run([sys.executable, "load_tpch.py", "--sf", "1"], check=True)


def run_step(name, args, logs, env):
    """Chạy `python run.py <args>`, ghi toàn bộ đầu ra vào logs/<name>.log. Trả về (mã thoát, đường dẫn log)."""
    log = logs / f"{name}.log"
    command = [sys.executable, "run.py"] + args
    with log.open("w") as f:
        code = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    return code, log


def main():
    ensure_venv()
    os.chdir(HERE)
    ensure_postgres()
    logs = HERE / "results" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"  # log được ghi ngay, không bị giữ trong bộ đệm

    for name, args in STEPS:
        print(f">> {name}", flush=True)
        code, log = run_step(name, args, logs, env)
        if code != 0:
            # Bước lỗi: in 20 dòng cuối của log rồi dừng
            lines = log.read_text().splitlines(keepends=True)
            print("".join(lines[-20:]))
            sys.exit(f"bước {name} lỗi (mã {code}), xem {log}")

    subprocess.run([sys.executable, "plot.py"], check=True)


if __name__ == "__main__":
    main()
