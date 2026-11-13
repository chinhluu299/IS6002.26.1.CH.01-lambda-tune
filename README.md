# λ-Tune: cài đặt lại trên PostgreSQL

> **Lưu ý:** mã nguồn chỉ phục vụ **mục đích học tập** (đồ án môn Hệ CSDL Tiên tiến), không dùng cho mục đích thương
> mại. Đây là bản cài đặt lại của nhóm, không phải mã của tác giả. Ý tưởng và thuật toán của λ-Tune thuộc về bài báo
> gốc bên dưới.

Thư mục này cài đặt lại pipeline của λ-Tune và chạy thực nghiệm trên PostgreSQL 17 với TPC-H SF1. Nó gồm:
- λ-Tune gốc: prompt nén, gọi LLM, Algorithm 2–4.
- Ba baseline mà bài báo dùng: Dexter, DB2 Advisor, LlamaTune.
- Ba phát triển của nhóm: D1 prompt mở rộng, D2 vòng phản hồi, D3 kiểm định cấu hình.

## Tài liệu tham khảo

**Bài báo gốc:** Victor Giannakouris, Immanuel Trummer. *λ-Tune: Harnessing Large Language Models for Automated
Database System Tuning.* Proceedings of the ACM on Management of Data (SIGMOD 2025), 3(1), 2025.
DOI [10.1145/3709652](https://dl.acm.org/doi/10.1145/3709652) · [arXiv:2411.03500](https://arxiv.org/abs/2411.03500) ·
mã gốc [github.com/gsvic/lambda-tune](https://github.com/gsvic/lambda-tune)

```bibtex
@article{giannakouris2025lambdatune,
  title   = {{$\lambda$-Tune}: Harnessing Large Language Models for Automated Database System Tuning},
  author  = {Giannakouris, Victor and Trummer, Immanuel},
  journal = {Proceedings of the ACM on Management of Data},
  volume  = {3},
  number  = {1},
  pages   = {1--26},
  year    = {2025},
  doi     = {10.1145/3709652}
}
```

**Công cụ và dữ liệu dùng trong thực nghiệm:**
- [TPC-H](https://www.tpc.org/tpch/), sinh bằng extension `tpch` của DuckDB.
- [Dexter](https://github.com/ankane/dexter).
- DB2 Advisor (Valentin et al., ICDE 2000), theo bản cài đặt của
  [hyrise/index_selection_evaluation](https://github.com/hyrise/index_selection_evaluation).
- [LlamaTune](https://github.com/uw-mad-dash/llamatune), chạy qua [MLOS](https://github.com/microsoft/MLOS) (`mlos_core`).
- [hypopg](https://github.com/HypoPG/hypopg).

## Cấu trúc

| Tệp | Nội dung |
|---|---|
| `lambdatune.py` | λ-Tune: sinh prompt + ILP (§3), gọi LLM (Alg. 1), chọn cấu hình (Alg. 2–4) |
| `extensions.py` | D1, D2, D3 |
| `baselines.py` | Dexter, DB2 Advisor, LlamaTune |
| `run.py` | Từng bước thực nghiệm (xem dưới) |
| `run_all.py`, `plot.py` | Chạy toàn bộ; vẽ hình và in bảng từ `results/` |
| `docker-compose.yml`, `load_tpch.py` | PostgreSQL 17 (kèm hypopg) trong Docker; sinh và nạp TPC-H |
| `queries/` | 22 truy vấn TPC-H |
| `configs*/`, `results/` | Phản hồi LLM, kết quả, hình và log; được tạo ra khi chạy |

## Chuẩn bị

- Python 3 và Docker. Với Docker Desktop, nên tăng RAM của máy ảo ở Settings → Resources (lần chạy chính thức dùng
  24 GB), vì RAM và số nhân ghi trong prompt là của máy chạy PostgreSQL.
- Một endpoint LLM theo API chuẩn OpenAI (`/chat/completions`):

  ```bash
  cp .env.example .env    # điền OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL
  ```

  `.env` chứa API key, nên không commit và không chia sẻ tệp này.

## Chạy toàn bộ

```bash
python3 run_all.py      # khoảng 3 giờ, trong đó LlamaTune 2 giờ
```

`run_all.py` lần lượt:
1. Tạo `.venv` từ `requirements.txt`.
2. Chạy `docker compose up -d` (PostgreSQL 17, cổng 5433, DB `tpch`).
3. Nạp TPC-H SF1 ở lần chạy đầu (khoảng 30 s).
4. Chạy mọi bước bên dưới, rồi `plot.py`.

Log của mỗi bước nằm ở `results/logs/`.

**Chạy lại sẽ ghi đè** `results/` và `configs*/`. LLM trả lời khác nhau mỗi lần, nên số liệu sẽ khác lần trước.
Muốn chỉ vẽ lại hình từ kết quả đã có: `python plot.py`.

## Chạy từng bước

```bash
source .venv/bin/activate
docker compose up -d && python load_tpch.py --sf 1      # nếu chưa có dữ liệu

# λ-Tune
python run.py prompt                                    # §3: prompt nén → results/prompt.txt
python run.py llm --k 5                                 # Alg. 1: k phản hồi LLM → configs/
python run.py tune --out lambda_tune.json               # Alg. 2–4: chọn cấu hình tốt nhất

# Baseline
python run.py dexter
python run.py db2advis
python run.py llamatune                                 # cần chạy dexter trước
python run.py tune --outlier --out outlier.json         # §4: thêm một cấu hình tệ (thêm --sequential để đối chứng)

# D1–D3
python run.py prompt --extended                         # D1 → results/prompt_ext.txt
python run.py llm --extended                            # → configs_ext/
python run.py tune --extended --validate --out d1_d3.json                # D1 + D3
python run.py refine --base lambda_tune.json --out d2.json               # λ-Tune + D2
python run.py refine --base d1_d3.json --validate --out d1_d2_d3.json    # D1 + D2 + D3

# Đo lại và vẽ hình
python run.py measure
python plot.py
```

Tham số hay dùng:

| Tham số | Ý nghĩa |
|---|---|
| `--t0 10 --alpha 10` | Timeout vòng đầu và hệ số tăng timeout của Alg. 2 (như bài báo) |
| `--budget B` | Ngân sách token B cho phần workload trong prompt (mặc định không giới hạn) |
| `--model M`, `--temperature T` | Mô hình LLM; mặc định lấy `LLM_MODEL` trong `.env` |
| `--no-llm` | `refine` không gọi LLM mà dùng phản hồi đã lưu trong `configs_<out>/` |
| `--llamatune-budget 7200`, `--llamatune-timeout 24` | Ngân sách LlamaTune (giây) và giới hạn mỗi lần thử (0 = không giới hạn) |

Muốn thử một LLM khác mà không gọi API, dán phản hồi vào trường `response` của `configs/*.json` rồi chạy `tune`.

## Ghi chú

- Mỗi lần đổi cấu hình, script chạy `ALTER SYSTEM SET` rồi khởi động lại PostgreSQL bằng `pg_ctl` trong container.
- Dexter chạy bằng image `ankane/dexter:v0.6.3`. LlamaTune tải không gian tham số của repo gốc về `.cache/` ở lần
  chạy đầu.
- Muốn dùng PostgreSQL cài sẵn trên máy thay cho Docker, đặt `PGDATA` (và nếu cần `LT_DSN`) trong `.env`.
- Xóa cụm và dữ liệu: `docker compose down -v`.
