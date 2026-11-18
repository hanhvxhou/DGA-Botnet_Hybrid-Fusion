# Thí nghiệm bổ sung cho bản sửa IJICS (DGA-HLF)

Đã chạy thử toàn bộ quy trình trên một dự án giả lập (dữ liệu nhỏ, mô hình tí hon). Kết quả: không có lỗi. Xác suất tính lại khớp với các file `test_predictions.csv` đã lưu đến sai số 1e-8. Riêng bước đo trên GPU và đo năng lượng NVML chưa chạy thử được, vì máy thử không có GPU.

## 1. Chép file vào thư mục dự án

Thư mục dự án là `D:\job\pycharm\DGABotnet2k25\DGA-HLF`. Chép ba file vào đó:

| File | Ghi chú |
|---|---|
| `revision_experiments.py` | file mới |
| `run_revision.ps1` | file mới |
| `selectAlgorithm.py` | ghi đè bản hiện có |

Bản `selectAlgorithm.py` mới vẫn tương thích ngược. Thay đổi duy nhất là cờ `DGA_BERT_RANDOM_INIT=1`, dùng cho thí nghiệm DistilBERT khởi tạo ngẫu nhiên: cùng kiến trúc, cùng tokenizer, cùng công thức huấn luyện, chỉ không dùng trọng số pretrained.

Script cần các file và thư mục sau, tất cả đã có sẵn trên máy:

- Mô hình và kết quả: `out_select/`, `out_select_seed43..46/`, `out_M6_v3_Markov_seed42/`, `out_M6_v3_Markov_fullseed43..46/`, `out_M1..5_seed42..46/`
- Dữ liệu: `DataNew/` (gồm cả `DataNew/Test/`) và `test_external/`

## 2. Chạy (để qua đêm, khoảng 5–6 giờ)

```powershell
cd D:\job\pycharm\DGABotnet2k25\DGA-HLF
.\run_revision.ps1
```

Script tự bỏ qua bước nào đã có kết quả. Nếu bị ngắt giữa chừng, chỉ cần chạy lại.

## 3. Gửi lại cho tôi

- File `rev_upload.zip`, khoảng 50–150 MB. File này gồm `rev_out/`, các file metadata nhỏ, `DataNew/` và `test_external/`.
- File log `revision_log_*.txt`

Nếu tải Tranco hoặc Umbrella thất bại, hãy tải thủ công và đặt cùng thư mục với tên `tranco_top-1m.csv.zip` và `umbrella_top-1m.csv.zip`:

- Tranco: https://tranco-list.eu/top-1m.csv.zip
- Umbrella: http://s3-us-west-1.amazonaws.com/umbrella-static/top-1m.csv.zip

## Script làm gì (ứng với ý kiến reviewer)

| Bước | Nội dung | Trả lời ý kiến |
|---|---|---|
| prep | Tạo các tập đánh giá:<br>- test đổi đuôi .com, test bỏ đuôi, test thêm `www.`<br>- tập DGA ngoài, lấy mẫu giống hệt `Test_M6`<br>- 50k benign từ Tranco và 50k từ Cisco Umbrella (dạng tên miền đăng ký và FQDN), không trùng DataNew | R1: benign độc lập, FPR thấp, đuôi tên miền và cách phân tách |
| prep | Đo mức trùng exact-domain giữa từng họ DGA ngoài và train/val/test | R1: overlap, họ unseen |
| prep | Tính lại 20 đặc trưng thủ công theo hai cách:<br>- Markov gốc của M6<br>- Markov đã sửa: thêm BOS/EOS, V=37, tổng xác suất có điều kiện bằng 1, ký tự lạ bị loại | R2: phương trình 2–5, vocab, ký tự lạ |
| prep | Bảng đặc trưng cho vài tên miền ví dụ; đặc tả từ điển NLTK | R2: worked examples, meaning_ratio |
| prep | Thống kê tokenization theo subtype; kiểm tra DistilBERT và BERT có dùng chung vocab không | R2: tokenisation |
| dump | Với mỗi seed 42–46: xác suất của M1–M5, BiLSTM, string head, DistilBERT, BERT, XGB handcraft (Markov gốc và Markov đã sửa) trên tất cả các tập | R1: ngưỡng chọn trên val cho mọi mô hình, calibration, CI theo cụm họ; R2: complementarity theo họ |
| dump | Early fusion với cùng đầu vào: [BiLSTM emb + DistilBERT CLS] → MLP và → XGB; thêm biến thể có 20 đặc trưng | R1: early vs late fusion |
| dump | DistilBERT khởi tạo ngẫu nhiên, 5 seed | R2: tách riêng đóng góp của pretraining |
| latency | Độ trễ end-to-end tính từ chuỗi thô:<br>- batch 1/32/256<br>- p50/p95/p99, throughput<br>- có và không có nhánh XGB<br>- trên GPU và CPU<br>- năng lượng GPU đo bằng NVML | R1: chi phí tính toán, năng lượng |

Các biến thể w_min (0.05–0.20) và các ensemble 3 nhánh sẽ do tôi tính lại offline từ xác suất validation. Kiểm tra trên bản giả lập cho thấy cách tính này tái tạo đúng bộ trọng số, ngưỡng và F1 mà M6 đã chọn. Vì vậy không cần chạy lại M6.
