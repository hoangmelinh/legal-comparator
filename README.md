# Legal Comparator (Offline 100%)

Dự án so sánh hợp đồng và tư liệu pháp lý, hoạt động hoàn toàn cục bộ (offline) để đảm bảo quyền riêng tư và bảo mật dữ liệu tuyệt đối. Tóm tắt các điều chỉnh để ép chạy offline:

## 1. Đã sửa những gì?
- **Embedding (`Sentence-Transformers - BAAI/bge-m3`)**: Bị ép buộc không được sử dụng mạng nhờ vào cờ cấu hình biến môi trường cục bộ. Thêm exception thân thiện báo lỗi nếu máy chưa tải model cache.
- **LLM (`Ollama`)**: Chỉ sử dụng Local model (Qwen ...) thông qua `localhost:11434`.
- **Database (`NanoVectorDB`)**: Chỉ đọc/ghi ra file json nội tại (`legal_data.json`).
- **Data Ingestion**: Ưu tiên sử dụng Container [Gotenberg](https://gotenberg.dev/) (chạy qua Docker ở cổng `3000`) để chuyển đổi các định dạng Word, TXT sang PDF. Đây là giải pháp hoàn toàn Local và cho ra PDF layout chính xác 100%. Nếu Gotenberg không sẵn sàng, hệ thống sẽ tự tìm cách gọi MS Word/LibreOffice cũ để dự phòng.

## 2. Câu lệnh chạy hệ thống
**(Bắt buộc)** Chạy Docker Gotenberg ở dưới nền trước khi ingest dữ liệu Word/TXT:
```bash
docker run --rm -p 3000:3000 gotenberg/gotenberg:8
```

So sánh hai văn bản:
```bash
python run_comparison.py --doc_a BAN_CU --doc_b BAN_MOI

python run_comparison.py --ingest "E:\TTCS-Light-rag\legal-comparator\data\raw\test1.docx" --doc_id BAN_MOI
```
(Để import file mới, dùng cờ `--ingest <file_path> --doc_id <ID>`)

## 3. Các biến môi trường đang dùng (Đã được hardcode)
Hệ thống tự động kích hoạt 3 cờ sau bên trong mã nguồn (`src/models/embedding.py`) để Hugging Face Transformers không check/download từ Internet:
* `HF_DATASETS_OFFLINE=1`
* `TRANSFORMERS_OFFLINE=1`
* `HF_HUB_OFFLINE=1`

## 4. Cách kiểm tra chạy Offline
1. **Lần đầu tiên (Online)**: Mở internet, chạy một lệnh ingest hợp đồng bất kỳ để máy tải model LLM (Ollama) và trọng số SentenceTransformers (`BAAI/bge-m3`) về `~/.cache/huggingface`.
2. **Kiểm tra Offline**: Ngắt hoàn toàn Wifi / Ethernet trên máy tính. 
3. Chạy lệnh: `python run_comparison.py ...`. Nếu kết quả được in ra hoàn hảo mà không có bất kỳ "Internet timeout" nào, hệ thống đã chuẩn Local 100%. Nếu model HF bị mất, sẽ hiện thông báo Tiếng Việt yêu cầu kết nối lại để tải cache.

## 5. Điểm hạn chế
- Cấu hình máy phải đủ gánh LLM và Vector search cục bộ.
- Phải có mạng tải model vào bộ đệm ở lần *đầu tiên* sử dụng.
- OCR chưa tích hợp trọn vẹn 100% offline nếu gặp PDF hoàn toàn dạng ảnh (Image-only scan), chỉ lấy được rất ít văn bản.