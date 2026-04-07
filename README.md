# Legal Comparator

Du an so sanh van ban phap ly/hop dong chay local, gom:
- backend FastAPI
- frontend static trong `static/`
- ingest/chunk/vector compare
- compare bang Ollama khi can phan tich noi dung

## Chay local

Kich hoat moi truong va cai dependency:

```powershell
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Chay backend:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Mo frontend:

- `http://127.0.0.1:8000/`
- docs API: `http://127.0.0.1:8000/docs`

Neu muon chay CLI ingest/demo:

```powershell
.\.venv\Scripts\python.exe main.py
```

## Luong dung

1. Upload `file_1`
2. Doi backend xu ly xong
3. Upload `file_2`
4. Doi backend xu ly xong
5. Bam compare
6. Xem PDF preview, danh sach thay doi va report

Neu upload lai `file_1`, he thong se reset toan bo bo file/ket qua cua compare truoc, xoa cac file tam sinh ra tu lan chay cu, va bat dau mot phien moi hoan toan.
Neu upload lai `file_2`, he thong se xoa du lieu tam cua `file_2` va ket qua compare cu lien quan den workflow hien tai.

## API chinh

- `GET /api/health`
- `POST /api/documents/upload?workflow_id=<id>&slot=file_1|file_2`
- `GET /api/progress/{job_id}`
- `POST /api/compare`
- `GET /api/compare/{compare_job_id}/result`
- `GET /api/documents/{document_id}/pdf?workflow_id=<id>`
- `GET /api/workflows/{workflow_id}`

Backend tra ve `document_a.pdf_url` va `document_b.pdf_url` da kem `workflow_id`
de frontend load dung PDF cua workflow hien tai.

## Ollama

So sanh noi dung co thay doi thuc su van dung Ollama:

- URL mac dinh: `http://localhost:11434`
- model mac dinh: `qwen2.5:7b`

Case hai file giong nhau hoac thay doi rat nho co the duoc bypass LLM theo logic hien tai.
Neu chua co Ollama, cac flow upload va ingest van chay duoc, chi co phan compare semantic co the bi han che neu can model moi.

Neu muon dung Gotenberg cho DOCX, service local se nghe o port `3000`.

## Toi uu / Caching

- **Fresh-run upload flow**: Moi lan upload `file_1` se xoa sach file tam, cache tam, compare output cu, metadata/job data cua workflow truoc va ingest lai tu dau.
- **Khong reuse file cu**: Flow upload/compare khong con tim file da co, khong dung hash de bo qua xu ly, va khong tai su dung ket qua ingest cu cho file moi.
- **Dinh dang**: PDF va txt duoc xu ly fallback nhanh, docx moi su dung Gotenberg de render PDF view.
- **Cleanup an toan**: Chi xoa artifact sinh ra trong qua trinh xu ly nhu upload tam, normalized PDF, preview, text cache, vector/chunk cache va compare result cu.

## Test

Syntax/backend:

```powershell
.\.venv\Scripts\python.exe -m py_compile app.py src\core\comparator.py src\core\ingestion.py src\core\llm.py
```

Test API flow:

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_api_flow.py
```

Ban cung co the chay:

```powershell
.\.venv\Scripts\python.exe -m py_compile app.py main.py run_comparison.py src\core\ingestion.py src\core\comparator.py src\core\llm.py src\core\cache.py src\core\document_registry.py src\database\vector_store.py
```

## Ghi chu

- `data/raw/e2e_same_a.txt` va `data/raw/e2e_same_b.txt` la file nho de test runtime local.
- Frontend la static app, duoc serve truc tiep boi FastAPI qua `app.py`.
