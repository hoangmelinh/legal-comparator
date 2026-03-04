import fitz  # PyMuPDF
from docx import Document

def extract_text(file_path):
    """Trích xuất text thuần từ file, hỗ trợ PDF và DOCX."""
    text = ""
    try:
        if file_path.lower().endswith('.pdf'):
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text("text") + "\n"
        elif file_path.lower().endswith('.docx'):
            doc = Document(file_path)
            text = "\n".join([p.text for p in doc.paragraphs])
        else:
            raise ValueError(f"Định dạng file không được hỗ trợ: {file_path}")
    except Exception as e:
        print(f"Lỗi khi trích xuất văn bản từ {file_path}: {e}")
    return text

if __name__ == "__main__":
    print("Đang test module Ingestion")
    test_data = extract_text("data/raw/sample.pdf")
    print(test_data[:200])