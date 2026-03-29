import sys
import os
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.append('e:/TTCS-Light-rag/legal-comparator')

from src.core.chunking import structural_chunking

text = """
CỘNG HOÀ XÃ HỘI CHỦ NGHĨA VIỆT NAM
Độc lập - Tự do - Hạnh phúc

Luật số 123/2024/QH15

Điều 1. Phạm vi điều chỉnh
Luật này quy định về...

Điều 2. Đối tượng áp dụng
1. Công dân Việt Nam.

Điều 3. Giải thích từ ngữ
Trong Luật này, các từ ngữ dưới đây được hiểu như sau:
"""

chunks = structural_chunking(text)
for i, c in enumerate(chunks):
    print(f"Chunk {i}: Article: {c['article']}")
    print(f"Content:\n{c['content']}\n")
    print("-" * 40)
