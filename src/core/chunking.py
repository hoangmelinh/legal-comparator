import re

def structural_chunking(text, doc_id="doc1"):
    pattern = r'(?=(?:^|\n)\s*Điều\s+(\d+|[IVXLCDM]+)[:.]?)'
    
    splits = list(re.finditer(pattern, text))
    
    chunks = []
    
    for i, match in enumerate(splits):
        start = match.start()
        end = splits[i + 1].start() if i + 1 < len(splits) else len(text)
        
        chunk_text = text[start:end].strip()
        
        if len(chunk_text) > 30:
            article_number = match.group(1)
            
            chunks.append({
                "doc_id": doc_id,
                "article": article_number,
                "content": chunk_text,
                "start_char": start
            })
    
    return chunks