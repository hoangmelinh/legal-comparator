from src.core.comparator import _resolve_changed_spans_to_anchors

spans = [{"text": "năm", "start": 90, "end": 93, "token_count": 1}]
anchors = [
    {
        "page": 1,
        "text": "11.1. Giá trị tạm tính của Hợp đồng là 4.850.000.000 đồng (bốn tỷ tám trăm năm",
        "bbox": [50.0, 100.0, 550.0, 110.0],
        "text_items": [
            {
                "text": "11.1. Giá trị tạm tính của Hợp đồng là 4.850.000.000 đồng (bốn ",
                "bbox": [50.0, 100.0, 400.0, 110.0],
            },
            {"text": "tỷ ", "bbox": [400.0, 100.0, 420.0, 110.0]},
            {"text": "tám ", "bbox": [420.0, 100.0, 450.0, 110.0]},
            {"text": "trăm ", "bbox": [450.0, 100.0, 490.0, 110.0]},
            {"text": "năm", "bbox": [490.0, 100.0, 550.0, 110.0]},
        ],
        "block_index": 1,
        "line_index": 1,
    }
]
prov = {"anchors": anchors}
source_text = (
    "11.1. Giá trị tạm tính của Hợp đồng là 4.850.000.000 đồng (bốn tỷ tám trăm năm"
)
# "năm" starts at 75? Wait.
# "11.1. Giá trị tạm tính của Hợp đồng là 4.850.000.000 đồng (bốn tỷ tám trăm năm" length:
print("Source text len:", len(source_text))
n_idx = source_text.find("năm")
print("năm index:", n_idx)

spans[0]["start"] = n_idx
spans[0]["end"] = n_idx + 3

res = _resolve_changed_spans_to_anchors(spans, prov, source_text)
print("Before fix:", res[0] if res else "No result")


# We will implement the new logic here to test.
def calc_new(anchor, offset_start, offset_end):
    text_items = anchor.get("text_items", [])
    current_idx = 0
    start_box = None
    end_box = None
    for item in text_items:
        item_text = item.get("text", "")
        item_len = len(item_text)
        item_start = current_idx
        item_end = current_idx + item_len
        if start_box is None and item_start <= offset_start < item_end:
            ibox = item.get("bbox")
            local_offset = offset_start - item_start
            char_w = (ibox[2] - ibox[0]) / max(1, item_len)
            start_box = ibox[0] + local_offset * char_w
        if end_box is None and item_start < offset_end <= item_end:
            ibox = item.get("bbox")
            local_offset = offset_end - item_start
            char_w = (ibox[2] - ibox[0]) / max(1, item_len)
            end_box = ibox[0] + local_offset * char_w
        if end_box is None and offset_end == item_end:  # exact match at the end
            ibox = item.get("bbox")
            end_box = ibox[2]
        current_idx += item_len
    return start_box, end_box


print("New logic:", calc_new(anchors[0], spans[0]["start"], spans[0]["end"]))
