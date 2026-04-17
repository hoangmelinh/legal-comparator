import re


def clean_legal_text(text):

    text = re.sub(r"(?<![.!?])\n\s*([a-zà-ỹ])", r" \1", text)

    text = re.sub(r" +", " ", text)

    return text.strip()
