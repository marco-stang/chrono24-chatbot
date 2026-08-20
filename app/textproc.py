import re

TOKEN_RE = re.compile(r"[a-zäöüß0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())
