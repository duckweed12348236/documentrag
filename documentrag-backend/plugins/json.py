import orjson

from typing import Any


def load(text: str) -> Any:
    return orjson.loads(text)


def dump(obj: Any) -> str:
    return orjson.dumps(obj).decode()
