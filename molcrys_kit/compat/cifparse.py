"""CIF reader that returns the data-block dictionary the CIF pipeline expects."""

from __future__ import annotations


class CifParser:
    """Stand-in for ``pymatgen.io.cif.CifParser`` with ``as_dict`` only."""

    def __init__(self, text: str):
        self._blocks = _parse_blocks(text)

    @classmethod
    def from_str(cls, text: str, **kwargs) -> "CifParser":
        del kwargs
        return cls(text)

    def as_dict(self) -> dict:
        return self._blocks


def _parse_blocks(text: str) -> dict:
    tokens = _tokenize(text)
    blocks: dict[str, dict] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.lower().startswith("data_"):
            index += 1
            continue
        name = token
        index += 1
        block: dict = {}
        while index < len(tokens) and not tokens[index].lower().startswith("data_"):
            current = tokens[index]
            if current.lower() == "loop_":
                index += 1
                tags = []
                while index < len(tokens) and tokens[index].startswith("_"):
                    tags.append(tokens[index].lower())
                    index += 1
                if not tags:
                    # Without column tags no value can be consumed, so the
                    # value loop below would never advance.
                    raise ValueError(
                        f"Malformed CIF: loop_ in data block {name!r} has no "
                        "column tags (expected '_tag' names after 'loop_')"
                    )
                columns = [[] for _ in tags]
                while index < len(tokens) and _is_loop_value(tokens[index]):
                    for column in columns:
                        if index >= len(tokens) or not _is_loop_value(tokens[index]):
                            break
                        column.append(tokens[index])
                        index += 1
                for tag, column in zip(tags, columns):
                    block[tag] = column
                continue
            if current.startswith("_"):
                tag = current.lower()
                index += 1
                if index >= len(tokens):
                    break
                block[tag] = tokens[index]
                index += 1
                continue
            index += 1
        blocks[name] = block
    return blocks


def _is_loop_value(token: str) -> bool:
    lowered = token.lower()
    return not (
        lowered.startswith("data_")
        or lowered == "loop_"
        or token.startswith("_")
        or lowered in {"save_", "global_"}
    )


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    index = 0
    length = len(text)
    line_start = True
    while index < length:
        char = text[index]
        if char in " \t\r":
            index += 1
            continue
        if char == "\n":
            index += 1
            line_start = True
            continue
        if char == "#" and (line_start or text[index - 1] in " \t"):
            while index < length and text[index] != "\n":
                index += 1
            continue
        if char == ";" and line_start:
            index += 1
            end = text.find("\n;", index)
            if end < 0:
                tokens.append(text[index:])
                break
            tokens.append(text[index:end])
            index = end + 2
            line_start = True
            continue
        if char in {"'", '"'}:
            quote = char
            index += 1
            start = index
            pieces = []
            while index < length:
                if text[index] == quote:
                    if quote == "'" and index + 1 < length and text[index + 1] == "'":
                        pieces.append(text[start:index])
                        pieces.append("'")
                        index += 2
                        start = index
                        continue
                    pieces.append(text[start:index])
                    index += 1
                    break
                index += 1
            else:
                pieces.append(text[start:])
            tokens.append("".join(pieces))
            line_start = False
            continue
        start = index
        while index < length and text[index] not in " \t\r\n":
            index += 1
        tokens.append(text[start:index])
        line_start = False
    return tokens
