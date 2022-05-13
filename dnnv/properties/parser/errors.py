class ParserError(Exception):
    def __init__(self, msg: str, *args: object, lineno=None, col_offset=None) -> None:
        if lineno is not None:
            prefix = f"line {lineno}"
            if col_offset is not None:
                prefix = f"{prefix}, column {col_offset}"
            msg = f"{prefix}: {msg}"
        super().__init__(msg, *args)


__all__ = ["ParserError"]
