from typing import TextIO, TypeAlias

YamlScalar: TypeAlias = str | int | float | bool | None
YamlValue: TypeAlias = YamlScalar | list[YamlValue] | dict[str, YamlValue]

def safe_load(stream: TextIO) -> YamlValue: ...
