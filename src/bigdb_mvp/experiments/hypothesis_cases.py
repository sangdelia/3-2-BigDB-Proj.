"""Authored diagnostic expectations are separate from corpus gold labels."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Final, Literal

SEPARATOR: Final = r"[\s._@*0-9]*"
WORDS: Final = ("시발", "씨발", "병신", "지랄", "개새끼", "꺼져", "닥쳐", "좆같")
PATTERNS: Final = tuple(
    (re.compile(SEPARATOR.join(map(re.escape, word))), word) for word in WORDS
)
CONTEXTS: Final = (
    ("너는 바보야.", "너는 바보가 아니야."),
    ("너는 멍청해.", "너는 멍청하지 않아."),
    ("너는 병신이야.", "너는 병신이 아니야."),
    ("너는 쓸모없는 인간이야.", "너는 쓸모없는 인간이 아니야."),
    ("넌 쓰레기야.", "넌 쓰레기가 아니야."),
    ("넌 개새끼야.", "넌 개새끼가 아니야."),
)
ORDINARY: Final = (
    "안녕하세요.",
    "감사합니다.",
    "오늘 날씨가 좋네요.",
    "내일 오전에 만나요.",
    "씨앗 발아율을 확인해 주세요.",
    "병원 신관에 도착했습니다.",
    "18번 버스가 도착했습니다.",
    "시 1회 발표가 예정돼 있습니다.",
    "강아지가 새끼를 낳았습니다.",
    "그 영화는 별로 재미없었어요.",
    "계산 결과는 1.5입니다.",
    "이메일 주소에 @ 기호가 있어요.",
    "외국인 친구와 함께 한국어를 공부합니다.",
    "모든 사람을 존중해야 합니다.",
    "회의 자료를 확인해 주세요.",
    "종교가 달라도 차별하면 안 됩니다.",
)


@dataclass(frozen=True, slots=True)
class Case:
    """An authored expected class expresses test intent, not consensus annotation."""

    case_id: str
    family: str
    category: str
    text: str
    expected: Literal["clean", "harmful"]


def normalize(text: str) -> str:
    """Apply predeclared bounded replacements without deleting all punctuation."""
    result = (
        unicodedata.normalize("NFC", text).replace("ㅅㅣ", "시").replace("ㅆㅣ", "씨")
    )
    for pattern, word in PATTERNS:
        result = pattern.sub(word, result)
    return result


def cases() -> tuple[Case, ...]:
    """Freeze paired contexts, obfuscations, and ordinary controls before scoring."""
    result: list[Case] = []
    for i, (direct, negative) in enumerate(CONTEXTS):
        family = f"context-{i}"
        result.extend(
            (
                Case(f"{family}-direct", family, "direct", direct, "harmful"),
                Case(f"{family}-negative", family, "negation", negative, "clean"),
                Case(
                    f"{family}-critic",
                    family,
                    "quote_criticism",
                    f"'{direct}'라는 말로 사람을 모욕하면 안 됩니다.",
                    "clean",
                ),
                Case(
                    f"{family}-support",
                    family,
                    "quote_endorsement",
                    f"'{direct}'라는 말이 맞아. 나도 너를 그렇게 생각해.",
                    "harmful",
                ),
            )
        )
    for i, word in enumerate(WORDS[:4]):
        for j, separator in enumerate(("", " ", ".", "@", "1", "_", "*")):
            result.append(
                Case(
                    f"variant-{i}-{j}",
                    f"variant-{i}",
                    "short_variant",
                    separator.join(word),
                    "harmful",
                )
            )
    for i, text in enumerate(("ㅅㅣ발", "ㅆㅣ발", "ㅅㅂ")):
        result.append(Case(f"jamo-{i}", "jamo", "jamo_variant", text, "harmful"))
    result.extend(
        Case(f"ordinary-{i}", f"ordinary-{i}", "ordinary", text, "clean")
        for i, text in enumerate(ORDINARY)
    )
    return tuple(result)
