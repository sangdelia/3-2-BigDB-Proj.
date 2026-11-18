from __future__ import annotations

import pytest

from bigdb_mvp.experiments.hypothesis_cases import normalize


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("시.발", "시발"),
        ("씨@발", "씨발"),
        ("병  신", "병신"),
        ("지1랄", "지랄"),
        ("ㅅㅣ발", "시발"),
        ("시발", "시발"),
        ("18번 버스", "18번 버스"),
        ("씨앗 발아율", "씨앗 발아율"),
        ("병원 신관", "병원 신관"),
        ("계산 결과는 1.5입니다.", "계산 결과는 1.5입니다."),
        ("'시@발'이라고 말하면 안 됩니다.", "'시발'이라고 말하면 안 됩니다."),
    ],
)
def test_bounded_normalization_preserves_surrounding_meaning(
    text: str,
    expected: str,
) -> None:
    result = normalize(text)
    assert result == expected
    assert normalize(result) == result
