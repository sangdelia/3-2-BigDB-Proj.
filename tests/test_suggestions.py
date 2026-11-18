from __future__ import annotations

from bigdb_mvp.suggestions import suggest_rewrite


def test_suggest_rewrite_substitutes_longest_overlapping_lexicon_entry() -> None:
    # Given: overlapping deterministic entries inside a Korean harmful sentence.
    text = "너는 정말 멍청이야"

    # When: the sentence is classified as harmful in Korean.
    suggestions = suggest_rewrite(text, language="ko", harmful=True)

    # Then: the longest matching entry is substituted with an exact Unicode span.
    assert len(suggestions) == 1
    suggestion = suggestions[0]
    assert suggestion.strategy == "lexicon_substitution"
    assert suggestion.start == 6
    assert suggestion.end == 10
    assert text[suggestion.start : suggestion.end] == "멍청이야"
    assert suggestion.revised_text == "너는 정말 부적절한 표현이야"


def test_suggest_rewrite_uses_english_template_without_exact_match() -> None:
    # Given: a harmful English sentence with no approved lexicon term.
    text = "I want you gone forever"

    # When: the sentence is classified as harmful in English.
    suggestions = suggest_rewrite(text, language="en", harmful=True)

    # Then: the approved English neutral template is returned without a span.
    assert len(suggestions) == 1
    suggestion = suggestions[0]
    assert suggestion.strategy == "neutral_template"
    assert suggestion.start is None
    assert suggestion.end is None
    assert suggestion.revised_text == "I disagree, but I will express it respectfully."


def test_suggest_rewrite_returns_empty_for_clean_text() -> None:
    # Given: a clean sentence that happens to include an approved lexical term.
    text = "The word idiot appears in a dictionary entry."

    # When: the detector labels it clean.
    suggestions = suggest_rewrite(text, language="en", harmful=False)

    # Then: no rewrite is proposed because suggestions follow the prediction label.
    assert suggestions == ()
