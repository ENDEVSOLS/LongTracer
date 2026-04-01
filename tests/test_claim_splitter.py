"""
Tests: Claim splitter — sentence splitting and pattern detection.

Covers: split_into_claims, is_meta_statement, is_hallucination_pattern, analyze_claim.
"""

import pytest
from longtracer.guard.claim_splitter import (
    split_into_claims,
    is_meta_statement,
    is_hallucination_pattern,
    analyze_claim,
)


class TestSplitIntoClaims:
    """split_into_claims correctly segments LLM responses."""

    def test_empty_string_returns_empty(self):
        assert split_into_claims("") == []

    def test_whitespace_only_returns_empty(self):
        assert split_into_claims("   \n\t  ") == []

    def test_very_short_text_returns_empty(self):
        # Less than 10 chars
        assert split_into_claims("Hi.") == []

    def test_short_text_under_500_chars_returns_single_claim(self):
        text = "The Eiffel Tower is located in Paris, France."
        result = split_into_claims(text)
        assert len(result) == 1
        assert result[0] == text

    def test_long_text_splits_into_multiple_claims(self):
        # Build a text > 500 chars with clear sentence boundaries
        sentences = [
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris.",
            "It was constructed from 1887 to 1889 as the centerpiece of the 1889 World's Fair.",
            "The tower is 330 metres tall and was the tallest man-made structure in the world for 41 years.",
            "It is named after the engineer Gustave Eiffel, whose company designed and built the tower.",
            "More than 300 million people have visited the Eiffel Tower since it was built.",
            "The tower has three levels for visitors, with restaurants on the first and second levels.",
        ]
        text = " ".join(sentences)
        assert len(text) > 500
        result = split_into_claims(text)
        assert len(result) >= 3  # should split into multiple claims

    def test_decimal_numbers_not_split(self):
        text = "The model achieved 98.5 percent accuracy on the benchmark dataset."
        result = split_into_claims(text)
        # Should be treated as single claim (under 500 chars)
        assert len(result) == 1
        assert "98.5" in result[0]

    def test_abbreviations_not_split(self):
        text = "Dr. Smith published the paper in 2023."
        result = split_into_claims(text)
        assert len(result) == 1
        assert "Dr." in result[0]

    def test_claims_stripped_of_whitespace(self):
        text = "  The sky is blue.  "
        result = split_into_claims(text)
        if result:
            assert result[0] == result[0].strip()

    def test_minimum_claim_length_filter(self):
        # Claims shorter than 15 chars are filtered out
        long_text = " ".join(["A" * 20 + "." for _ in range(30)])
        result = split_into_claims(long_text)
        for claim in result:
            assert len(claim) > 15


class TestIsMetaStatement:
    """is_meta_statement detects honest uncertainty patterns."""

    def test_documents_do_not_contain(self):
        assert is_meta_statement("The provided documents do not contain this information.")

    def test_cannot_find_information(self):
        assert is_meta_statement("I cannot find information about this topic.")

    def test_context_does_not_provide(self):
        assert is_meta_statement("The context does not provide details on this.")

    def test_no_specific_information(self):
        assert is_meta_statement("There is no specific information available.")

    def test_i_could_not_find(self):
        assert is_meta_statement("I could not find any relevant data.")

    def test_not_mentioned_in_the(self):
        assert is_meta_statement("This is not mentioned in the provided documents.")

    def test_case_insensitive(self):
        assert is_meta_statement("THE PROVIDED DOCUMENTS DO NOT CONTAIN this.")

    def test_normal_claim_is_not_meta(self):
        assert not is_meta_statement("The Eiffel Tower is in Paris.")

    def test_empty_string_is_not_meta(self):
        assert not is_meta_statement("")


class TestIsHallucinationPattern:
    """is_hallucination_pattern detects outside-knowledge signals."""

    def test_based_on_my_knowledge(self):
        assert is_hallucination_pattern("Based on my knowledge, this is correct.")

    def test_based_on_general_knowledge(self):
        assert is_hallucination_pattern("Based on general knowledge, the answer is yes.")

    def test_from_my_training(self):
        assert is_hallucination_pattern("From my training, I know this is true.")

    def test_i_know_that(self):
        assert is_hallucination_pattern("I know that the capital of France is Paris.")

    def test_generally_speaking(self):
        assert is_hallucination_pattern("Generally speaking, this approach works well.")

    def test_it_is_well_known(self):
        assert is_hallucination_pattern("It is well known that water boils at 100 degrees.")

    def test_as_we_all_know(self):
        assert is_hallucination_pattern("As we all know, the Earth orbits the Sun.")

    def test_in_my_understanding(self):
        assert is_hallucination_pattern("In my understanding, this is the correct answer.")

    def test_case_insensitive(self):
        assert is_hallucination_pattern("BASED ON MY KNOWLEDGE this is true.")

    def test_normal_claim_not_hallucination_pattern(self):
        assert not is_hallucination_pattern("The Eiffel Tower is 330 metres tall.")

    def test_meta_statement_not_hallucination_pattern(self):
        assert not is_hallucination_pattern("The documents do not contain this information.")


class TestAnalyzeClaim:
    """analyze_claim returns correct classification dict."""

    def test_returns_dict_with_required_keys(self):
        result = analyze_claim("The sky is blue.")
        assert "is_meta_statement" in result
        assert "has_hallucination_pattern" in result

    def test_normal_claim_both_false(self):
        result = analyze_claim("The Eiffel Tower is in Paris.")
        assert result["is_meta_statement"] is False
        assert result["has_hallucination_pattern"] is False

    def test_meta_statement_detected(self):
        result = analyze_claim("The provided documents do not contain this information.")
        assert result["is_meta_statement"] is True
        assert result["has_hallucination_pattern"] is False

    def test_hallucination_pattern_detected(self):
        result = analyze_claim("Based on my knowledge, this is correct.")
        assert result["is_meta_statement"] is False
        assert result["has_hallucination_pattern"] is True
