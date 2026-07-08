# Copyright 2026 Roboflow, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import MagicMock

import pytest

from vlm_exam.tasks.qa import (
    answers_match,
    normalize_answer,
    parse_count,
    strict_match,
    transcription_similarity,
)


class TestNormalizeAnswer:
    def test_strips_bold_markdown(self) -> None:
        assert normalize_answer("**bold**") == "bold"

    def test_strips_italic_markdown(self) -> None:
        assert normalize_answer("*italic*") == "italic"

    def test_strips_code_markdown(self) -> None:
        assert normalize_answer("`code`") == "code"

    def test_strips_leading_article_the(self) -> None:
        assert normalize_answer("the answer") == "answer"

    def test_strips_leading_article_a(self) -> None:
        assert normalize_answer("a dog") == "dog"

    def test_strips_leading_article_an(self) -> None:
        assert normalize_answer("an apple") == "apple"

    def test_lowercases(self) -> None:
        assert normalize_answer("Hello World") == "hello world"

    def test_collapses_whitespace(self) -> None:
        assert normalize_answer("too   many   spaces") == "too many spaces"

    def test_strips_surrounding_whitespace(self) -> None:
        assert normalize_answer("  padded  ") == "padded"

    def test_combined_normalization(self) -> None:
        assert normalize_answer("  **The**  answer  ") == "answer"


class TestStrictMatch:
    def test_exact_match(self) -> None:
        assert strict_match("red", "red") is True

    def test_case_insensitive_match(self) -> None:
        assert strict_match("Red", "red") is True

    def test_article_insensitive_match(self) -> None:
        assert strict_match("the car", "car") is True

    def test_space_insensitive_match(self) -> None:
        assert strict_match("new york", "newyork") is True

    def test_markdown_in_prediction(self) -> None:
        assert strict_match("hello", "**hello**") is True

    def test_no_match(self) -> None:
        assert strict_match("red", "blue") is False

    def test_partial_overlap_no_match(self) -> None:
        assert strict_match("cat", "caterpillar") is False

    def test_substring_digit_no_match(self) -> None:
        assert strict_match("18", "8") is False

    def test_substring_prefix_no_match(self) -> None:
        assert strict_match("G230", "230") is False

    def test_truncated_no_match(self) -> None:
        assert strict_match("2 000111 111112", "2 0001") is False

    def test_empty_strings_match(self) -> None:
        assert strict_match("", "") is True

    @pytest.mark.parametrize(
        ("expected", "predicted"),
        [
            ("42", "42"),
            ("3.14", "3.14"),
            ("yes", "Yes"),
            ("no", "No"),
            ("Toyota", "**Toyota**"),
            ("205/60 R 16", "205/60 R16"),
            ("2 000111 111112", "2000111111112"),
        ],
    )
    def test_common_vqa_answers(self, expected: str, predicted: str) -> None:
        assert strict_match(expected, predicted) is True


class TestAnswersMatchWithJudge:
    def test_strict_mode_no_substring(self) -> None:
        correct, method = answers_match("18", "8", match_mode="strict")
        assert correct is False
        assert method == "strict"

    def test_strict_mode_exact(self) -> None:
        correct, method = answers_match("red", "red", match_mode="strict")
        assert correct is True
        assert method == "strict"

    def test_judge_mode_calls_judge_on_mismatch(self) -> None:
        mock_judge = MagicMock()
        mock_judge.evaluate.return_value = True

        correct, method = answers_match(
            "checkered flag",
            "A checkered racing flag",
            question="What is the logo?",
            match_mode="judge",
            judge=mock_judge,
        )

        assert correct is True
        assert method == "judge"
        mock_judge.evaluate.assert_called_once_with(
            question="What is the logo?",
            expected="checkered flag",
            predicted="A checkered racing flag",
            guidance="",
        )

    def test_judge_mode_skips_judge_on_exact_match(self) -> None:
        mock_judge = MagicMock()

        correct, method = answers_match(
            "red",
            "RED",
            question="What color?",
            match_mode="judge",
            judge=mock_judge,
        )

        assert correct is True
        assert method == "strict"
        mock_judge.evaluate.assert_not_called()

    def test_judge_mode_returns_false_when_judge_rejects(self) -> None:
        mock_judge = MagicMock()
        mock_judge.evaluate.return_value = False

        correct, method = answers_match(
            "18",
            "8",
            question="How many items?",
            match_mode="judge",
            judge=mock_judge,
        )

        assert correct is False
        assert method == "judge"

    def test_judge_mode_without_judge_falls_back_to_strict(self) -> None:
        correct, method = answers_match(
            "phone",
            "smartphone",
            question="What device?",
            match_mode="judge",
            judge=None,
        )

        assert correct is False
        assert method == "strict"


class TestParseCount:
    def test_plain_digit(self) -> None:
        assert parse_count("4") == 4

    def test_digit_with_markdown(self) -> None:
        assert parse_count("**12**") == 12

    def test_spelled_out_unit(self) -> None:
        assert parse_count("six") == 6

    def test_spelled_out_teen(self) -> None:
        assert parse_count("fifteen") == 15

    def test_spelled_out_tens(self) -> None:
        assert parse_count("forty") == 40

    def test_spelled_out_compound(self) -> None:
        assert parse_count("twenty-one") == 21

    def test_uppercase_word(self) -> None:
        assert parse_count("Six") == 6

    def test_single_embedded_integer(self) -> None:
        assert parse_count("There are 4 bars") == 4

    def test_single_embedded_word(self) -> None:
        assert parse_count("I count six icons") == 6

    def test_multiple_integers_ambiguous(self) -> None:
        assert parse_count("between 4 and 5") is None

    def test_no_number(self) -> None:
        assert parse_count("many") is None

    def test_empty_string(self) -> None:
        assert parse_count("") is None


class TestTranscriptionSimilarity:
    def test_identical_text(self) -> None:
        assert transcription_similarity("hello world", "hello world") == 1.0

    def test_code_fence_stripped(self) -> None:
        assert (
            transcription_similarity("hello world", "```markdown\nhello world\n```")
            == 1.0
        )

    def test_trailing_whitespace_ignored(self) -> None:
        assert (
            transcription_similarity("line one\nline two", "line one  \nline two ")
            == 1.0
        )

    def test_case_matters(self) -> None:
        assert transcription_similarity("ABC", "abc") < 1.0

    def test_completely_different(self) -> None:
        assert transcription_similarity("aaaa", "zzzz") == 0.0

    def test_partial_match_between_zero_and_one(self) -> None:
        score = transcription_similarity("2COOL4U", "2COOL4V")
        assert 0.5 < score < 1.0

    def test_both_empty(self) -> None:
        assert transcription_similarity("", "") == 1.0
