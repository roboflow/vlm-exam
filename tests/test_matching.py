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

import pytest

from vlm_exam.tasks.vqa import answers_match, normalize_answer


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


class TestAnswersMatch:
    def test_exact_match(self) -> None:
        assert answers_match("red", "red") is True

    def test_case_insensitive_match(self) -> None:
        assert answers_match("Red", "red") is True

    def test_article_insensitive_match(self) -> None:
        assert answers_match("the car", "car") is True

    def test_space_insensitive_match(self) -> None:
        assert answers_match("new york", "newyork") is True

    def test_expected_contains_predicted(self) -> None:
        assert answers_match("red car", "red") is True

    def test_predicted_contains_expected(self) -> None:
        assert answers_match("red", "bright red") is True

    def test_markdown_in_prediction(self) -> None:
        assert answers_match("hello", "**hello**") is True

    def test_no_match(self) -> None:
        assert answers_match("red", "blue") is False

    def test_partial_overlap_no_match(self) -> None:
        assert answers_match("cat", "caterpillar") is True

    def test_empty_strings_match(self) -> None:
        assert answers_match("", "") is True

    @pytest.mark.parametrize(
        ("expected", "predicted"),
        [
            ("42", "42"),
            ("3.14", "3.14"),
            ("yes", "Yes"),
            ("no", "No"),
            ("Toyota", "**Toyota**"),
        ],
    )
    def test_common_vqa_answers(
        self, expected: str, predicted: str
    ) -> None:
        assert answers_match(expected, predicted) is True
