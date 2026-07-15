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

from pathlib import Path

import pytest
import yaml

from vlm_exam.config import (
    RouteConfig,
    _parse_model,
    load_config,
)
from vlm_exam.tasks.detection import DetectionCoordinateFormat


class TestParseModel:
    def test_legacy_single_provider_shape(self) -> None:
        model = _parse_model(
            {
                "name": "GPT-5.5",
                "lab": "openai",
                "provider": "openai",
                "provider_model_id": "gpt-5.5",
                "detection_coordinate_format": "yxyx_normalized_0_to_1000",
                "pricing": {
                    "input_per_million_tokens": 5.0,
                    "output_per_million_tokens": 30.0,
                },
            }
        )
        assert model.routes == (RouteConfig("openai", "gpt-5.5"),)
        assert model.provider == "openai"
        assert model.provider_model_id == "gpt-5.5"
        assert model.detection_coordinate_format == "yxyx_normalized_0_to_1000"

    def test_routes_shape(self) -> None:
        model = _parse_model(
            {
                "name": "Gemini 3.1 Pro",
                "lab": "google",
                "detection_coordinate_format": "yxyx_normalized_0_to_1000",
                "routes": [
                    {"provider": "google"},
                    {
                        "provider": "openrouter",
                        "provider_model_id": "google/gemini-3.1-pro-preview",
                    },
                ],
                "pricing": {
                    "input_per_million_tokens": 2.0,
                    "output_per_million_tokens": 12.0,
                },
            }
        )
        assert len(model.routes) == 2
        assert model.routes[0] == RouteConfig("google", None)
        assert model.routes[1] == RouteConfig(
            "openrouter", "google/gemini-3.1-pro-preview"
        )
        assert model.provider == "google"


class TestDetectionCoordinateFormat:
    def test_parse_model_coerces_string_to_enum(self) -> None:
        model = _parse_model(
            {
                "name": "Qwen3-VL",
                "lab": "qwen",
                "provider": "openrouter",
                "provider_model_id": "qwen/qwen3-vl-235b-a22b-instruct",
                "detection_coordinate_format": "xyxy_normalized_0_to_1000",
                "pricing": {
                    "input_per_million_tokens": 0.2,
                    "output_per_million_tokens": 0.88,
                },
            }
        )
        assert (
            model.detection_coordinate_format
            == DetectionCoordinateFormat.XYXY_NORMALIZED_0_TO_1000
        )

    def test_parse_model_requires_format(self) -> None:
        with pytest.raises(KeyError):
            _parse_model(
                {
                    "name": "Qwen3-VL",
                    "lab": "qwen",
                    "provider": "openrouter",
                    "pricing": {
                        "input_per_million_tokens": 0.2,
                        "output_per_million_tokens": 0.88,
                    },
                }
            )

    def test_parse_model_rejects_unknown_format(self) -> None:
        with pytest.raises(ValueError):
            _parse_model(
                {
                    "name": "Qwen3-VL",
                    "lab": "qwen",
                    "provider": "openrouter",
                    "detection_coordinate_format": "bogus",
                    "pricing": {
                        "input_per_million_tokens": 0.2,
                        "output_per_million_tokens": 0.88,
                    },
                }
            )


class TestResolutionTier:
    def test_defaults_to_high(self) -> None:
        model = _parse_model(
            {
                "name": "Claude Sonnet 5",
                "lab": "anthropic",
                "provider": "anthropic",
                "detection_coordinate_format": "xyxy_absolute_provider_upload",
                "pricing": {
                    "input_per_million_tokens": 2.0,
                    "output_per_million_tokens": 10.0,
                },
            }
        )
        assert model.resolution_tier == "high"

    def test_reads_explicit_tier(self) -> None:
        model = _parse_model(
            {
                "name": "Claude Haiku",
                "lab": "anthropic",
                "provider": "anthropic",
                "detection_coordinate_format": "xyxy_absolute_provider_upload",
                "resolution_tier": "standard",
                "pricing": {
                    "input_per_million_tokens": 1.0,
                    "output_per_million_tokens": 5.0,
                },
            }
        )
        assert model.resolution_tier == "standard"

    def test_rejects_unknown_tier_at_parse_time(self) -> None:
        with pytest.raises(ValueError, match="resolution_tier"):
            _parse_model(
                {
                    "name": "Claude Typo",
                    "lab": "anthropic",
                    "provider": "anthropic",
                    "detection_coordinate_format": "xyxy_absolute_provider_upload",
                    "resolution_tier": "standrad",
                    "pricing": {
                        "input_per_million_tokens": 1.0,
                        "output_per_million_tokens": 5.0,
                    },
                }
            )


class TestProviderUploadRouteGuard:
    def test_rejects_provider_upload_on_non_anthropic_route(self) -> None:
        with pytest.raises(ValueError, match="pre-resize"):
            _parse_model(
                {
                    "name": "Bad Model",
                    "lab": "openai",
                    "provider": "openai",
                    "detection_coordinate_format": "xyxy_absolute_provider_upload",
                    "pricing": {
                        "input_per_million_tokens": 1.0,
                        "output_per_million_tokens": 2.0,
                    },
                }
            )

    def test_rejects_provider_upload_with_openrouter_fallback(self) -> None:
        with pytest.raises(ValueError, match="incompatible routes"):
            _parse_model(
                {
                    "name": "Claude With Fallback",
                    "lab": "anthropic",
                    "detection_coordinate_format": "xyxy_absolute_provider_upload",
                    "routes": [
                        {"provider": "anthropic"},
                        {"provider": "openrouter", "provider_model_id": "x/y"},
                    ],
                    "pricing": {
                        "input_per_million_tokens": 1.0,
                        "output_per_million_tokens": 2.0,
                    },
                }
            )

    def test_allows_provider_upload_on_anthropic(self) -> None:
        model = _parse_model(
            {
                "name": "Claude Opus",
                "lab": "anthropic",
                "provider": "anthropic",
                "detection_coordinate_format": "xyxy_absolute_provider_upload",
                "pricing": {
                    "input_per_million_tokens": 5.0,
                    "output_per_million_tokens": 25.0,
                },
            }
        )
        assert model.provider == "anthropic"


class TestLoadBundledConfig:
    def test_gemini_has_fallback_route_and_format(self) -> None:
        config = load_config()
        model = config.models["gemini-3.1-pro-preview"]
        assert model.detection_coordinate_format == "yxyx_normalized_0_to_1000"
        assert len(model.routes) == 2
        assert model.routes[1].provider == "openrouter"

    def test_all_models_parse(self, tmp_path: Path) -> None:
        bundled = (
            Path(__file__).resolve().parents[1] / "src/vlm_exam/configs/models.yaml"
        )
        config = load_config(bundled)
        assert len(config.models) >= 10


def test_write_and_reload_round_trip(tmp_path: Path) -> None:
    raw = {
        "labs": {
            "google": {
                "name": "Google",
                "color": "#000",
                "logo_url": "https://example.com/logo.svg",
            }
        },
        "models": {
            "gemini-test": {
                "name": "Gemini Test",
                "lab": "google",
                "routes": [{"provider": "google"}],
                "detection_coordinate_format": "yxyx_normalized_0_to_1000",
                "pricing": {
                    "input_per_million_tokens": 1.0,
                    "output_per_million_tokens": 2.0,
                },
            }
        },
    }
    config_path = tmp_path / "models.yaml"
    config_path.write_text(yaml.dump(raw))
    config = load_config(config_path)
    assert "gemini-test" in config.models
