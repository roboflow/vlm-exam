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

import yaml

from vlm_exam.config import (
    ModelConfig,
    PricingConfig,
    RouteConfig,
    _parse_model,
    detection_coordinate_format,
    load_config,
)


class TestParseModel:
    def test_legacy_single_provider_shape(self) -> None:
        model = _parse_model(
            {
                "name": "GPT-5.5",
                "lab": "openai",
                "provider": "openai",
                "provider_model_id": "gpt-5.5",
                "detection_coordinate_format": "normalized_1000",
                "pricing": {
                    "input_per_million_tokens": 5.0,
                    "output_per_million_tokens": 30.0,
                },
            }
        )
        assert model.routes == (RouteConfig("openai", "gpt-5.5"),)
        assert model.provider == "openai"
        assert model.provider_model_id == "gpt-5.5"
        assert model.detection_coordinate_format == "normalized_1000"

    def test_routes_shape(self) -> None:
        model = _parse_model(
            {
                "name": "Gemini 3.1 Pro",
                "lab": "google",
                "detection_coordinate_format": "normalized_1000",
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
    def test_explicit_format_wins_over_provider_default(self) -> None:
        model = ModelConfig(
            name="Gemini 3.1 Pro",
            lab="google",
            routes=(
                RouteConfig("google"),
                RouteConfig("openrouter", "google/gemini-3.1-pro-preview"),
            ),
            pricing=PricingConfig(2.0, 12.0),
            detection_coordinate_format="normalized_1000",
        )
        assert detection_coordinate_format(model) == "normalized_1000"

    def test_openrouter_default_when_unset(self) -> None:
        model = ModelConfig(
            name="Qwen3-VL",
            lab="qwen",
            routes=(RouteConfig("openrouter", "qwen/qwen3-vl-235b-a22b-instruct"),),
            pricing=PricingConfig(0.2, 0.88),
        )
        assert detection_coordinate_format(model) == "normalized_1000_xyxy"

    def test_anthropic_default_when_unset(self) -> None:
        model = ModelConfig(
            name="Claude Opus",
            lab="anthropic",
            routes=(RouteConfig("anthropic"),),
            pricing=PricingConfig(5.0, 25.0),
        )
        assert detection_coordinate_format(model) == "pixel"

    def test_google_default_when_unset(self) -> None:
        model = ModelConfig(
            name="Gemini Flash",
            lab="google",
            routes=(RouteConfig("google"),),
            pricing=PricingConfig(0.5, 3.0),
        )
        assert detection_coordinate_format(model) == "normalized_1000"


class TestLoadBundledConfig:
    def test_gemini_has_fallback_route_and_format(self) -> None:
        config = load_config()
        model = config.models["gemini-3.1-pro-preview"]
        assert model.detection_coordinate_format == "normalized_1000"
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
                "detection_coordinate_format": "normalized_1000",
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
