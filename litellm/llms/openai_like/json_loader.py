"""
JSON-based provider configuration loader for OpenAI-compatible providers.
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

from pydantic import ConfigDict, TypeAdapter, with_config
from typing_extensions import NotRequired, ReadOnly, Required, TypedDict

from litellm._logging import verbose_logger


@with_config(ConfigDict(strict=True))
class _SimpleProviderConfigData(TypedDict, total=False, extra_items=object):
    base_url: ReadOnly[Required[str]]
    api_key_env: ReadOnly[Required[str]]
    api_base_env: ReadOnly[NotRequired[str | None]]
    base_class: ReadOnly[NotRequired[str | None]]
    param_mappings: ReadOnly[NotRequired[Mapping[str, str]]]
    constraints: ReadOnly[NotRequired[Mapping[str, int | float | bool]]]
    special_handling: ReadOnly[NotRequired[Mapping[str, bool]]]
    supported_endpoints: ReadOnly[NotRequired[Sequence[str]]]


_PROVIDERS_ADAPTER: Final = TypeAdapter(
    dict[str, _SimpleProviderConfigData],
    config=ConfigDict(strict=True),
)
_SIMPLE_PROVIDER_CONFIG_ADAPTER: Final = TypeAdapter(_SimpleProviderConfigData)


class SimpleProviderConfig:
    """Simple data class for JSON provider config"""

    def __init__(self, slug: str, data: Mapping[str, object]):
        validated_data: Final = _SIMPLE_PROVIDER_CONFIG_ADAPTER.validate_python(dict(data))
        self.slug: Final = slug
        self.base_url: Final = validated_data["base_url"]
        self.api_key_env: Final = validated_data["api_key_env"]
        self.api_base_env: Final = validated_data.get("api_base_env")
        self.base_class: Final = validated_data.get("base_class", "openai_gpt")
        self.param_mappings: Final = validated_data.get("param_mappings", {})
        self.constraints: Final = validated_data.get("constraints", {})
        self.special_handling: Final = validated_data.get("special_handling", {})
        self.supported_endpoints: Final = validated_data.get("supported_endpoints", [])


class JSONProviderRegistry:
    """Load providers from JSON once on import"""

    _providers: dict[str, SimpleProviderConfig] = {}
    _loaded = False

    @classmethod
    def load(cls):
        """Load providers from JSON configuration file"""
        if cls._loaded:
            return

        json_path: Final = Path(__file__).parent / "providers.json"

        if not json_path.exists():
            # No JSON file yet, that's okay
            cls._loaded = True
            return

        try:
            with open(json_path) as f:
                data: Final = _PROVIDERS_ADAPTER.validate_python(json.load(f))

            for slug, config in data.items():
                cls._providers[slug] = SimpleProviderConfig(slug, config)

            cls._loaded = True
        except Exception as e:
            verbose_logger.warning("Warning: Failed to load JSON provider configs: %s", e)
            cls._loaded = True

    @classmethod
    def get(cls, slug: str) -> SimpleProviderConfig | None:
        """Get a provider configuration by slug"""
        return cls._providers.get(slug)

    @classmethod
    def exists(cls, slug: str) -> bool:
        """Check if a provider is defined via JSON"""
        return slug in cls._providers

    @classmethod
    def get_by_base_url(cls, base_url: str) -> SimpleProviderConfig | None:
        """Get a provider configuration by its default base url"""
        return next((provider for provider in cls._providers.values() if provider.base_url == base_url), None)

    @classmethod
    def supports_responses_api(cls, slug: str) -> bool:
        """Check if a JSON provider supports the Responses API"""
        provider: Final = cls._providers.get(slug)
        if provider is None:
            return False
        return "/v1/responses" in provider.supported_endpoints

    @classmethod
    def list_providers(cls) -> Sequence[str]:
        """List all registered provider slugs"""
        return list(cls._providers.keys())


# Load on import
JSONProviderRegistry.load()
