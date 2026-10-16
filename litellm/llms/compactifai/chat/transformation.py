"""
CompactifAI chat completion transformation
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final

import httpx
from pydantic import ConfigDict, TypeAdapter, with_config
from typing_extensions import ReadOnly, Required, TypedDict

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.openai.common_utils import OpenAIError
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse

from ...openai.chat.gpt_transformation import OpenAIGPTConfig

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj
    from litellm.litellm_core_utils.tokenizer import Encoding as Tokenizer

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


@with_config(ConfigDict(extra="allow", strict=True))
class _CompactifAIMessage(TypedDict, total=False):
    content: ReadOnly[object]
    tool_calls: ReadOnly[Sequence[object] | None]


@with_config(ConfigDict(extra="allow", strict=True))
class _CompactifAIChoice(TypedDict, total=False):
    message: ReadOnly[_CompactifAIMessage | None]


@with_config(ConfigDict(extra="allow", strict=True))
class _CompactifAIResponseFields(TypedDict, total=False):
    choices: Required[ReadOnly[Sequence[_CompactifAIChoice]]]


@with_config(ConfigDict(extra="allow", strict=True))
class _CompactifAIBody(TypedDict, total=False):
    pass


@with_config(ConfigDict(extra="allow", strict=True))
class _CompactifAIFunction(TypedDict, total=False):
    arguments: ReadOnly[object]


@with_config(ConfigDict(extra="allow", strict=True))
class _CompactifAIToolCall(TypedDict):
    function: ReadOnly[_CompactifAIFunction]


_COMPACTIFAI_RESPONSE: Final = TypeAdapter(_CompactifAIResponseFields)
_COMPACTIFAI_BODY: Final = TypeAdapter(_CompactifAIBody)
_TOOL_CALL: Final = TypeAdapter(_CompactifAIToolCall)


def _json_mode_response(response_body: object) -> ModelResponse:
    response_fields: Final = _COMPACTIFAI_RESPONSE.validate_python(response_body)
    choices: Final = [
        {
            **choice,
            "message": {
                **message,
                "content": _TOOL_CALL.validate_python(tool_calls[0])["function"].get("arguments", ""),
                "tool_calls": None,
            },
        }
        if (message := choice.get("message")) and (tool_calls := message.get("tool_calls")) and len(tool_calls) == 1
        else choice
        for choice in response_fields["choices"]
    ]
    return ModelResponse(**{**response_fields, "choices": choices})


class CompactifAIChatConfig(OpenAIGPTConfig):
    """
    Configuration class for CompactifAI chat completions.
    Since CompactifAI is OpenAI-compatible, we extend OpenAIGPTConfig.
    """

    def _get_openai_compatible_provider_info(
        self,
        api_base: str | None,
        api_key: str | None,
    ) -> tuple[str | None, str | None]:
        """
        Get API base and key for CompactifAI provider.
        """
        api_base = api_base or "https://api.compactif.ai/v1"
        dynamic_api_key: Final = api_key or get_secret_str("COMPACTIFAI_API_KEY") or ""
        return api_base, dynamic_api_key

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: Mapping[str, object],
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        encoding: "Tokenizer | None",
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ModelResponse:
        """
        Transform CompactifAI response to LiteLLM format.
        Since CompactifAI is OpenAI-compatible, we can use the standard OpenAI transformation.
        """
        ## LOGGING
        logging_obj.post_call(
            input=messages,
            api_key=api_key,
            original_response=raw_response.text,
            additional_args={"complete_input_dict": request_data},
        )

        ## RESPONSE OBJECT
        # Handle JSON mode if needed
        # Convert tool calls to content for JSON mode
        returned_response: Final = (
            _json_mode_response(raw_response.json())
            if json_mode
            else ModelResponse(**_COMPACTIFAI_BODY.validate_python(raw_response.json()))
        )

        # Set model name with provider prefix
        returned_response.model = f"compactifai/{model}"

        return returned_response

    def get_error_class(
        self, error_message: str, status_code: int, headers: dict[str, str] | httpx.Headers
    ) -> BaseLLMException:
        """
        Get the appropriate error class for CompactifAI errors.
        Since CompactifAI is OpenAI-compatible, we use OpenAI error handling.
        """
        return OpenAIError(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
