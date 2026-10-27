"""UNC's OpenAI-compatible Responses configuration and client factories."""

import os
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from openai import AsyncOpenAI, OpenAI

UNC_ENDPOINT = "https://azureaiapi.cloud.unc.edu"


def _env_value(primary: str, legacy: str, default: str = "") -> str:
    """Use Azure aliases first, matching LearnWithAI's configuration precedence."""
    return os.environ.get(primary, os.environ.get(legacy, default)).strip()


def deployment_name(model: str) -> str:
    """Map Harbor's optional openai/ prefix to a UNC deployment name."""
    deployment = model.strip().removeprefix("openai/")
    if not deployment or "/" in deployment or any(char.isspace() for char in deployment):
        raise ValueError("Use a UNC deployment name, optionally prefixed with openai/ for Harbor.")
    return deployment


def _base_url(endpoint: str) -> str:
    """Require an HTTPS origin and prevent direct public OpenAI API access."""
    try:
        url = urlsplit(endpoint)
        host = (url.hostname or "").lower().rstrip(".")
        port = url.port
    except ValueError:
        raise ValueError("OPENAI_ENDPOINT must be a valid HTTPS endpoint origin.") from None
    if (
        url.scheme != "https"
        or not host
        or url.username is not None
        or url.password is not None
        or port not in (None, 443)
        or url.path.strip("/")
        or url.query
        or url.fragment
        or host == "openai.com"
        or host.endswith(".openai.com")
    ):
        raise ValueError(
            "Set OPENAI_ENDPOINT or AZURE_OPENAI_ENDPOINT to a UNC/Azure HTTPS origin "
            "without a path, credentials, query, or fragment; public OpenAI endpoints are disabled."
        )
    return f"{endpoint.rstrip('/')}/openai/v1/"


@dataclass(frozen=True)
class UNCOpenAISettings:
    """Keep subscription keys out of repr and require an explicit deployment."""

    api_key: str = field(repr=False)
    model: str
    endpoint: str = UNC_ENDPOINT

    @classmethod
    def from_env(cls, *, model: str | None = None) -> "UNCOpenAISettings":
        """Prefer the run's model, then Azure's deployment alias, then OPENAI_MODEL.

        Only None selects the environment default; a blank override is an error.
        uv --env-file loads local configuration before this method is called.
        """
        api_key = _env_value("AZURE_OPENAI_API_KEY", "OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "Set OPENAI_API_KEY or AZURE_OPENAI_API_KEY to a UNC subscription key."
            )
        selected_model = (
            model if model is not None else _env_value("AZURE_OPENAI_DEPLOYMENT", "OPENAI_MODEL")
        )
        if not selected_model.strip():
            raise ValueError(
                "Set OPENAI_MODEL or AZURE_OPENAI_DEPLOYMENT to a UNC deployment name."
            )
        endpoint = _env_value("AZURE_OPENAI_ENDPOINT", "OPENAI_ENDPOINT", UNC_ENDPOINT)
        _base_url(endpoint)
        return cls(api_key=api_key, model=deployment_name(selected_model), endpoint=endpoint)

    def create_client(self) -> OpenAI:
        """Use the same v1 route and bearer authentication as LearnWithAI chat."""
        return OpenAI(
            api_key=self.api_key,
            base_url=_base_url(self.endpoint),
            default_headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=30.0,
            max_retries=0,
        )

    def create_async_client(self, *, timeout: float = 30.0) -> AsyncOpenAI:
        """Provide the same routing with an explicit, per-client timeout override."""
        return AsyncOpenAI(
            api_key=self.api_key,
            base_url=_base_url(self.endpoint),
            default_headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=timeout,
            max_retries=0,
        )
