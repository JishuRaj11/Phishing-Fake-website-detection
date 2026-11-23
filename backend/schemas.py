import ipaddress
import re
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config import MAX_URL_LENGTH

ALLOWED_SCHEMES = {"http", "https"}
SCHEME_PREFIX = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")


class PredictRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    url: str = Field(..., min_length=1, max_length=MAX_URL_LENGTH, examples=["https://example.com"])

    @field_validator("url")
    @classmethod
    def validate_url_format(cls, value: str) -> str:
        url = value.strip()

        if not url:
            raise ValueError("URL must not be empty.")
        if len(url) > MAX_URL_LENGTH:
            raise ValueError(f"URL must be at most {MAX_URL_LENGTH} characters.")
        if any(ord(char) < 32 or ord(char) == 127 for char in url):
            raise ValueError("URL must not contain control characters.")
        if any(char.isspace() for char in url):
            raise ValueError("URL must not contain whitespace.")

        if SCHEME_PREFIX.match(url):
            scheme = url.split(":", 1)[0].lower()
            if scheme not in ALLOWED_SCHEMES:
                raise ValueError("Only http and https URLs are supported.")

        candidate = url if "://" in url else "http://" + url

        try:
            parts = urlsplit(candidate)
            hostname = parts.hostname
            parts.port
        except ValueError as exc:
            raise ValueError("URL has an invalid format or port.") from exc

        if not hostname:
            raise ValueError("URL must include a hostname.")

        if "." not in hostname:
            try:
                ipaddress.ip_address(hostname)
            except ValueError as exc:
                raise ValueError("URL hostname must be a domain name or IP address.") from exc

        return url


class PredictResponse(BaseModel):
    url: str
    prediction: str
    is_phishing: bool
    confidence: float
    features: dict[str, Any]
