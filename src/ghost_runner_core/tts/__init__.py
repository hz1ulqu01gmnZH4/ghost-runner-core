"""Streaming text-to-speech clients (one per supported engine)."""

from .client import TtsClient, TtsError
from .irodori import IrodoriClient

__all__ = ["IrodoriClient", "TtsClient", "TtsError"]
