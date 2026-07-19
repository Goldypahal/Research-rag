"""
connectivity.py
---------------
Checks real internet connectivity and manages the active runtime mode
(local vs cloud) so the system can switch without a server restart.
"""
from __future__ import annotations
import socket
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class LLMMode(str, Enum):
    LOCAL = "local"
    CLOUD = "cloud"


def check_internet(host: str = "8.8.8.8", port: int = 53, timeout: float = 2.0) -> bool:
    """
    Probe Google's public DNS (8.8.8.8:53) with a raw TCP socket.
    This is fast (~2ms when connected), requires no HTTP library,
    and doesn't send any user data over the network.
    """
    try:
        socket.setdefaulttimeout(timeout)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((host, port))
        return True
    except OSError:
        return False


class ModeManager:
    """
    Singleton that holds the current LLM mode at runtime.
    Initialized on startup based on connectivity and key availability;
    automatically falls back to local.
    """
    _instance: "ModeManager | None" = None

    def __new__(cls) -> "ModeManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._mode = LLMMode.LOCAL
            cls._instance._internet_available = False
        return cls._instance

    def detect_and_set(self) -> None:
        """
        Called once on server startup.
        Detects connectivity and sets the initial mode automatically.
        """
        from .settings import settings
        self._internet_available = check_internet()
        has_keys = bool(settings.GOOGLE_API_KEY and settings.COHERE_API_KEY)

        if self._internet_available and has_keys:
            self._mode = LLMMode.CLOUD
            logger.info("Connectivity: Internet detected and Cloud API keys found. Configured to CLOUD mode.")
        else:
            self._mode = LLMMode.LOCAL
            logger.info(f"Connectivity: Configured to LOCAL mode. (Internet: {self._internet_available}, Keys present: {has_keys})")

    @property
    def mode(self) -> LLMMode:
        from .settings import settings
        has_keys = bool(settings.GOOGLE_API_KEY and settings.COHERE_API_KEY)
        if self._internet_available and has_keys:
            return LLMMode.CLOUD
        return LLMMode.LOCAL

    @property
    def internet_available(self) -> bool:
        return self._internet_available

    def set_mode(self, mode: LLMMode) -> bool:
        """
        Mode is managed automatically. Manual override prints a warning but returns True.
        """
        logger.info("set_mode: Mode changes are managed automatically based on connectivity and configured API keys.")
        return True

    def refresh_connectivity(self) -> bool:
        """Re-check internet connectivity on demand."""
        self._internet_available = check_internet()
        return self._internet_available


# Global singleton
mode_manager = ModeManager()
