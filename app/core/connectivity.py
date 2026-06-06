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
    Initialized on startup based on connectivity; can be changed
    at any time via the /api/v1/mode endpoint.
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

        if not self._internet_available:
            self._mode = LLMMode.LOCAL
            logger.info("Connectivity: No internet detected. Forced to LOCAL mode.")
        elif settings.USE_LOCAL_LLM:
            # Internet is available but user's default preference is local
            self._mode = LLMMode.LOCAL
            logger.info("Connectivity: Internet available. Starting in LOCAL mode (user default).")
        else:
            self._mode = LLMMode.CLOUD
            logger.info("Connectivity: Internet available. Starting in CLOUD mode.")

    @property
    def mode(self) -> LLMMode:
        return self._mode

    @property
    def internet_available(self) -> bool:
        return self._internet_available

    def set_mode(self, mode: LLMMode) -> bool:
        """
        Switch mode. Returns False if cloud was requested but no internet.
        """
        if mode == LLMMode.CLOUD and not self._internet_available:
            logger.warning("Cannot switch to CLOUD: no internet connection.")
            return False
        self._mode = mode
        logger.info(f"Mode switched to: {mode.value.upper()}")
        return True

    def refresh_connectivity(self) -> bool:
        """Re-check internet connectivity on demand."""
        self._internet_available = check_internet()
        if not self._internet_available and self._mode == LLMMode.CLOUD:
            logger.warning("Internet lost! Auto-switching back to LOCAL mode.")
            self._mode = LLMMode.LOCAL
        return self._internet_available


# Global singleton
mode_manager = ModeManager()
