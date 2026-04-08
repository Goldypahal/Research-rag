from __future__ import annotations
import logging
from typing import Iterable, Tuple, Type
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

class RetryableAPIError(Exception):
    """Base class for temporary API failures."""

class APITimeoutError(RetryableAPIError):
    pass

class APIRateLimitError(RetryableAPIError):
    pass

class APIServerError(RetryableAPIError):
    pass

def retry_api_call(max_attempts: int = 5, min_wait: int = 1, max_wait: int = 20):
    return retry(
        retry=retry_if_exception_type(
            (APITimeoutError, APIRateLimitError, APIServerError)
        ),
        wait=wait_random_exponential(multiplier=min_wait, max=max_wait),
        stop=stop_after_attempt(max_attempts),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
