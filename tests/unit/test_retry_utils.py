import pytest
from app.core.retry_utils import retry_api_call, APIRateLimitError, APITimeoutError
from unittest.mock import MagicMock

def test_retry_logic_success():
    """Verify that successful calls return immediately."""
    mock_fn = MagicMock(return_value="success")
    decorator = retry_api_call(max_attempts=3)
    decorated_fn = decorator(mock_fn)
    
    result = decorated_fn()
    assert result == "success"
    assert mock_fn.call_count == 1

def test_retry_logic_eventual_success():
    """Verify that it retries on allowed exceptions and eventually succeeds."""
    mock_fn = MagicMock()
    mock_fn.side_effect = [APIRateLimitError("rate"), APIRateLimitError("rate"), "success"]
    
    decorator = retry_api_call(max_attempts=5, min_wait=0.01, max_wait=0.05)
    decorated_fn = decorator(mock_fn)
    
    result = decorated_fn()
    assert result == "success"
    assert mock_fn.call_count == 3

def test_retry_logic_failure():
    """Verify that it reraises after max attempts."""
    mock_fn = MagicMock(side_effect=APITimeoutError("timeout"))
    
    decorator = retry_api_call(max_attempts=3, min_wait=0.01, max_wait=0.05)
    decorated_fn = decorator(mock_fn)
    
    with pytest.raises(APITimeoutError):
        decorated_fn()
    
    assert mock_fn.call_count == 3

def test_retry_logic_non_retryable():
    """Verify that it doesn't retry on non-retryable exceptions."""
    mock_fn = MagicMock(side_effect=ValueError("bad data"))
    
    decorator = retry_api_call(max_attempts=3)
    decorated_fn = decorator(mock_fn)
    
    with pytest.raises(ValueError):
        decorated_fn()
    
    assert mock_fn.call_count == 1
