import time
from collections import defaultdict
from typing import Dict
from backend.src import config
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    In-memory rate limiter with configurable limits
    In production, use Redis or similar for distributed rate limiting
    """

    # Dictionary to store request timestamps for each session
    requests: Dict[str, list] = defaultdict(list)

    @classmethod
    def is_allowed(cls, identifier: str, max_requests: int = None, time_window: int = None) -> bool:
        """
        Check if an identifier is allowed to make a request based on rate limits

        :param identifier: The identifier to check (could be session_id, IP, etc.)
        :param max_requests: Maximum number of requests allowed (uses config default if None)
        :param time_window: Time window in seconds (uses config default if None)
        :return: True if request is allowed, False otherwise
        """
        # Use config values if not provided
        max_requests = max_requests or config.settings.RATE_LIMIT_REQUESTS
        time_window = time_window or config.settings.RATE_LIMIT_WINDOW

        current_time = time.time()

        # Clean old requests outside the time window
        cls.requests[identifier] = [
            req_time for req_time in cls.requests[identifier]
            if current_time - req_time < time_window
        ]

        # Check if request limit is exceeded
        if len(cls.requests[identifier]) >= max_requests:
            logger.warning(f"Rate limit exceeded for identifier: {identifier}")
            return False

        # Add current request timestamp
        cls.requests[identifier].append(current_time)
        return True

    @classmethod
    def get_remaining_requests(cls, identifier: str, max_requests: int = None, time_window: int = None) -> int:
        """
        Get the number of remaining requests for an identifier

        :param identifier: The identifier to check
        :param max_requests: Maximum number of requests allowed (uses config default if None)
        :param time_window: Time window in seconds (uses config default if None)
        :return: Number of remaining requests
        """
        max_requests = max_requests or config.settings.RATE_LIMIT_REQUESTS
        time_window = time_window or config.settings.RATE_LIMIT_WINDOW

        current_time = time.time()

        # Clean old requests outside the time window
        cls.requests[identifier] = [
            req_time for req_time in cls.requests[identifier]
            if current_time - req_time < time_window
        ]

        return max_requests - len(cls.requests[identifier])

    @classmethod
    def get_reset_time(cls, identifier: str, time_window: int = None) -> float:
        """
        Get the time when the rate limit will reset for an identifier

        :param identifier: The identifier to check
        :param time_window: Time window in seconds (uses config default if None)
        :return: Unix timestamp when the rate limit will reset
        """
        time_window = time_window or config.settings.RATE_LIMIT_WINDOW

        current_time = time.time()

        # Find the earliest request in the current window
        active_requests = [
            req_time for req_time in cls.requests[identifier]
            if current_time - req_time < time_window
        ]

        if not active_requests:
            return current_time

        # The reset time is the earliest request time + time window
        return min(active_requests) + time_window