"""Thread-safe application-level chat rate and token quota enforcement."""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import threading
import time
from typing import Callable

import tiktoken


UTC = timezone.utc


class ChatLimitExceeded(RuntimeError):
    """Describe an admission limit and when the client may retry."""

    def __init__(self, message: str, retry_after: int):
        super().__init__(message)
        self.retry_after = max(1, int(retry_after))


@dataclass
class _Usage:
    request_times: deque[float] = field(default_factory=deque)
    quota_day: str = ""
    charged_tokens: int = 0
    last_seen: float = 0.0


class ChatUsageLimiter:
    """Bound request frequency and estimated daily model usage per user.

    The quota is deliberately conservative rather than billing-accurate. Each
    admitted turn is charged its input token count plus a fixed base charge for
    planner/output work. Accepted requests remain charged when downstream work
    fails because provider resources may already have been consumed.
    """

    def __init__(
        self,
        *,
        max_requests: int,
        window_seconds: int,
        daily_token_quota: int,
        base_token_charge: int,
        max_tracked_users: int = 10_000,
        monotonic: Callable[[], float] = time.monotonic,
        wall_time: Callable[[], float] = time.time,
    ):
        values = (
            max_requests,
            window_seconds,
            daily_token_quota,
            max_tracked_users,
        )
        if any(value <= 0 for value in values) or base_token_charge < 0:
            raise ValueError("Chat limit settings must be positive")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.daily_token_quota = daily_token_quota
        self.base_token_charge = base_token_charge
        self.max_tracked_users = max_tracked_users
        self._monotonic = monotonic
        self._wall_time = wall_time
        self._lock = threading.Lock()
        self._usage: OrderedDict[str, _Usage] = OrderedDict()
        self._encoding = tiktoken.get_encoding("cl100k_base")

    def estimate_charge(self, message: str) -> int:
        return len(self._encoding.encode(message)) + self.base_token_charge

    @staticmethod
    def _seconds_until_next_utc_day(now: datetime) -> int:
        tomorrow = datetime.combine(
            now.date() + timedelta(days=1), datetime.min.time(), tzinfo=UTC
        )
        return max(1, int((tomorrow - now).total_seconds()))

    def check(self, user_id: str, message: str) -> dict:
        if not user_id:
            raise ValueError("user_id is required for chat limiting")
        charge = self.estimate_charge(message)
        now_mono = self._monotonic()
        now_utc = datetime.fromtimestamp(self._wall_time(), UTC)
        today = now_utc.date().isoformat()

        with self._lock:
            usage = self._usage.get(user_id)
            if usage is None:
                usage = _Usage(quota_day=today)
                self._usage[user_id] = usage
            else:
                self._usage.move_to_end(user_id)

            cutoff = now_mono - self.window_seconds
            while usage.request_times and usage.request_times[0] <= cutoff:
                usage.request_times.popleft()
            if usage.quota_day != today:
                usage.quota_day = today
                usage.charged_tokens = 0

            if len(usage.request_times) >= self.max_requests:
                retry_after = self.window_seconds - (
                    now_mono - usage.request_times[0]
                )
                raise ChatLimitExceeded(
                    "Too many chat requests; try again later", retry_after
                )
            if usage.charged_tokens + charge > self.daily_token_quota:
                raise ChatLimitExceeded(
                    "Daily chat token quota exceeded",
                    self._seconds_until_next_utc_day(now_utc),
                )

            usage.request_times.append(now_mono)
            usage.charged_tokens += charge
            usage.last_seen = now_mono

            while len(self._usage) > self.max_tracked_users:
                self._usage.popitem(last=False)

            return {
                "charged_tokens": charge,
                "remaining_tokens": self.daily_token_quota
                - usage.charged_tokens,
                "remaining_requests": self.max_requests
                - len(usage.request_times),
            }

    def clear(self) -> None:
        """Reset state, primarily for deterministic tests."""
        with self._lock:
            self._usage.clear()
