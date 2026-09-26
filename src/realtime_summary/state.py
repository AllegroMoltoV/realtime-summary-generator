from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from time import monotonic


class TurnOrderError(ValueError):
    """発話の前後関係が現在のセッションと一致しない。"""


@dataclass(frozen=True, slots=True)
class CompletedTurn:
    sequence: int
    item_id: str
    transcript: str
    completed_at: float = field(compare=False)


@dataclass(frozen=True, slots=True)
class SummaryRequest:
    transcripts: tuple[str, ...]


class SummaryState:
    def __init__(
        self,
        *,
        window_seconds: float,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self._window_seconds = window_seconds
        self._clock = clock
        self._transcripts: list[tuple[float, str]] = []
        self._in_flight = False
        self.display_lines: tuple[str, str, str] | None = None

    def add_transcript(self, transcript: str, *, completed_at: float) -> bool:
        now = self._clock()
        self._prune(now)
        if not transcript or completed_at < now - self._window_seconds:
            return False
        self._transcripts.append((completed_at, transcript))
        return True

    def begin_update(self) -> SummaryRequest | None:
        if self._in_flight:
            return None
        self._prune(self._clock())
        if not self._transcripts:
            return None

        self._in_flight = True
        return SummaryRequest(tuple(text for _, text in self._transcripts))

    def succeed_update(self, *, display_lines: tuple[str, str, str]) -> None:
        if not self._in_flight:
            raise RuntimeError("No summary update is in progress")
        self.display_lines = display_lines
        self._in_flight = False

    def fail_update(self) -> None:
        if not self._in_flight:
            raise RuntimeError("No summary update is in progress")
        self._in_flight = False

    def _prune(self, now: float) -> None:
        cutoff = now - self._window_seconds
        self._transcripts = [
            (completed_at, text)
            for completed_at, text in self._transcripts
            if completed_at >= cutoff
        ]


class TurnTracker:
    def __init__(self, *, clock: Callable[[], float] = monotonic) -> None:
        self._clock = clock
        self._sequence_by_item: dict[str, int] = {}
        self._resolved: dict[int, CompletedTurn | None] = {}
        self._next_release = 0

    def commit(self, item_id: str, *, previous_item_id: str | None) -> int:
        if item_id in self._sequence_by_item:
            return self._sequence_by_item[item_id]

        if previous_item_id is None:
            sequence = 0
        else:
            try:
                sequence = self._sequence_by_item[previous_item_id] + 1
            except KeyError as exc:
                raise TurnOrderError(
                    f"Unknown previous item for committed turn: {previous_item_id}"
                ) from exc

        if sequence in self._sequence_by_item.values():
            raise TurnOrderError(f"Duplicate turn sequence: {sequence}")
        self._sequence_by_item[item_id] = sequence
        return sequence

    def complete(self, item_id: str, transcript: str) -> list[CompletedTurn]:
        sequence = self._require_sequence(item_id)
        self._resolved[sequence] = CompletedTurn(
            sequence, item_id, transcript, self._clock()
        )
        return self._drain()

    def fail(self, item_id: str) -> list[CompletedTurn]:
        sequence = self._require_sequence(item_id)
        self._resolved[sequence] = None
        return self._drain()

    def sequence_for(self, item_id: str) -> int | None:
        return self._sequence_by_item.get(item_id)

    @property
    def pending_count(self) -> int:
        return sum(
            sequence >= self._next_release
            for sequence in self._sequence_by_item.values()
        )

    def _require_sequence(self, item_id: str) -> int:
        try:
            return self._sequence_by_item[item_id]
        except KeyError as exc:
            raise TurnOrderError(f"Unknown transcription item: {item_id}") from exc

    def _drain(self) -> list[CompletedTurn]:
        released: list[CompletedTurn] = []
        while self._next_release in self._resolved:
            turn = self._resolved.pop(self._next_release)
            self._next_release += 1
            if turn is not None:
                released.append(turn)
        return released
