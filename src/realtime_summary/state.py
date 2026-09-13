from __future__ import annotations

from collections import deque
from dataclasses import dataclass


class TurnOrderError(ValueError):
    """発話の前後関係が現在のセッションと一致しない。"""


@dataclass(frozen=True, slots=True)
class CompletedTurn:
    sequence: int
    item_id: str
    transcript: str


@dataclass(frozen=True, slots=True)
class SummaryRequest:
    context_memory: str
    transcripts: tuple[str, ...]


class SummaryState:
    def __init__(self, *, max_cycles: int) -> None:
        if max_cycles < 1:
            raise ValueError("max_cycles must be positive")
        self._max_cycles = max_cycles
        self._current: list[str] = []
        self._pending: deque[tuple[str, ...]] = deque()
        self._in_flight: tuple[str, ...] | None = None
        self.context_memory = ""
        self.display_lines: tuple[str, str, str] | None = None
        self.dropped_cycles = 0

    def add_transcript(self, transcript: str) -> None:
        if transcript:
            self._current.append(transcript)

    def begin_update(self) -> SummaryRequest | None:
        if self._in_flight is not None:
            return None
        self._close_current_cycle()
        if not self._pending:
            return None

        transcripts = tuple(text for cycle in self._pending for text in cycle)
        self._pending.clear()
        self._in_flight = transcripts
        return SummaryRequest(self.context_memory, transcripts)

    def succeed_update(
        self,
        *,
        context_memory: str,
        display_lines: tuple[str, str, str],
    ) -> None:
        if self._in_flight is None:
            raise RuntimeError("No summary update is in progress")
        self.context_memory = context_memory
        self.display_lines = display_lines
        self._in_flight = None

    def fail_update(self) -> None:
        if self._in_flight is None:
            raise RuntimeError("No summary update is in progress")
        self._pending.appendleft(self._in_flight)
        self._in_flight = None
        self._trim_pending()

    @property
    def pending_cycles(self) -> int:
        return len(self._pending) + bool(self._current)

    def _close_current_cycle(self) -> None:
        if self._current:
            self._pending.append(tuple(self._current))
            self._current.clear()
            self._trim_pending()

    def _trim_pending(self) -> None:
        while len(self._pending) > self._max_cycles:
            self._pending.popleft()
            self.dropped_cycles += 1


class TurnTracker:
    def __init__(self) -> None:
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
        self._resolved[sequence] = CompletedTurn(sequence, item_id, transcript)
        return self._drain()

    def fail(self, item_id: str) -> list[CompletedTurn]:
        sequence = self._require_sequence(item_id)
        self._resolved[sequence] = None
        return self._drain()

    def sequence_for(self, item_id: str) -> int | None:
        return self._sequence_by_item.get(item_id)

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
