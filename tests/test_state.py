from realtime_summary.state import CompletedTurn, SummaryState, TurnTracker


def test_completed_turns_are_released_in_commit_order() -> None:
    clock = [0.0]
    tracker = TurnTracker(clock=lambda: clock[0])
    tracker.commit("item-a", previous_item_id=None)
    tracker.commit("item-b", previous_item_id="item-a")

    clock[0] = 5.0
    assert tracker.complete("item-b", "二番目の発話") == []
    clock[0] = 8.0
    released = tracker.complete("item-a", "最初の発話")
    assert released == [
        CompletedTurn(0, "item-a", "最初の発話", 8.0),
        CompletedTurn(1, "item-b", "二番目の発話", 5.0),
    ]
    assert [turn.completed_at for turn in released] == [8.0, 5.0]


def test_summary_window_uses_completion_time_and_retains_failed_input() -> None:
    clock = [1000.0]
    state = SummaryState(window_seconds=300.0, clock=lambda: clock[0])
    state.add_transcript("直近の先行発話", completed_at=999.0)
    state.add_transcript("遅れて届いた古い発話", completed_at=699.0)
    state.add_transcript("境界の発話", completed_at=700.0)
    first = state.begin_update()
    assert first is not None
    assert first.transcripts == ("直近の先行発話", "境界の発話")

    state.add_transcript("処理中に届いた発話", completed_at=1000.0)
    state.fail_update()
    retry = state.begin_update()
    assert retry is not None
    assert retry.transcripts == (
        "直近の先行発話",
        "境界の発話",
        "処理中に届いた発話",
    )

    state.succeed_update(display_lines=("話題", "", ""))
    clock[0] = 1300.0
    later = state.begin_update()
    assert later is not None
    assert later.transcripts == ("処理中に届いた発話",)
