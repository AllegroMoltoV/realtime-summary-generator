from realtime_summary.state import CompletedTurn, SummaryState, TurnTracker


def test_completed_turns_are_released_in_commit_order() -> None:
    tracker = TurnTracker()
    tracker.commit("item-a", previous_item_id=None)
    tracker.commit("item-b", previous_item_id="item-a")

    assert tracker.complete("item-b", "二番目の発話") == []
    assert tracker.complete("item-a", "最初の発話") == [
        CompletedTurn(sequence=0, item_id="item-a", transcript="最初の発話"),
        CompletedTurn(sequence=1, item_id="item-b", transcript="二番目の発話"),
    ]


def test_failed_summary_input_returns_before_new_transcripts() -> None:
    state = SummaryState(max_cycles=3)
    state.add_transcript("失敗した周期の発話")
    first = state.begin_update()
    assert first is not None

    state.add_transcript("処理中に届いた発話")
    state.fail_update()
    retry = state.begin_update()

    assert retry is not None
    assert retry.transcripts == ("失敗した周期の発話", "処理中に届いた発話")
