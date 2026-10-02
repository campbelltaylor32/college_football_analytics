import pandas as pd

from cfb_playcall_predictability.plays import build_calls, is_garbage_time, label_calls


def _pbp(**overrides):
    base = {
        "play_type": "Rush", "play_text": "J.Doe run for 4 yds", "rush": 1, "pass": 0,
        "down": 1.0, "distance": 10.0, "yards_to_goal": 60.0, "period": 1,
        "pos_score_diff_start": 0.0, "game_id": 1,
    }
    return {**base, **overrides}


def test_play_type_labels(data_cfg):
    pbp = pd.DataFrame([
        _pbp(play_type="Rush"), _pbp(play_type="Rushing Touchdown"),
        _pbp(play_type="Pass Reception", rush=0, **{"pass": 1}), _pbp(play_type="Sack", rush=0, **{"pass": 1}),
        _pbp(play_type="Interception Return", rush=0, **{"pass": 1}),
        _pbp(play_type="Punt", rush=0), _pbp(play_type="Penalty", rush=0), _pbp(play_type="Timeout", rush=0),
    ])
    assert label_calls(pbp, data_cfg).tolist()[:5] == [0, 0, 1, 1, 1]
    assert label_calls(pbp, data_cfg).iloc[5:].isna().all()


def test_fumbles_use_cfbfastr_flags(data_cfg):
    pbp = pd.DataFrame([
        _pbp(play_type="Fumble Recovery (Opponent)", rush=1, **{"pass": 0}),
        _pbp(play_type="Fumble Recovery (Own)", rush=0, **{"pass": 1}),
        _pbp(play_type="Fumble Recovery (Own)", rush=0, **{"pass": 0}),  # unknown -> dropped
    ])
    labels = label_calls(pbp, data_cfg)
    assert labels.iloc[0] == 0 and labels.iloc[1] == 1 and pd.isna(labels.iloc[2])


def test_kneels_spikes_overtime_and_no_down_dropped(data_cfg):
    pbp = pd.DataFrame([
        _pbp(),
        _pbp(play_text="QB kneel for loss of 2"),
        _pbp(play_type="Pass Incompletion", play_text="QB spike", rush=0, **{"pass": 1}),
        _pbp(period=5),
        _pbp(down=float("nan")),
    ])
    calls = build_calls(pbp, data_cfg)
    assert len(calls) == 1


def test_garbage_time_thresholds(data_cfg):
    pbp = pd.DataFrame([
        _pbp(period=1, pos_score_diff_start=50.0),   # Q1 never garbage
        _pbp(period=2, pos_score_diff_start=38.0),   # at threshold -> not garbage
        _pbp(period=2, pos_score_diff_start=-39.0),
        _pbp(period=4, pos_score_diff_start=23.0),
        _pbp(period=4, pos_score_diff_start=22.0),
    ])
    assert is_garbage_time(pbp, data_cfg).tolist() == [False, False, True, True, False]
