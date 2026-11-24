import numpy as np
import pandas as pd
import pytest

from app.config import (
    BACKTEST_FOLD_HOURS, MIN_FOLD_HOURS, MIN_TRAIN_HOURS, N_FORECAST_HOURS,
)
from app.models import backtest as bt
from app.models.features import build_rows, from_model_space, to_model_space
from tests.conftest import make_hourly

H = N_FORECAST_HOURS


# ── plan_folds ───────────────────────────────────────────────────────────────

def test_folds_are_contiguous_and_end_at_last_hour():
    idx = pd.date_range("2025-10-01", periods=24 * 365, freq="h")
    folds = bt.plan_folds(idx, n_folds=6)
    assert len(folds) == 6
    assert folds[-1][1] == idx[-1] + pd.Timedelta(hours=1)
    for (s, e), (s2, _) in zip(folds, folds[1:]):
        assert e == s2
        assert e - s == pd.Timedelta(hours=BACKTEST_FOLD_HOURS)
    assert folds[0][0] - idx[0] >= pd.Timedelta(hours=MIN_TRAIN_HOURS)


def test_folds_shrink_for_new_sensor():
    """Sensor berumur sebulan tetap bisa dilatih, dengan lipatan lebih pendek."""
    idx = pd.date_range("2026-10-04", periods=714, freq="h")
    folds = bt.plan_folds(idx, n_folds=6)
    size = folds[0][1] - folds[0][0]
    assert 2 <= len(folds) <= 6
    assert pd.Timedelta(hours=MIN_FOLD_HOURS) <= size < pd.Timedelta(hours=BACKTEST_FOLD_HOURS)
    assert folds[0][0] - idx[0] >= pd.Timedelta(hours=MIN_TRAIN_HOURS)


def test_folds_reduce_count_before_giving_up():
    n = MIN_TRAIN_HOURS + 2 * MIN_FOLD_HOURS
    folds = bt.plan_folds(pd.date_range("2026-01-01", periods=n, freq="h"), n_folds=6)
    assert len(folds) == 2


def test_too_little_data_raises_clear_error():
    n = MIN_TRAIN_HOURS + 2 * MIN_FOLD_HOURS - 1
    with pytest.raises(ValueError, match="Data belum cukup"):
        bt.plan_folds(pd.date_range("2026-01-01", periods=n, freq="h"))


# ── backtest_target ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def oos_and_inputs():
    df = make_hourly(days=40)
    z = to_model_space(df)
    X, meta = build_rows(z, "tsp")
    folds = bt.plan_folds(z.index, n_folds=3, fold_hours=24 * 5)
    params = {"n_estimators": 25, "learning_rate": 0.2, "num_leaves": 7,
              "min_child_samples": 10, "n_jobs": 1}
    return bt.backtest_target(X, meta, z["tsp"], folds, params), X, meta, z, folds, params


def test_backtest_covers_every_fold(oos_and_inputs):
    oos, _, _, _, folds, _ = oos_and_inputs
    assert sorted(oos["fold"].unique()) == [0, 1, 2]
    for k, (s, e) in enumerate(folds):
        t = oos.loc[oos["fold"] == k, "t"]
        assert t.min() >= s and t.max() < e


def test_training_rows_never_overlap_the_test_fold(oos_and_inputs, monkeypatch):
    """Tidak ada baris latih yang JAM TARGETNYA masuk lipatan uji (purging)."""
    _, X, meta, z, folds, params = oos_and_inputs
    seen = []
    real_fit = bt.fit_models

    def spy(X_train, y, p=None):
        seen.append(X_train.index)
        return real_fit(X_train, y, p)

    monkeypatch.setattr(bt, "fit_models", spy)
    bt.backtest_target(X, meta, z["tsp"], folds, params)
    assert len(seen) == len(folds)
    for rows, (start, _) in zip(seen, folds):
        used = meta.loc[rows]
        target_time = used["t"] + pd.to_timedelta(used["h"], unit="h")
        assert (target_time < start).all()
        assert used["delta"].notna().all()


def test_first_fold_is_calibration_only(oos_and_inputs):
    oos = oos_and_inputs[0]
    first = oos[oos["fold"] == 0]
    assert first[["cal_point", "cal_lower", "cal_upper"]].isna().all().all()
    assert first[["point", "lower", "upper"]].notna().all().all()
    later = oos[oos["fold"] > 0]
    assert later[["cal_point", "cal_lower", "cal_upper"]].notna().all().all()


def test_calibration_for_a_fold_uses_only_earlier_folds(oos_and_inputs):
    """Kalibrasi lipatan k harus bisa direproduksi HANYA dari lipatan < k."""
    oos = oos_and_inputs[0]
    for k in (1, 2):
        alpha, conformal = bt.fit_calibration(oos[oos["fold"] < k])
        cur = oos[oos["fold"] == k]
        expected, _, _ = bt.calibrated_deltas(
            cur["point"].values, cur["lower"].values, cur["upper"].values,
            cur["h"].values, alpha, conformal)
        assert np.allclose(cur["cal_point"].values, expected)


def test_calibrated_band_is_ordered(oos_and_inputs):
    later = oos_and_inputs[0].query("fold > 0")
    assert (later["cal_lower"] <= later["cal_point"]).all()
    assert (later["cal_point"] <= later["cal_upper"]).all()


def test_seasonal_reference_is_same_hour_yesterday(oos_and_inputs):
    oos, _, _, z, _, _ = oos_and_inputs
    row = oos.iloc[len(oos) // 2]
    tt = row["t"] + pd.Timedelta(hours=int(row["h"]))
    assert row["z_seasonal"] == pytest.approx(z["tsp"].loc[tt - pd.Timedelta(hours=24)])


def test_backtest_returns_empty_when_nothing_can_be_trained():
    df = make_hourly(days=22)
    z = to_model_space(df)
    z["tsp"] = np.nan
    z.iloc[-30:, z.columns.get_loc("tsp")] = 3.0
    X, meta = build_rows(z, "tsp")
    assert bt.backtest_target(X, meta, z["tsp"], bt.plan_folds(z.index), {"n_estimators": 5}).empty


# ── metrik ───────────────────────────────────────────────────────────────────

def _fake_oos(n_per_h=400, model_noise=0.05, seed=0):
    """Prakiraan luar-sampel buatan dengan selisih sebenarnya ~N(0, 0,5)."""
    rng = np.random.default_rng(seed)
    rows = []
    for h in range(1, H + 1):
        t = pd.date_range("2026-05-01", periods=n_per_h, freq="h")
        delta = rng.normal(0, 0.5, n_per_h)
        anchor = np.full(n_per_h, np.log1p(100.0))
        point = delta + rng.normal(0, model_noise, n_per_h)
        rows.append(pd.DataFrame({
            "t": t, "h": h, "anchor": anchor, "z_true": anchor + delta, "delta": delta,
            "fold": 1, "point": point, "lower": point - 0.2, "upper": point + 0.2,
            "cal_point": point, "cal_lower": point - 0.2, "cal_upper": point + 0.2,
            "z_seasonal": anchor + rng.normal(0, 0.7, n_per_h),
        }))
    return pd.concat(rows, ignore_index=True)


def test_score_is_reported_in_physical_units():
    oos = _fake_oos()
    report = bt.score_target(oos, "tsp")
    y = np.expm1(oos["z_true"].values)
    pred = np.expm1((oos["anchor"] + oos["cal_point"]).values)
    assert report["overall"]["mae"] == pytest.approx(np.abs(pred - y).mean(), abs=1e-3)
    # Persistence = nilai jangkar = 100 µg/m³ untuk semua baris.
    assert report["overall"]["mae_persistence"] == pytest.approx(np.abs(100.0 - y).mean(), abs=1e-3)
    assert report["overall"]["n"] == len(oos)


def test_good_model_shows_positive_skill_and_significant_dm():
    report = bt.score_target(_fake_oos(model_noise=0.05), "tsp")
    assert report["overall"]["skill_vs_persistence"] > 0.8
    assert report["overall"]["skill_vs_seasonal_naive"] > 0.8
    for step in range(1, H + 1):
        block = report["by_horizon"][str(step)]
        assert block["skill_vs_persistence"] > 0.8
        assert block["dm_pvalue_vs_persistence"] < 0.001


def test_useless_model_shows_negative_skill():
    report = bt.score_target(_fake_oos(model_noise=2.0), "tsp")
    assert report["overall"]["skill_vs_persistence"] < 0


def test_coverage_counts_truth_inside_band():
    oos = _fake_oos(model_noise=0.05)
    assert bt.score_target(oos, "tsp")["overall"]["coverage"] > 0.99      # ±0,2 vs derau 0,05
    oos["cal_lower"] = oos["cal_point"] - 1e-6
    oos["cal_upper"] = oos["cal_point"] + 1e-6
    assert bt.score_target(oos, "tsp")["overall"]["coverage"] < 0.01


def test_rows_without_calibration_are_not_scored():
    oos = _fake_oos()
    first = oos.copy()
    first["fold"] = 0
    first[["cal_point", "cal_lower", "cal_upper"]] = np.nan
    report = bt.score_target(pd.concat([first, oos], ignore_index=True), "tsp")
    assert report["overall"]["n"] == len(oos)
    assert report["folds_scored"] == 1
    assert bt.score_target(first, "tsp") is None


def test_exceedance_detection_metrics():
    n = 200
    anchor = np.full(n, np.log1p(100.0))
    truth = np.where(np.arange(n) % 4 == 0, 400.0, 100.0)       # 50 jam pelampauan TSP > 230
    z_true = np.log1p(truth)
    oos = pd.DataFrame({
        "t": pd.date_range("2026-05-01", periods=n, freq="h"), "h": 1, "anchor": anchor,
        "z_true": z_true, "delta": z_true - anchor, "fold": 1,
        "point": z_true - anchor, "lower": z_true - anchor - 0.1, "upper": z_true - anchor + 0.1,
        "cal_point": z_true - anchor, "cal_lower": z_true - anchor - 0.1,
        "cal_upper": z_true - anchor + 0.1, "z_seasonal": np.nan,
    })
    exc = bt.score_target(oos, "tsp")["exceedance"]
    assert exc["threshold"] == 230.0 and exc["events"] == 50
    assert exc["point"] == {"pod": 1.0, "far": 0.0, "csi": 1.0}
    # Persistence (selalu 100) tidak pernah memperingatkan.
    assert exc["persistence"]["pod"] == 0.0 and exc["persistence"]["far"] is None


def test_no_exceedance_block_for_parameters_without_threshold():
    oos = _fake_oos()
    oos["anchor"] = 27.0
    oos["z_true"] = oos["anchor"] + oos["delta"]
    assert "exceedance" not in bt.score_target(oos, "temp")


def test_diebold_mariano_basic_properties():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 2000)
    assert bt.diebold_mariano(0.2 * ref, ref, h=1) < 1e-6
    same_quality = rng.normal(0, 1, 2000)
    assert bt.diebold_mariano(same_quality, ref, h=3) > 0.01
    assert bt.diebold_mariano(ref, ref, h=1) is None              # selisih kerugian nol
    assert bt.diebold_mariano(ref[:10], ref[:10] * 2, h=1) is None  # sampel terlalu sedikit


def test_normalized_mae_is_dimensionless_fraction_of_range():
    oos = _fake_oos(model_noise=0.05)
    series = pd.Series(np.linspace(0.0, 10.0, 50))
    value = bt.normalized_mae(oos, series)
    assert value == pytest.approx(np.abs(oos["cal_point"] - oos["delta"]).mean() / 10.0)
    assert 0 < value < 0.05
    assert bt.normalized_mae(pd.DataFrame(), series) is None
