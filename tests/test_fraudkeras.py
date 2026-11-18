import numpy as np
import pandas as pd
import pytest

from fraudkeras import metrics, models
from fraudkeras.data import NUMERIC, build_features, simulate, split_by_day


@pytest.fixture(scope="module")
def small():
    df = simulate(n_cards=500, days=90, seed=1)
    X = build_features(df)
    tr, va, te = split_by_day(df)
    return df, X, tr, va, te


def test_features_do_not_look_into_the_future(small):
    df, X, *_ = small
    prefix = df[df["time"] < df["time"].quantile(0.6)]
    Xp = build_features(prefix)
    pd.testing.assert_frame_equal(X.loc[Xp.index], Xp, check_exact=False, atol=1e-9)


def test_mule_cashout_only_in_test_period(small):
    df, _, tr, va, te = small
    mule = (df["pattern"] == "mule_cashout").to_numpy()
    assert mule[te].any() and not mule[tr].any() and not mule[va].any()


def test_preprocessing_is_fit_on_training_rows_only(small):
    _, X, tr, va, _ = small
    prep = models.Prep().fit(X[tr])
    z = prep.num(X[tr])
    assert z.shape[1] == len(NUMERIC)
    assert np.abs(z.mean(axis=0)).max() < 0.2  # roughly centred on the training block
    assert np.abs(prep.num(X[va])).max() <= 6.0  # clipped


def test_dense_inputs_have_one_hot_blocks(small):
    _, X, tr, *_ = small
    prep = models.Prep().fit(X[tr])
    d = prep.dense_inputs(X[tr].head(50))
    assert d.shape[1] == len(NUMERIC) + models.N_CHANNELS + models.N_CATEGORIES
    assert (d[:, len(NUMERIC) :].sum(axis=1) == 2).all()


def test_metrics_average_precision_known_value():
    y = np.array([1, 0, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    assert metrics.average_precision(y, s) == pytest.approx((1 * 1 + 2 / 3 * 1) / 2)


@pytest.mark.parametrize("loss", ["bce", "bce_weighted", "focal"])
def test_mlp_variants_train_and_output_probabilities(small, loss):
    df, X, tr, va, te = small
    y = df["is_fraud"].to_numpy()
    prep = models.Prep().fit(X[tr])
    model, epochs = models.fit_mlp(prep, X[tr], y[tr], X[va], y[va], loss=loss, epochs=3)
    p = models.predict_mlp(model, prep, X[te])
    assert epochs >= 1 and p.shape == (te.sum(),)
    assert ((p >= 0) & (p <= 1)).all()


def test_mlp_learns_known_patterns_better_than_chance(small):
    df, X, tr, va, te = small
    y = df["is_fraud"].to_numpy()
    prep = models.Prep().fit(X[tr])
    model, _ = models.fit_mlp(prep, X[tr], y[tr], X[va], y[va], epochs=15)
    known = te & (df["pattern"] != "mule_cashout").to_numpy()
    ap = metrics.average_precision(y[known], models.predict_mlp(model, prep, X[known]))
    assert ap > 5 * y[known].mean()


def test_same_seed_gives_same_predictions(small):
    df, X, tr, va, te = small
    y = df["is_fraud"].to_numpy()
    prep = models.Prep().fit(X[tr])
    a, _ = models.fit_mlp(prep, X[tr], y[tr], X[va], y[va], epochs=2, seed=3)
    b, _ = models.fit_mlp(prep, X[tr], y[tr], X[va], y[va], epochs=2, seed=3)
    np.testing.assert_allclose(models.predict_mlp(a, prep, X[te]), models.predict_mlp(b, prep, X[te]), atol=1e-4)


def test_autoencoder_scores_are_nonnegative_and_flag_odd_rows(small):
    df, X, tr, va, te = small
    y = df["is_fraud"].to_numpy()
    prep = models.Prep().fit(X[tr])
    ae = models.fit_autoencoder(prep, X[tr], y[tr], X[va], y[va], epochs=5)
    s = models.anomaly_score(ae, prep, X[te])
    assert (s >= 0).all()
    assert s[y[te] == 1].mean() > s[y[te] == 0].mean()  # fraud reconstructs worse than normal traffic


def test_pipeline_smoke():
    from fraudkeras.pipeline import run_eval

    res = run_eval(n_cards=400, days=90, seed=2, n_seeds=2, n_boot=10)
    assert {"mlp_bce", "mlp_bce_weighted", "mlp_focal", "autoencoder", "hybrid_mean"} <= set(res["models"])
    assert res["seed_variance_mlp_bce_weighted"]["n_seeds"] == 2
