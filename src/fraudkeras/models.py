"""Keras 3 models for card-fraud scoring.

* ``build_mlp``: numeric block + learned embeddings for ``channel`` and ``category``,
  two dense layers with batch-norm and dropout, sigmoid output.
* three ways to handle the 0.5% class imbalance: plain BCE, BCE with a positive class weight,
  and binary focal loss.
* ``build_autoencoder``: unsupervised; trained on legitimate transactions only, the
  reconstruction error is the anomaly score (so it can flag attacks it never saw).

Early stopping watches the PR-AUC on the later (validation) time block, never on a random
split, because a random split would leak the future.
"""

from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from keras import layers  # noqa: E402

from .data import CATEGORIES, NUMERIC  # noqa: E402

# Heavy-tailed, non-negative features get log1p before standardisation.
SKEWED = [
    "dist_home_km", "dist_prev_km", "speed_kmh", "cnt_10m", "cnt_1h", "cnt_24h",
    "sum_1h_ratio", "amount_ratio", "n_prev",
]  # fmt: skip
N_CHANNELS, N_CATEGORIES = 3, len(CATEGORIES)


class Prep:
    """log1p on skewed columns, then standardisation fitted on the training block only."""

    def fit(self, X: pd.DataFrame) -> Prep:
        raw = self._raw(X)
        self.mean, self.std = raw.mean(axis=0), raw.std(axis=0) + 1e-6
        return self

    def _raw(self, X: pd.DataFrame) -> np.ndarray:
        a = X[NUMERIC].to_numpy(dtype="float64").copy()
        idx = [NUMERIC.index(c) for c in SKEWED]
        a[:, idx] = np.log1p(np.maximum(a[:, idx], 0.0))
        return a

    def num(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip((self._raw(X) - self.mean) / self.std, -6, 6).astype("float32")

    def mlp_inputs(self, X: pd.DataFrame) -> dict:
        return {
            "num": self.num(X),
            "channel": X["channel"].to_numpy("int32").reshape(-1, 1),
            "category": X["category"].to_numpy("int32").reshape(-1, 1),
        }

    def dense_inputs(self, X: pd.DataFrame) -> np.ndarray:
        """Numeric block + one-hot channel and category (for the autoencoder)."""
        ch = np.eye(N_CHANNELS, dtype="float32")[X["channel"].to_numpy(int)]
        cat = np.eye(N_CATEGORIES, dtype="float32")[X["category"].to_numpy(int)]
        return np.hstack([self.num(X), ch, cat])


def build_mlp(n_num: int, loss: str = "bce_weighted", lr: float = 1e-3) -> keras.Model:
    num = keras.Input((n_num,), name="num")
    ch = keras.Input((1,), dtype="int32", name="channel")
    cat = keras.Input((1,), dtype="int32", name="category")
    e_ch = layers.Flatten()(layers.Embedding(N_CHANNELS, 2)(ch))
    e_cat = layers.Flatten()(layers.Embedding(N_CATEGORIES, 4)(cat))
    x = layers.Concatenate()([num, e_ch, e_cat])
    for units in (64, 32):
        x = layers.Dense(units, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid", name="p_fraud")(x)
    model = keras.Model([num, ch, cat], out)
    fn = (
        keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True, alpha=0.9, gamma=2.0)
        if loss == "focal"
        else keras.losses.BinaryCrossentropy()
    )
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=fn,
        metrics=[keras.metrics.AUC(curve="PR", name="pr_auc")],
    )
    return model


def fit_mlp(
    prep: Prep, X_tr, y_tr, X_va, y_va, loss: str = "bce_weighted", seed: int = 0, epochs: int = 40
) -> tuple[keras.Model, int]:
    """Returns the model restored to its best validation PR-AUC and the number of epochs run."""
    keras.utils.set_random_seed(seed)
    model = build_mlp(len(NUMERIC), loss)
    y_tr = np.asarray(y_tr)
    pos_weight = float(np.sqrt((y_tr == 0).sum() / max((y_tr == 1).sum(), 1)))
    class_weight = {0: 1.0, 1: pos_weight} if loss == "bce_weighted" else None
    stop = keras.callbacks.EarlyStopping(
        monitor="val_pr_auc", mode="max", patience=6, restore_best_weights=True
    )
    hist = model.fit(
        prep.mlp_inputs(X_tr), y_tr.astype("float32"),
        validation_data=(prep.mlp_inputs(X_va), np.asarray(y_va).astype("float32")),
        epochs=epochs, batch_size=1024, class_weight=class_weight, callbacks=[stop], verbose=0,
    )  # fmt: skip
    return model, len(hist.history["loss"])


def predict_mlp(model: keras.Model, prep: Prep, X) -> np.ndarray:
    return model.predict(prep.mlp_inputs(X), batch_size=4096, verbose=0).ravel()


def build_autoencoder(dim: int, lr: float = 1e-3) -> keras.Model:
    inp = keras.Input((dim,))
    x = layers.Dense(24, activation="relu")(inp)
    z = layers.Dense(8, activation="relu")(x)
    x = layers.Dense(24, activation="relu")(z)
    out = layers.Dense(dim)(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="mse")
    return model


def fit_autoencoder(prep: Prep, X_tr, y_tr, X_va, y_va, seed: int = 0, epochs: int = 40) -> keras.Model:
    """Trained on legitimate rows only; early stopping on legitimate validation rows."""
    keras.utils.set_random_seed(seed)
    a, b = prep.dense_inputs(X_tr[np.asarray(y_tr) == 0]), prep.dense_inputs(X_va[np.asarray(y_va) == 0])
    model = build_autoencoder(a.shape[1])
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
    model.fit(a, a, validation_data=(b, b), epochs=epochs, batch_size=1024, callbacks=[stop], verbose=0)
    return model


def anomaly_score(model: keras.Model, prep: Prep, X) -> np.ndarray:
    z = prep.dense_inputs(X)
    rec = model.predict(z, batch_size=4096, verbose=0)
    return ((z - rec) ** 2).mean(axis=1)
