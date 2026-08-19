"""Harness invariants (Phase 2). All CPU, all synthetic, runs in seconds.

The three invariants from the project plan:
  1. Averaging two identical clients must return the identical model.
  2. A one-client federation must equal centralised training on that client.
  3. Client weights must sum to one (and be non-negative).

Plus sanity checks that pin down the loss and the FedOpt server math.

Determinism notes: tests use a plain nn.Linear (no dropout/BN), SGD (stateless,
so per-round optimizer re-initialisation is a no-op), and shuffle=False loaders
sized so that local_steps == batches_per_epoch — which makes the federated
data order identical to the centralised one.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedisic.fed.averaging import (  # noqa: E402
    check_weights,
    normalized_client_weights,
    weighted_average,
)
from fedisic.fed.simulate import Client, FedConfig, run_federated  # noqa: E402
from fedisic.fed.strategies import ServerOptimizer, local_train  # noqa: E402
from fedisic.losses import WeightedFocalLoss, inverse_frequency_alpha  # noqa: E402

D, C = 5, 3  # feature dim, classes


def make_dataset(n=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, D, generator=g)
    y = torch.randint(0, C, (n,), generator=g)
    return TensorDataset(X, y)


def make_loader(ds, bs=8):
    return DataLoader(ds, batch_size=bs, shuffle=False)


def make_client(cid, ds, bs=8):
    return Client(
        id=cid,
        name=f"c{cid}",
        train_loader=make_loader(ds, bs),
        test_loader=None,
        n_train=len(ds),
        loss_fn=nn.CrossEntropyLoss(),
    )


def linear_model(seed=0):
    torch.manual_seed(seed)
    return nn.Linear(D, C)


def sd_allclose(a, b, atol=1e-6):
    assert set(a.keys()) == set(b.keys())
    return all(torch.allclose(a[k], b[k], atol=atol) for k in a)


def sd_distance(a, b):
    return sum(torch.norm(a[k].float() - b[k].float()).item() for k in a)


# ---------------------------- invariant 3 ---------------------------------- #

def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        check_weights([0.5, 0.4])
    with pytest.raises(ValueError):
        check_weights([1.5, -0.5])
    check_weights(normalized_client_weights([9930, 3163, 2691, 1807, 655, 351]))


def test_weighted_average_hand_check():
    sd1 = {"w": torch.zeros(2, 2)}
    sd2 = {"w": torch.full((2, 2), 2.0)}
    out = weighted_average([sd1, sd2], [0.75, 0.25])
    assert torch.allclose(out["w"], torch.full((2, 2), 0.5))


def test_integer_buffers_keep_dtype():
    # BatchNorm's num_batches_tracked is int64; averaging must not change dtype.
    sd1 = {"b": torch.tensor(3, dtype=torch.int64)}
    sd2 = {"b": torch.tensor(3, dtype=torch.int64)}
    out = weighted_average([sd1, sd2], [0.5, 0.5])
    assert out["b"].dtype == torch.int64
    assert out["b"].item() == 3


# ---------------------------- invariant 1 ---------------------------------- #

def test_identical_clients_average_to_identical_model():
    ds = make_dataset()
    clients = [make_client(0, ds), make_client(1, ds)]
    model = linear_model()
    init = {k: v.clone() for k, v in model.state_dict().items()}

    cfg = FedConfig(strategy="fedavg", rounds=2, local_steps=5, lr=0.1,
                    optimizer="sgd", device="cpu")
    fed_sd = run_federated(model, clients, cfg)

    solo = linear_model()
    solo.load_state_dict(init)
    solo_sd = run_federated(solo, [make_client(0, ds)], cfg)
    assert sd_allclose(fed_sd, solo_sd)


# ---------------------------- invariant 2 ---------------------------------- #

def test_one_client_federation_equals_centralized():
    ds = make_dataset(n=40)  # 40 samples / batch 8 = 5 batches = local_steps
    model = linear_model()
    init = {k: v.clone() for k, v in model.state_dict().items()}

    cfg = FedConfig(strategy="fedavg", rounds=3, local_steps=5, lr=0.05,
                    optimizer="sgd", device="cpu")
    fed_sd = run_federated(model, [make_client(0, ds)], cfg)

    central = linear_model()
    central.load_state_dict(init)
    local_train(central, make_loader(ds, 8), nn.CrossEntropyLoss(),
                steps=15, lr=0.05, device="cpu", optimizer="sgd")
    assert sd_allclose(fed_sd, dict(central.state_dict()))


# ---------------------------- strategies ----------------------------------- #

def test_fedprox_mu_zero_matches_fedavg():
    ds = make_dataset()
    model_a = linear_model()
    init = {k: v.clone() for k, v in model_a.state_dict().items()}

    cfg_avg = FedConfig(strategy="fedavg", rounds=2, local_steps=5, lr=0.1,
                        optimizer="sgd", device="cpu")
    sd_avg = run_federated(model_a, [make_client(0, ds), make_client(1, make_dataset(seed=1))], cfg_avg)

    model_b = linear_model()
    model_b.load_state_dict(init)
    cfg_prox = FedConfig(strategy="fedprox", prox_mu=0.0, rounds=2, local_steps=5,
                         lr=0.1, optimizer="sgd", device="cpu")
    sd_prox = run_federated(model_b, [make_client(0, ds), make_client(1, make_dataset(seed=1))], cfg_prox)
    assert sd_allclose(sd_avg, sd_prox)


def test_fedprox_pulls_updates_toward_global():
    ds0, ds1 = make_dataset(seed=0), make_dataset(seed=1)
    model_a = linear_model()
    init = {k: v.clone() for k, v in model_a.state_dict().items()}

    cfg_avg = FedConfig(strategy="fedavg", rounds=3, local_steps=5, lr=0.1,
                        optimizer="sgd", device="cpu")
    sd_avg = run_federated(model_a, [make_client(0, ds0), make_client(1, ds1)], cfg_avg)

    model_b = linear_model()
    model_b.load_state_dict(init)
    cfg_prox = FedConfig(strategy="fedprox", prox_mu=10.0, rounds=3, local_steps=5,
                         lr=0.1, optimizer="sgd", device="cpu")
    sd_prox = run_federated(model_b, [make_client(0, ds0), make_client(1, ds1)], cfg_prox)

    assert sd_distance(sd_prox, init) < sd_distance(sd_avg, init)


def test_fedadagrad_server_step_numeric():
    opt = ServerOptimizer("fedadagrad", trainable_keys=["w"], lr=0.1,
                          beta1=0.9, beta2=0.99, tau=1e-3)
    g = {"w": torch.zeros(1)}
    a = {"w": torch.ones(1)}
    out = opt.step(g, a)
    # delta=1, m=(1-0.9)*1=0.1, v=tau^2+1, step = lr*m/(sqrt(v)+tau)
    expected = 0.1 * 0.1 / (math.sqrt(1.0 + 1e-6) + 1e-3)
    assert torch.allclose(out["w"], torch.tensor([expected]), atol=1e-8)


def test_fedopt_buffers_take_plain_average():
    opt = ServerOptimizer("fedadam", trainable_keys=["w"], lr=0.1)
    g = {"w": torch.zeros(1), "running_mean": torch.tensor([5.0])}
    a = {"w": torch.ones(1), "running_mean": torch.tensor([7.0])}
    out = opt.step(g, a)
    assert out["running_mean"].item() == pytest.approx(7.0)
    assert out["w"].item() != pytest.approx(1.0)  # adaptive step, not the raw average


# ------------------------------- loss --------------------------------------- #

def test_focal_reduces_to_cross_entropy_at_gamma0():
    torch.manual_seed(0)
    logits = torch.randn(16, C)
    y = torch.randint(0, C, (16,))
    fl = WeightedFocalLoss(alpha=torch.ones(C), gamma=0.0)(logits, y)
    ce = torch.nn.functional.cross_entropy(logits, y)
    assert torch.allclose(fl, ce, atol=1e-6)


def test_inverse_frequency_alpha_handles_absent_classes():
    a = inverse_frequency_alpha([10, 0, 30], 3)
    assert a[1] == 0.0
    assert a[0] > a[2] > 0.0


# --------------------------- aggregate reporting ---------------------------- #

def test_gap_closed_is_na_when_pooled_does_not_beat_local():
    """The metric normalises by pooled-minus-local. When that headroom is zero
    or negative the ratio flips sign or explodes, so it must refuse to answer
    rather than print a number that reads like a result."""
    from scripts.aggregate_results import gap_closed

    assert gap_closed(local=0.60, fed=0.65, pooled=0.70) == pytest.approx(50.0)
    assert gap_closed(local=0.70, fed=0.65, pooled=0.60) is None  # local beats pooled
    assert gap_closed(local=0.70, fed=0.72, pooled=0.70) is None  # zero headroom
    assert gap_closed(local=None, fed=0.65, pooled=0.70) is None  # arm absent


def test_gap_closed_reports_out_of_range_values():
    """Above 100% (federation beat centralised) and negative (worse than the
    centre's own model) are real findings, not errors to be clamped away."""
    from scripts.aggregate_results import gap_closed

    assert gap_closed(local=0.50, fed=0.80, pooled=0.70) == pytest.approx(150.0)
    assert gap_closed(local=0.50, fed=0.40, pooled=0.70) == pytest.approx(-50.0)


def test_agg_over_seeds_never_fakes_a_std_for_one_seed():
    """A single seed has no std. Printing +/- 0.000 would claim a precision the
    run cannot support."""
    from scripts.aggregate_results import agg_over_seeds

    mean, std, n = agg_over_seeds([0.75])
    assert (mean, std, n) == (pytest.approx(0.75), None, 1)

    mean, std, n = agg_over_seeds([0.70, 0.80])
    assert mean == pytest.approx(0.75)
    assert std == pytest.approx(0.0707, abs=1e-4)  # ddof=1, not the population std
    assert n == 2


def test_fast_balanced_accuracy_matches_sklearn():
    """The bootstrap runs tens of thousands of evaluations, so it uses a
    bincount implementation instead of sklearn. It must agree exactly,
    including when y_pred contains classes absent from y_true."""
    from sklearn.metrics import balanced_accuracy_score

    from scripts.aggregate_results import fast_balanced_accuracy

    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.integers(0, 8, 200)
        p = rng.integers(0, 8, 200)
        assert fast_balanced_accuracy(y, p, 8) == pytest.approx(
            balanced_accuracy_score(y, p)
        )
    # y_true missing several classes entirely
    y = rng.integers(0, 3, 100)
    p = rng.integers(0, 8, 100)
    assert fast_balanced_accuracy(y, p, 8) == pytest.approx(balanced_accuracy_score(y, p))


def test_paired_delta_of_a_model_against_itself_is_exactly_zero():
    """Same predictions on both sides means every replicate differences to 0,
    so the interval must be exactly [0, 0] -- no width from resampling noise.
    This is what pairing buys and an unpaired comparison cannot deliver."""
    from scripts.aggregate_results import paired_bootstrap_delta

    rng = np.random.default_rng(1)
    y = rng.integers(0, 8, 300)
    p = rng.integers(0, 8, 300)
    d, lo, hi = paired_bootstrap_delta(y, p, p.copy(), n_boot=200, alpha=0.05, seed=0)
    assert (d, lo, hi) == (0.0, 0.0, 0.0)


def test_paired_ci_is_narrower_than_unpaired_on_correlated_predictions():
    """Two arms that agree on most images have correlated errors. Pairing
    cancels that shared noise, so the interval on the difference must be
    strictly tighter than one built from the two marginal CIs."""
    from scripts.aggregate_results import fast_balanced_accuracy, paired_bootstrap_delta

    rng = np.random.default_rng(2)
    n = 600
    y = rng.integers(0, 8, n)
    # arm A: right 70% of the time. arm B: copies A except on a few images.
    p_a = np.where(rng.random(n) < 0.70, y, rng.integers(0, 8, n))
    flip = rng.random(n) < 0.08
    p_b = np.where(flip, rng.integers(0, 8, n), p_a)

    _, lo, hi = paired_bootstrap_delta(y, p_a, p_b, n_boot=400, alpha=0.05, seed=0)
    paired_width = hi - lo

    # unpaired: independent resamples per arm, differenced
    rng2 = np.random.default_rng(0)
    diffs = np.empty(400)
    for b in range(400):
        ia = rng2.integers(0, n, n)
        ib = rng2.integers(0, n, n)
        diffs[b] = fast_balanced_accuracy(y[ia], p_a[ia], 8) - fast_balanced_accuracy(
            y[ib], p_b[ib], 8
        )
    ulo, uhi = np.quantile(diffs, [0.025, 0.975])
    assert paired_width < (uhi - ulo)
