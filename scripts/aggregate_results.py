"""Phase 6 - turn results/ into the headline table, with bootstrap CIs.

Reads every run's config, reloads its saved *_final.pt weights, recomputes
predictions on the test silos, and emits:

  1. the headline table: rows = centres 0-5, pooled union, mean-over-centres;
     columns = local, fedavg, best fed variant, pooled -- mean +/- std over
     seeds, with a 95% percentile-bootstrap CI per cell;
  2. gap-closed %: how much of the pooled-minus-local headroom federation
     recovers, per centre;
  3. rare-class recall (classes under --rare-prevalence of pooled train) per
     arm, since balanced accuracy averages those away.

Numbers are recomputed from weights rather than copied from the training CSVs,
so the report is an independent check on them, not a restatement. Predictions
are cached under results/_cache keyed by checkpoint mtime -- the first run
needs a GPU pass per run, reruns are instant.

Reporting policy (results/PHASE3_SUMMARY.md, fixed in Phase 3):
  - final epoch / final round, never the best -- selecting the best round on
    the test set would be test-set model selection;
  - the local arm is reported in BOTH aggregations, always labeled: routed
    union (deployable: each centre served by its own model) and
    mean-over-centres (comparable: every centre weighted equally).

Usage:
    python scripts/aggregate_results.py --mode finetune
    python scripts/aggregate_results.py --mode probe --n-boot 2000
"""

import argparse
import glob
import json
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

warnings.filterwarnings("ignore")

from fedisic.data import (  # noqa: E402
    NUM_CLASSES,
    build_feature_silos,
    build_image_silos,
    make_loader,
    pooled_label_counts,
)
from fedisic.evaluate import (  # noqa: E402
    bootstrap_balanced_accuracy,
    collect_predictions,
    per_class_recall,
)
from fedisic.models import LinearProbe, build_finetune_model  # noqa: E402
from fedisic.utils import resolve_device  # noqa: E402

NUM_CLIENTS = 6


# ------------------------------- discovery ---------------------------------- #

def discover_runs(out, mode, exclude):
    """Every run in `out` matching `mode`, from its config sidecar."""
    runs = []
    for cfg_path in sorted(glob.glob(os.path.join(out, "*_config.json"))):
        with open(cfg_path) as f:
            cfg = json.load(f)
        if cfg.get("mode") != mode:
            continue
        name = os.path.basename(cfg_path)[: -len("_config.json")]
        if any(pat in name for pat in exclude):
            continue
        cfg["_name"] = name
        runs.append(cfg)
    return runs


def arm_key(cfg):
    """Column this run belongs to: 'pooled', 'local', or the fed strategy."""
    arm = cfg["arm"]
    return arm if arm in ("pooled", "local") else cfg["strategy"].lower()


# ------------------------------ predictions --------------------------------- #

def _ckpt_paths(cfg, out):
    name = cfg["_name"]
    if cfg["arm"] == "local":
        return [os.path.join(out, f"{name}_c{c}_final.pt") for c in range(NUM_CLIENTS)]
    return [os.path.join(out, f"{name}_final.pt")]


def _build_model(cfg):
    return LinearProbe() if cfg["mode"] == "probe" else build_finetune_model()


def predictions(cfg, out, loaders, device, cache_dir):
    """{centre: (y_true, y_pred)} for this run, cached by checkpoint mtime.

    pooled/fed use one global model on every centre. local is *routed*: each
    centre is scored by its own silo's model, which is the only deployable
    reading of a per-silo arm.
    """
    paths = _ckpt_paths(cfg, out)
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        names = ", ".join(os.path.basename(m) for m in missing)
        return None, f"missing checkpoint(s): {names}"

    stamp = int(max(os.path.getmtime(p) for p in paths))
    cache = os.path.join(cache_dir, f"{cfg['_name']}_{stamp}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return {c: (z[f"y{c}"], z[f"p{c}"]) for c in range(NUM_CLIENTS)}, None

    per_centre = {}
    if cfg["arm"] == "local":
        for cid in range(NUM_CLIENTS):
            m = _build_model(cfg)
            m.load_state_dict(torch.load(paths[cid], map_location="cpu")["model"])
            m.to(device).eval()
            per_centre[cid] = collect_predictions(m, loaders[cid], device)
            del m
    else:
        m = _build_model(cfg)
        m.load_state_dict(torch.load(paths[0], map_location="cpu")["model"])
        m.to(device).eval()
        for cid in range(NUM_CLIENTS):
            per_centre[cid] = collect_predictions(m, loaders[cid], device)
        del m
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    os.makedirs(cache_dir, exist_ok=True)
    arrays = {f"y{c}": per_centre[c][0] for c in per_centre}
    arrays.update({f"p{c}": per_centre[c][1] for c in per_centre})
    np.savez(cache, **arrays)
    return per_centre, None


# -------------------------------- metrics ----------------------------------- #

def score_run(per_centre, n_boot, alpha, seed):
    """Balanced accuracy + bootstrap CI per centre, for the routed union, and
    the mean over centres."""
    out = {}
    for cid, (y, p) in per_centre.items():
        out[f"c{cid}"] = bootstrap_balanced_accuracy(
            y, p, n_boot=n_boot, alpha=alpha, seed=seed
        )
    order = sorted(per_centre)
    Y = np.concatenate([per_centre[c][0] for c in order])
    P = np.concatenate([per_centre[c][1] for c in order])
    out["union"] = bootstrap_balanced_accuracy(Y, P, n_boot=n_boot, alpha=alpha, seed=seed)
    centres = [out[f"c{c}"][0] for c in range(NUM_CLIENTS)]
    out["mean_centres"] = (float(np.mean(centres)), float("nan"), float("nan"))
    out["_pooled_preds"] = (Y, P)
    out["_per_centre"] = per_centre
    return out


def fast_balanced_accuracy(y, p, num_classes=NUM_CLASSES):
    """Mean recall over the classes present in y_true.

    Same definition as sklearn's balanced_accuracy_score (asserted in the
    tests), but built from two bincounts so the bootstrap can afford tens of
    thousands of evaluations.
    """
    total = np.bincount(y, minlength=num_classes)
    correct = np.bincount(y[y == p], minlength=num_classes)
    present = total > 0
    return float((correct[present] / total[present]).mean())


def paired_bootstrap_delta(y, p_a, p_b, n_boot, alpha, seed, num_classes=NUM_CLASSES):
    """95% CI on bal_acc(a) - bal_acc(b) from ONE index resample per replicate.

    The two arms are scored on identical test images, so their errors are
    correlated and their marginal CIs are not independent. Overlapping
    marginals therefore say nothing about whether the arms differ. Resampling
    the images once per replicate and scoring both arms on that same resample
    cancels the shared component, which is what makes the interval on the
    difference a valid test: significant iff it excludes 0.
    """
    n = len(y)
    point = fast_balanced_accuracy(y, p_a, num_classes) - fast_balanced_accuracy(
        y, p_b, num_classes
    )
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        ys = y[idx]
        stats[b] = fast_balanced_accuracy(ys, p_a[idx], num_classes) - \
            fast_balanced_accuracy(ys, p_b[idx], num_classes)
    lo, hi = np.quantile(stats, [alpha / 2.0, 1.0 - alpha / 2.0])
    return point, float(lo), float(hi)


def paired_bootstrap_delta_stratified(pc_a, pc_b, n_boot, alpha, seed,
                                      num_classes=NUM_CLASSES):
    """Paired CI on the difference in mean-over-centres balanced accuracy.

    Mean-over-centres weights every centre equally, so the resample must too:
    each centre is resampled within itself, both arms are scored per centre on
    that centre's resample, the per-centre scores are averaged, and only then
    differenced. Pooling the centres into one draw would let the largest silo
    dominate a statistic defined to be size-blind.
    """
    cids = sorted(pc_a)
    point = float(np.mean([fast_balanced_accuracy(*pc_a[c], num_classes) for c in cids]) -
                  np.mean([fast_balanced_accuracy(*pc_b[c], num_classes) for c in cids]))
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        va, vb = [], []
        for c in cids:
            y, p_a = pc_a[c]
            _, p_b = pc_b[c]
            idx = rng.integers(0, len(y), len(y))
            ys = y[idx]
            va.append(fast_balanced_accuracy(ys, p_a[idx], num_classes))
            vb.append(fast_balanced_accuracy(ys, p_b[idx], num_classes))
        stats[b] = np.mean(va) - np.mean(vb)
    lo, hi = np.quantile(stats, [alpha / 2.0, 1.0 - alpha / 2.0])
    return point, float(lo), float(hi)


def agg_over_seeds(values):
    """mean +/- std over seeds. One seed is reported as itself, never +/- 0.000."""
    v = np.array(values, dtype=float)
    if len(v) == 1:
        return float(v[0]), None, 1
    return float(v.mean()), float(v.std(ddof=1)), len(v)


def fmt_cell(mean, std, ci=None):
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "--"
    s = f"{mean:.4f}"
    if std is not None:
        s += f" +/- {std:.4f}"
    if ci is not None and not np.isnan(ci[0]):
        s += f" [{ci[0]:.3f}, {ci[1]:.3f}]"
    return s


def gap_closed(local, fed, pooled):
    """(fed - local) / (pooled - local), as a percentage.

    Undefined when pooled does not beat local: with a non-positive denominator
    the ratio flips sign or explodes and stops meaning "share of the headroom
    recovered", so it is reported as n/a rather than as a number.
    """
    if local is None or fed is None or pooled is None:
        return None
    denom = pooled - local
    if denom <= 1e-6:
        return None
    return 100.0 * (fed - local) / denom


def rare_classes(silos, threshold):
    counts = pooled_label_counts(silos).astype(float)
    prev = counts / counts.sum()
    return [k for k in range(NUM_CLASSES) if prev[k] < threshold], prev


# --------------------------------- report ----------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mode", default="finetune", choices=["probe", "finetune"])
    ap.add_argument("--out", default="results")
    ap.add_argument("--features-dir", default="data/features")
    ap.add_argument("--cache-dir", default=None, help="HF datasets cache dir")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=0.05, help="1-alpha CI (0.05 -> 95%%)")
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--rare-prevalence", type=float, default=0.02)
    ap.add_argument("--exclude", nargs="*", default=["lrsweep"],
                    help="substrings of run names to skip (default: hyperparameter sweeps)")
    ap.add_argument("--report", default=None, help="write the markdown report here")
    args = ap.parse_args()

    device = resolve_device(args.device)
    runs = discover_runs(args.out, args.mode, args.exclude)
    if not runs:
        sys.exit(f"No {args.mode} runs found in {args.out}/ (excluding {args.exclude}).")

    silos = (build_feature_silos(args.features_dir) if args.mode == "probe"
             else build_image_silos(cache_dir=args.cache_dir, augment_train=False))
    loaders = {s.id: make_loader(s.test_ds, args.batch_size, shuffle=False,
                                 num_workers=0, device=device) for s in silos}
    test_n = {s.id: len(s.test_ds) for s in silos}

    lines = []

    def emit(line=""):
        print(line)
        lines.append(line)

    emit(f"# Phase 6 aggregate - {args.mode} arms")
    emit()
    emit(f"Recomputed from saved weights on {device}; {args.n_boot} bootstrap "
         f"resamples, {100 * (1 - args.alpha):.0f}% percentile CIs.")
    emit()

    scored, skipped = {}, []
    for cfg in runs:
        per_centre, err = predictions(
            cfg, args.out, loaders, device, os.path.join(args.out, "_cache")
        )
        if err:
            skipped.append((cfg["_name"], err))
            continue
        scored.setdefault(arm_key(cfg), []).append(
            (cfg, score_run(per_centre, args.n_boot, args.alpha, args.boot_seed))
        )

    emit("## Runs included")
    emit()
    emit("| arm | run | seed | epochs/rounds |")
    emit("|---|---|---|---|")
    for arm in sorted(scored):
        for cfg, _ in scored[arm]:
            sched = cfg["rounds"] if cfg["arm"] == "fed" else cfg["epochs"]
            emit(f"| {arm} | `{cfg['_name']}` | {cfg['seed']} | {sched} |")
    emit()
    if skipped:
        emit("Skipped (no usable checkpoint):")
        emit()
        for name, err in skipped:
            emit(f"- `{name}` - {err}")
        emit()

    fed_arms = [a for a in scored if a not in ("pooled", "local")]
    best_fed = None
    if fed_arms:
        best_fed = max(fed_arms,
                       key=lambda a: np.mean([s["union"][0] for _, s in scored[a]]))

    cols = [("local", "local (routed)"), ("fedavg", "fedavg")]
    if best_fed and best_fed != "fedavg":
        cols.append((best_fed, f"best fed ({best_fed})"))
    cols.append(("pooled", "pooled"))
    cols = [(k, lab) for k, lab in cols if k in scored]

    # Paired deltas are computed before the headline table, because gap-closed
    # needs to know whether its own denominator is distinguishable from zero.
    pairs = []
    if "pooled" in scored and "local" in scored:
        pairs.append(("pooled", "local"))
    for fa in sorted(fed_arms):
        if "local" in scored:
            pairs.append((fa, "local"))
        if "pooled" in scored:
            pairs.append((fa, "pooled"))

    row_keys = [f"c{c}" for c in range(NUM_CLIENTS)] + ["union", "mean_centres"]
    paired = {}
    for a, b in pairs:
        pc_a = scored[a][0][1]["_per_centre"]
        pc_b = scored[b][0][1]["_per_centre"]
        for cid in range(NUM_CLIENTS):
            y_a, p_a = pc_a[cid]
            y_b, p_b = pc_b[cid]
            if not np.array_equal(y_a, y_b):
                sys.exit(f"centre {cid}: {a} and {b} were scored on different "
                         f"labels; pairing them would be invalid.")
            paired[(a, b, f"c{cid}")] = paired_bootstrap_delta(
                y_a, p_a, p_b, args.n_boot, args.alpha, args.boot_seed
            )
        Y_a, P_a = scored[a][0][1]["_pooled_preds"]
        _, P_b = scored[b][0][1]["_pooled_preds"]
        paired[(a, b, "union")] = paired_bootstrap_delta(
            Y_a, P_a, P_b, args.n_boot, args.alpha, args.boot_seed
        )
        paired[(a, b, "mean_centres")] = paired_bootstrap_delta_stratified(
            pc_a, pc_b, args.n_boot, args.alpha, args.boot_seed
        )

    def headroom_is_significant(row):
        """Is pooled meaningfully above local at this row?

        gap-closed divides by pooled - local. If that headroom is itself
        indistinguishable from zero the quotient is noise over noise: a tiny
        denominator swings it to +-hundreds of percent on a difference the data
        cannot even establish. Report n/a instead of a number that will be read
        as a result.
        """
        key = ("pooled", "local", row)
        if key not in paired:
            return False
        d, lo, hi = paired[key]
        return d > 0 and lo > 0

    def cell(arm, row):
        if arm not in scored:
            return None, None, None
        vals = [s[row][0] for _, s in scored[arm]]
        mean, std, n = agg_over_seeds(vals)
        ci = scored[arm][0][1][row][1:] if n == 1 else None
        return mean, std, ci

    emit("## Headline table")
    emit()
    emit("Balanced accuracy, final epoch/round. `[lo, hi]` = 95% bootstrap CI "
         "(shown when a single seed makes a std undefined).")
    emit()
    emit("These are **per-cell** uncertainties: each says how precisely that "
         "one number is measured. They are **not** a way to compare two "
         "columns - the arms share test images, so their intervals share noise "
         "and overlap carries no verdict. Comparisons live in the paired "
         "section below.")
    emit()
    emit("| row | test n | " + " | ".join(lab for _, lab in cols) + " | gap-closed |")
    emit("|" + "---|" * (len(cols) + 3))

    rows = [(f"c{c}", f"centre {c}", test_n[c]) for c in range(NUM_CLIENTS)]
    rows += [("union", "**pooled union**", sum(test_n.values())),
             ("mean_centres", "**mean over centres**", sum(test_n.values()))]
    for row, label, n in rows:
        cells, byarm = [], {}
        for arm, _ in cols:
            mean, std, ci = cell(arm, row)
            byarm[arm] = mean
            cells.append(fmt_cell(mean, std, ci))
        gc = gap_closed(byarm.get("local"),
                        byarm.get(best_fed) if best_fed else None,
                        byarm.get("pooled"))
        if gc is None:
            cells.append("n/a")
        elif not headroom_is_significant(row):
            cells.append(f"({gc:.1f}%) n.s.")
        else:
            cells.append(f"{gc:.1f}%")
        emit(f"| {label} | {n} | " + " | ".join(cells) + " |")
    emit()

    if best_fed:
        emit(f"gap-closed = (fed - local) / (pooled - local), using **{best_fed}**. "
             f"It reads as the share of the local-to-pooled headroom that "
             f"federation recovers at that centre. Values outside 0-100% are "
             f"meaningful, not errors: **above 100%** means federation beat "
             f"centralised training there, **negative** means it landed below "
             f"that centre's own local model. It is **n/a** wherever pooled did "
             f"not beat local, because the headroom it normalises by is then "
             f"zero or negative and the ratio stops meaning anything. A value "
             f"in parentheses marked **n.s.** is worse than n/a: pooled leads "
             f"local numerically, but by an amount the paired test cannot "
             f"distinguish from zero, so the quotient is noise divided by "
             f"noise and its magnitude carries no information.")
        emit()

    if "local" in scored:
        lu = cell("local", "union")[0]
        lm = cell("local", "mean_centres")[0]
        emit(f"Local is reported in both aggregations per the Phase 3 policy: "
             f"routed union **{lu:.4f}**, mean over centres **{lm:.4f}**. The "
             f"union is size-weighted, so the largest silo dominates it; the "
             f"mean weights every centre equally.")
        emit()

    emit("## Paired comparisons")
    emit()
    if not pairs:
        emit("Fewer than two arms present; nothing to compare.")
        emit()
    else:
        emit("Every arm is scored on the **same** test images, so the marginal "
             "CIs above share their noise and their overlap is not a test - two "
             "arms can differ significantly with overlapping marginals, and can "
             "fail to differ with disjoint ones. Each replicate below draws one "
             "index resample and scores both arms on it, so the shared component "
             "cancels. **A difference is significant iff its CI excludes 0.** "
             "Mean-over-centres resamples within each centre so that every "
             "centre keeps equal weight.")
        emit()
        if any(len(scored[a]) > 1 for a, _ in pairs):
            emit("Where an arm has several seeds the first is used, since pairing "
                 "requires one prediction vector per arm.")
            emit()
        labels = {**{f"c{c}": f"centre {c}" for c in range(NUM_CLIENTS)},
                  "union": "**pooled union**", "mean_centres": "**mean over centres**"}
        emit("| comparison | row | delta | 95% CI | significant |")
        emit("|---|---|---|---|---|")
        n_sig = 0
        for a, b in pairs:
            for row in row_keys:
                d, lo, hi = paired[(a, b, row)]
                sig = lo > 0 or hi < 0
                n_sig += int(sig)
                emit(f"| {a} - {b} | {labels[row]} | {d:+.4f} | "
                     f"[{lo:+.4f}, {hi:+.4f}] | {'**yes**' if sig else 'no'} |")
        emit()
        emit(f"{n_sig} of {len(pairs) * len(row_keys)} comparisons are "
             f"significant at the 95% level.")
        emit()

    rare, prev = rare_classes(silos, args.rare_prevalence)
    emit(f"## Rare-class recall (< {args.rare_prevalence:.0%} of pooled train)")
    emit()
    if not rare:
        emit("No class falls below the threshold.")
        emit()
    else:
        emit("Balanced accuracy averages these away; a model can look fine "
             "overall and still miss them entirely.")
        emit()
        emit("| arm | " + " | ".join(f"class {k} ({prev[k]:.1%})" for k in rare)
             + " | mean rare | mean common |")
        emit("|" + "---|" * (len(rare) + 3))
        common = [k for k in range(NUM_CLASSES) if k not in rare]
        for arm, lab in cols:
            per_seed = []
            for _, s in scored[arm]:
                Y, P = s["_pooled_preds"]
                per_seed.append(per_class_recall(Y, P, NUM_CLASSES))
            rec = np.mean(per_seed, axis=0)
            emit(f"| {lab} | " + " | ".join(f"{rec[k]:.3f}" for k in rare)
                 + f" | {rec[rare].mean():.3f} | {rec[common].mean():.3f} |")
        emit()

    emit("## Caveats")
    emit()
    n_seeds = {a: len(v) for a, v in scored.items()}
    single = sorted(a for a, n in n_seeds.items() if n == 1)
    if single:
        emit(f"- **Single seed** for: {', '.join(single)}. No std over seeds is "
             f"computable, so the bootstrap CI is the only uncertainty estimate "
             f"here - and it captures test-set sampling only, not run-to-run "
             f"variance from initialisation, augmentation, and client order.")
    if not fed_arms:
        emit("- **No federated arm present**, so gap-closed % is undefined: it "
             "measures how much of the pooled-minus-local headroom federation "
             "recovers, and there is nothing here to recover it. The pooled and "
             "local columns are the two endpoints only.")
    small = [c for c in range(NUM_CLIENTS) if test_n[c] < 200]
    if small:
        sizes = ", ".join(f"c{c}={test_n[c]}" for c in small)
        emit(f"- **Small test sets**: centre(s) {', '.join(f'c{c}' for c in small)} "
             f"have under 200 test images ({sizes}). Their CIs are wide and their "
             f"point estimates should not be read as measurements.")
    emit("- Per-centre balanced accuracy also moves epoch to epoch within the "
         "converged region, which the bootstrap does not capture: it resamples "
         "one fixed set of predictions. Treat a per-centre difference as real "
         "only if it clears both the CI and that epoch-to-epoch scatter.")
    emit()

    path = args.report or os.path.join(args.out, f"AGGREGATE_{args.mode}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n-> wrote {path}")


if __name__ == "__main__":
    main()
