#!/usr/bin/env python3
"""
Analyze and plot compliant MTU logs produced by MuJoCo engine.

Usage:
  python compliant_analyze.py --log compliant_mtu_log.csv --out compliant_mtu_log_plots.png --show
"""

import argparse
import csv
import os
from typing import List

import numpy as np
import matplotlib.pyplot as plt


def _to_float_or_nan(s: str) -> float:
    s = s.strip() if isinstance(s, str) else s
    if s is None or s == "":
        return float('nan')
    return float(s)


def _to_int_or_neg1(s: str) -> int:
    s = s.strip() if isinstance(s, str) else s
    if s is None or s == "":
        return -1
    return int(s)


def load_log_csv(path: str):
    times: List[float] = []
    actuator_id: List[int] = []
    ctrl: List[float] = []
    act: List[float] = []
    tendon_length: List[float] = []
    tendon_velocity: List[float] = []
    l_ce: List[float] = []
    v_ce: List[float] = []
    l_se: List[float] = []
    F_mtu: List[float] = []
    force_applied: List[float] = []
    F_max_row: List[float] = []
    l_opt_row: List[float] = []
    l_slack_row: List[float] = []
    v_max_row: List[float] = []
    # Detailed columns
    l_ce0: List[float] = []
    l_se0: List[float] = []
    f_se0: List[float] = []
    f_be0: List[float] = []
    f_pe0: List[float] = []
    f_lce0: List[float] = []
    fvce_denom: List[float] = []
    f_vce0: List[float] = []
    v_ce0: List[float] = []

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(_to_float_or_nan(row.get("time", "")))
            actuator_id.append(_to_int_or_neg1(row.get("actuator_id", "")))
            ctrl.append(_to_float_or_nan(row.get("ctrl", "")))
            act.append(_to_float_or_nan(row.get("act", "")))
            tendon_length.append(_to_float_or_nan(row.get("tendon_length", "")))
            tendon_velocity.append(_to_float_or_nan(row.get("tendon_velocity", "")))
            l_ce.append(_to_float_or_nan(row.get("l_ce", "")))
            v_ce.append(_to_float_or_nan(row.get("v_ce", "")))
            l_se.append(_to_float_or_nan(row.get("l_se", "")))
            F_mtu.append(_to_float_or_nan(row.get("F_mtu", "")))
            # detailed values are optional; handle missing as NaN
            l_ce0.append(_to_float_or_nan(row.get("l_ce0", "")))
            l_se0.append(_to_float_or_nan(row.get("l_se0", "")))
            f_se0.append(_to_float_or_nan(row.get("f_se0", "")))
            f_be0.append(_to_float_or_nan(row.get("f_be0", "")))
            f_pe0.append(_to_float_or_nan(row.get("f_pe0", "")))
            f_lce0.append(_to_float_or_nan(row.get("f_lce0", "")))
            fvce_denom.append(_to_float_or_nan(row.get("fvce_denom", "")))
            f_vce0.append(_to_float_or_nan(row.get("f_vce0", "")))
            v_ce0.append(_to_float_or_nan(row.get("v_ce0", "")))
            force_applied.append(_to_float_or_nan(row.get("force_applied", "")))
            F_max_row.append(_to_float_or_nan(row.get("F_max", "")))
            l_opt_row.append(_to_float_or_nan(row.get("l_opt", "")))
            l_slack_row.append(_to_float_or_nan(row.get("l_slack", "")))
            v_max_row.append(_to_float_or_nan(row.get("v_max", "")))

    return {
        "time": np.array(times),
        "actuator_id": np.array(actuator_id),
        "ctrl": np.array(ctrl),
        "act": np.array(act),
        "tendon_length": np.array(tendon_length),
        "tendon_velocity": np.array(tendon_velocity),
        "l_ce": np.array(l_ce),
        "v_ce": np.array(v_ce),
        "l_se": np.array(l_se),
        "F_mtu": np.array(F_mtu),
        "force_applied": np.array(force_applied),
        "l_ce0": np.array(l_ce0),
        "l_se0": np.array(l_se0),
        "f_se0": np.array(f_se0),
        "f_be0": np.array(f_be0),
        "f_pe0": np.array(f_pe0),
        "f_lce0": np.array(f_lce0),
        "fvce_denom": np.array(fvce_denom),
        "f_vce0": np.array(f_vce0),
        "v_ce0": np.array(v_ce0),
        # parameters appended to each row end
        "F_max": np.array(F_max_row),
        "l_opt": np.array(l_opt_row),
        "l_slack": np.array(l_slack_row),
        "v_max": np.array(v_max_row),
    }


def plot_log(data: dict, out_path: str | None, show: bool, n: int = None) -> None:
    """Plot log up to first n samples (if n is given), else plot all, aligning lengths per subplot."""

    def compute_len(base, *arrays):
        lengths = [len(base)] + [len(a) for a in arrays]
        L = min(lengths)
        if n is not None:
            if n >= 0:
                L = min(L, n)
            else:
                L = max(0, L + n)
        return L

    fig, axs = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

    # Panel 1: lengths
    t = data["time"]
    L = compute_len(t, data["tendon_length"], data["l_ce"], data["l_se"])
    axs[0].plot(t[:L], data["tendon_length"][:L], label="tendon_length")
    axs[0].plot(t[:L], data["l_ce"][:L], label="l_ce")
    axs[0].plot(t[:L], data["l_se"][:L], label="l_se")
    axs[0].legend(); axs[0].set_ylabel("length [m]")

    # Panel 2: velocities
    L = compute_len(t, data["tendon_velocity"], data["v_ce"])
    axs[1].plot(t[:L], data["tendon_velocity"][:L], label="tendon_velocity")
    axs[1].plot(t[:L], data["v_ce"][:L], label="v_ce")
    axs[1].axhline(0.0, color='k', linewidth=1, linestyle='--')
    axs[1].legend(); axs[1].set_ylabel("velocity [m/s]")

    # Panel 3: forces
    L = compute_len(t, data["F_mtu"], data["force_applied"])
    axs[2].plot(t[:L], data["F_mtu"][:L], label="F_mtu")
    axs[2].plot(t[:L], data["force_applied"][:L], label="force_applied")
    axs[2].axhline(0.0, color='k', linewidth=1, linestyle='--')
    axs[2].legend(); axs[2].set_ylabel("force [N]")

    # Panel 4: normalized l_se
    L = compute_len(t, data["l_se"]) 
    l_se_slice = data["l_se"][:L]
    if l_se_slice.size and np.max(np.abs(l_se_slice)) > 0:
        axs[3].plot(t[:L], l_se_slice/np.max(np.abs(l_se_slice)), label="l_se norm")
        axs[3].legend()
    axs[3].set_xlabel("time [s]")

    # Panel 5: Detailed FVCE internals (align lengths)
    internals = [
        data.get("f_se0", np.array([])),
        data.get("f_be0", np.array([])),
        data.get("f_pe0", np.array([])),
        data.get("f_lce0", np.array([])),
        data.get("f_vce0", np.array([])),
    ]
    valid = [arr for arr in internals if arr.size]
    if valid:
        L = compute_len(t, *valid)
        labels = ["f_se0","f_be0","f_pe0","f_lce0","f_vce0"]
        for arr, label in zip(internals, labels):
            if arr.size:
                axs[4].plot(t[:L], arr[:L], label=label)
    axs[4].axhline(0.0, color='k', linewidth=1, linestyle='--')
    axs[4].legend(); axs[4].set_ylabel("FL/FV terms")

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze compliant MTU CSV log")
    parser.add_argument("--log", default="compliant_mtu_log.csv", help="Path to CSV log")
    parser.add_argument("--out", default="compliant_mtu_log_plots.png", help="Output image path")
    parser.add_argument("--show", default=True, action="store_true", help="Show interactive plots")
    args = parser.parse_args()

    if not os.path.exists(args.log):
        print(f"Log file not found: {args.log}")
        return

    data = load_log_csv(args.log)

    # Print v_ce derivation per row, with formulas and numeric substitution
    rows = len(data["time"])
    for i in range(rows):
        # guard missing fields
        try:
            t = data["time"][i]
            ctrl = data["ctrl"][i]
            act = data["act"][i]
            l_ce = data["l_ce"][i]
            l_se = data["l_se"][i]
            l_ce0 = data["l_ce0"][i]
            l_se0 = data["l_se0"][i]
            f_se0 = data["f_se0"][i]
            f_be0 = data["f_be0"][i]
            f_pe0 = data["f_pe0"][i]
            f_lce0 = data["f_lce0"][i]
            fvce_denom = data["fvce_denom"][i]
            f_vce0 = data["f_vce0"][i]
            v_ce0 = data["v_ce0"][i]
            v_ce = data["v_ce"][i]
            l_opt = data["l_opt"][i] if data.get("l_opt") is not None and len(data.get("l_opt"))>i else float('nan')
            l_slack = data["l_slack"][i] if data.get("l_slack") is not None and len(data.get("l_slack"))>i else float('nan')
            v_max = data["v_max"][i] if data.get("v_max") is not None and len(data.get("v_max"))>i else float('nan')
            # equations
            print(f"\n[t={t:.6f}] v_ce computation")
            print(f"  l_ce0 = l_ce / l_opt = {l_ce:.6f} / {l_opt:.6f} = {l_ce0:.6f}")
            print(f"  l_se0 = l_se / l_slack = {l_se:.6f} / {l_slack:.6f} = {l_se0:.6f}")
            print(f"  f_se0 = fp(l_se0) = {f_se0:.6f}")
            print(f"  f_be0 = fp_ext(l_ce0) = {f_be0:.6f}")
            print(f"  f_pe0 = fp(l_ce0) = {f_pe0:.6f}")
            print(f"  f_lce0 = fl(l_ce0) = {f_lce0:.6f}")
            print(f"  denom = f_pe0 + act * f_lce0 = {f_pe0:.6f} + {act:.6f} * {f_lce0:.6f} = {fvce_denom:.6f}")
            print(f"  f_vce0 = (f_se0 + f_be0) / denom = ({f_se0:.6f} + {f_be0:.6f}) / {fvce_denom:.6f} = {f_vce0:.6f}")
            print(f"  v_ce0 = inv_fvce(f_vce0) = {v_ce0:.6f}")
            print(f"  v_ce = l_opt * v_max * v_ce0 = {l_opt:.6f} * {v_max:.6f} * {v_ce0:.6f} = {l_opt*v_max*v_ce0 if np.isfinite(l_opt) and np.isfinite(v_max) and np.isfinite(v_ce0) else float('nan'):.6f} -> logged {v_ce:.6f}")
        except Exception:
            # skip incomplete rows
            continue

    plot_log(data, args.out, args.show, n=-4)


if __name__ == "__main__":
    main()


