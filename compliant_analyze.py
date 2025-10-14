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
            force_applied.append(_to_float_or_nan(row.get("force_applied", "")))

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
    }


def plot_log(data: dict, out_path: str | None, show: bool, n: int = None) -> None:
    """
    Plot log up to first n samples (if n is given), else plot all.
    """
    # Determine slice
    if n is not None:
        t = data["time"][:n]
        tendon_length = data["tendon_length"][:n]
        l_ce = data["l_ce"][:n]
        l_se = data["l_se"][:n]
        tendon_velocity = data["tendon_velocity"][:n]
        v_ce = data["v_ce"][:n]
        F_mtu = data["F_mtu"][:n]
        force_applied = data["force_applied"][:n]
        l_se_plot = data["l_se"][:n]
    else:
        t = data["time"]
        tendon_length = data["tendon_length"]
        l_ce = data["l_ce"]
        l_se = data["l_se"]
        tendon_velocity = data["tendon_velocity"]
        v_ce = data["v_ce"]
        F_mtu = data["F_mtu"]
        force_applied = data["force_applied"]
        l_se_plot = data["l_se"]

    fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    axs[0].plot(t, tendon_length, label="tendon_length")
    axs[0].plot(t, l_ce, label="l_ce")
    axs[0].plot(t, l_se, label="l_se")
    axs[0].legend(); axs[0].set_ylabel("length [m]")

    axs[1].plot(t, tendon_velocity, label="tendon_velocity")
    axs[1].plot(t, v_ce, label="v_ce")
    axs[1].legend(); axs[1].set_ylabel("velocity [m/s]")

    axs[2].plot(t, F_mtu, label="F_mtu")
    axs[2].plot(t, force_applied, label="force_applied")
    axs[2].legend(); axs[2].set_ylabel("force [N]")

    if np.max(np.abs(l_se_plot)) > 0:
        axs[3].plot(t, l_se_plot/np.max(np.abs(l_se_plot)), label="l_se norm")
        axs[3].legend()
    axs[3].set_xlabel("time [s]")

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
    plot_log(data, args.out, args.show, n=-2)


if __name__ == "__main__":
    main()


