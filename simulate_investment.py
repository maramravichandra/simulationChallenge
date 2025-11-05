"""simulate_investment.py

Simulate the yearly coin-flip investment game described in the Quarto doc.

Rules:
- Start with an initial balance (default $30,000).
- Each year flip a fair coin:
  - Heads: multiply balance by 1.5 (increase by 50%)
  - Tails: multiply balance by 0.6 (decrease by 40%)
- Play annually until age `end_age` (default 75). Use a start_age to compute number of periods.

This script provides a function `simulate_investment` and a small CLI for quick runs.
"""
from __future__ import annotations
import argparse
from typing import Optional, Tuple
import numpy as np
import pandas as pd


def simulate_investment(
    n_sim: int = 50,
    start_age: int = 30,
    end_age: int = 75,
    initial: float = 30000.0,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate the investment game.

    Parameters
    ----------
    n_sim : int
        Number of independent simulation paths to run (default 50).
    start_age : int
        Age at buy-in (default 30). Simulation runs from start_age+1 through end_age inclusive.
    end_age : int
        Final age to stop playing (default 75).
    initial : float
        Initial account balance at buy-in (default 30000.0).
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    (trajectories, summary)
    - trajectories : DataFrame with columns ['sim', 'age', 'balance']
    - summary : DataFrame with one row per sim and final balance and summary stats
    """
    rng = np.random.default_rng(seed)
    periods = end_age - start_age
    # Pre-allocate arrays
    sim_ids = []
    ages = []
    balances = []

    final_balances = np.empty(n_sim, dtype=float)

    for s in range(n_sim):
        bal = float(initial)
        # record starting balance at start_age
        sim_ids.append(s + 1)
        ages.append(start_age)
        balances.append(bal)

        for t in range(1, periods + 1):
            age = start_age + t
            # fair coin: 1=heads (prob 0.5), 0=tails
            flip = rng.integers(0, 2)
            if flip == 1:
                bal *= 1.5
            else:
                bal *= 0.6
            sim_ids.append(s + 1)
            ages.append(age)
            balances.append(bal)

        final_balances[s] = bal

    traj = pd.DataFrame({"sim": sim_ids, "age": ages, "balance": balances})
    summary = pd.DataFrame({"sim": np.arange(1, n_sim + 1), "final_balance": final_balances})
    summary["gain_pct"] = (summary["final_balance"] - initial) / initial * 100.0
    return traj, summary


def summarize_summary(summary: pd.DataFrame, initial: float = 30000.0) -> None:
    """Print simple summary statistics for the simulations."""
    mean = summary["final_balance"].mean()
    median = summary["final_balance"].median()
    p_above = (summary["final_balance"] > initial).mean()
    p_double = (summary["final_balance"] >= 2 * initial).mean()
    print(f"Simulations: {len(summary)}")
    print(f"Mean final balance: ${mean:,.2f}")
    print(f"Median final balance: ${median:,.2f}")
    print(f"P(final > ${initial:,.0f}): {p_above:.3f}")
    print(f"P(final >= ${2*initial:,.0f}): {p_double:.3f}")


def _parse_args():
    p = argparse.ArgumentParser(description="Simulate the yearly coin-flip investment game")
    p.add_argument("--n", type=int, default=50, help="Number of simulations (default 50)")
    p.add_argument("--start-age", type=int, default=30, help="Starting age at buy-in (default 30)")
    p.add_argument("--end-age", type=int, default=75, help="End age (default 75)")
    p.add_argument("--initial", type=float, default=30000.0, help="Initial balance (default 30000)")
    p.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    p.add_argument("--save-traj", type=str, default=None, help="Optional path to save trajectories CSV")
    p.add_argument("--save-summary", type=str, default=None, help="Optional path to save summary CSV")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    traj, summary = simulate_investment(
        n_sim=args.n, start_age=args.start_age, end_age=args.end_age, initial=args.initial, seed=args.seed
    )
    summarize_summary(summary, initial=args.initial)
    # show first few final balances
    print("\nFirst 10 final balances")
    print(summary.head(10).to_string(index=False))
    if args.save_traj:
        traj.to_csv(args.save_traj, index=False)
        print(f"Saved trajectories to {args.save_traj}")
    if args.save_summary:
        summary.to_csv(args.save_summary, index=False)
        print(f"Saved summary to {args.save_summary}")
