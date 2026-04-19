#!/usr/bin/env python3
"""
FinMEM Ablation Study Runner — Train + Test Pipeline

For each configuration (Base, +Obj1, +Obj1+2, +Obj1+2+3, +Obj1+2+3+4):
  1. Train phase: populate memory with historical data
  2. Test phase:  make trading decisions using populated memory
  3. Collect the paper's 5 evaluation metrics

Supports parallel execution via --parallel flag.

Usage:
    # Run all configs sequentially (safe, less API load)
    python run_ablation.py

    # Run all configs in parallel (fast, more API load)
    python run_ablation.py --parallel

    # Test-only (skip training, e.g. if already trained)
    python run_ablation.py --skip-train

    # Single config only
    python run_ablation.py --only "Base FinMEM"
"""

import os
import sys
import subprocess
import re
import csv
import argparse
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── Configuration ──────────────────────────────────────────────
TICKER = "TSLA"
PYTHON = sys.executable  # uses the active .venv interpreter

# Train period (populate memory)
TRAIN_START = "2022-03-14"
TRAIN_END   = "2022-04-08"   # ~20 trading days

# Test period (blind trading)
TEST_START  = "2022-10-03"
TEST_END    = "2022-10-28"   # ~20 trading days

CHECKPOINT_DIR = "./checkpoints"

# Pre-built datasets (skip Yahoo Finance during simulation)
TRAIN_DATASET = "./datasets/tsla_train_paper.pkl"
TEST_DATASET  = "./datasets/tsla_test_paper.pkl"

CONFIGS = [
    {
        "name": "Base FinMEM",
        "script": "run.py",
        "ckp": "base",
        "env": {
            "ADAPTIVE_Q": "false",
            "LEARNED_IMPORTANCE": "false",
            "CROSS_TICKER": "false",
            "MULTIAGENT": "false",
        },
    },
    {
        "name": "FinMEM + Obj1",
        "script": "run_obj1.py",
        "ckp": "obj1",
        "env": {
            "ADAPTIVE_Q": "true",
            "LEARNED_IMPORTANCE": "false",
            "CROSS_TICKER": "false",
            "MULTIAGENT": "false",
        },
    },
    {
        "name": "FinMEM + Obj1+2",
        "script": "run_obj2.py",
        "ckp": "obj1_2",
        "env": {
            "ADAPTIVE_Q": "true",
            "LEARNED_IMPORTANCE": "true",
            "CROSS_TICKER": "false",
            "MULTIAGENT": "false",
        },
    },
    {
        "name": "FinMEM + Obj1+2+3",
        "script": "run_obj3.py",
        "ckp": "obj1_2_3",
        "env": {
            "ADAPTIVE_Q": "true",
            "LEARNED_IMPORTANCE": "true",
            "CROSS_TICKER": "true",
            "MULTIAGENT": "false",
        },
    },
    {
        "name": "FinMEM + Obj1+2+3+4",
        "script": "run_obj4.py",
        "ckp": "obj1_2_3_4",
        "env": {
            "ADAPTIVE_Q": "true",
            "LEARNED_IMPORTANCE": "true",
            "CROSS_TICKER": "true",
            "MULTIAGENT": "true",
        },
    },
]


# ── Metric Parsing ─────────────────────────────────────────────

def parse_simulation_output(output: str) -> dict:
    """Parse metrics from either the base simulator output or Obj4 output."""
    metrics = {}

    # ── Base simulator format (run.py / run_obj1 / run_obj2 / run_obj3) ──

    # Final Value:    $99,942.98
    m = re.search(r"Final Value:\s+\$?([\d,]+\.?\d*)", output)
    if m:
        metrics["Final Value ($)"] = m.group(1).replace(",", "")

    # Total Return:   $-57.02 (-0.06%)
    m = re.search(r"Total Return:\s+\$?[\d,\-+.]+\s+\(([+-]?\d+\.?\d*)%\)", output)
    if m:
        metrics["Total Return (%)"] = m.group(1)

    # Paper's 5 metrics — FinMEM column (first number after the label)
    metric_patterns = [
        ("Cumulative Return (%)",    r"Cum(?:ulative)?\.?\s*Return\s*\(%?\)?\s*:?\s*([\-+]?\d+\.?\d*)%"),
        ("Sharpe Ratio",             r"Sharpe Ratio\s*:?\s*([\-+]?\d+\.?\d+)"),
        ("Ann. Volatility",          r"Ann(?:ualized)?\.?\s*Volatility\s*:?\s*([\-+]?\d+\.?\d+)"),
        ("Daily Volatility",         r"Daily Volatility\s*:?\s*([\-+]?\d+\.?\d+)"),
        ("Max Drawdown (%)",         r"Max Drawdown\s*\(%?\)?\s*:?\s*([\-+]?\d+\.?\d*)%"),
    ]
    for label, pattern in metric_patterns:
        m = re.search(pattern, output)
        if m:
            metrics[label] = m.group(1)

    # Buy & Hold metrics (second number on the same line)
    bh_patterns = [
        ("BH Cum. Return (%)", r"Cum(?:ulative)?\.?\s*Return\s*\(%?\)?\s*:?\s*[\-+]?\d+\.?\d*%\s+([\-+]?\d+\.?\d*)%"),
        ("BH Sharpe Ratio",    r"Sharpe Ratio\s*:?\s*[\-+]?\d+\.?\d+\s+([\-+]?\d+\.?\d+)"),
        ("BH Max Drawdown (%)",r"Max Drawdown\s*\(%?\)?\s*:?\s*[\-+]?\d+\.?\d*%\s+([\-+]?\d+\.?\d*)%"),
    ]
    for label, pattern in bh_patterns:
        m = re.search(pattern, output)
        if m:
            metrics[label] = m.group(1)

    # Days processed
    m = re.search(r"Days(?:\s+Processed)?:\s+(\d+)", output)
    if m:
        metrics["Days"] = m.group(1)

    # ── Obj4 format (run_obj4.py prints differently) ──
    # CR:            +7.37%
    if "Final Value ($)" not in metrics:
        m = re.search(r"Final:\s+\$?([\d,]+\.?\d*)", output)
        if m:
            metrics["Final Value ($)"] = m.group(1).replace(",", "")

    if "Total Return (%)" not in metrics:
        m = re.search(r"CR:\s+([+-]?\d+\.?\d*)%", output)
        if m:
            metrics["Total Return (%)"] = m.group(1)
            metrics["Cumulative Return (%)"] = m.group(1)

    if "BH Cum. Return (%)" not in metrics:
        m = re.search(r"Buy & Hold CR:\s+([+-]?\d+\.?\d*)%", output)
        if m:
            metrics["BH Cum. Return (%)"] = m.group(1)

    return metrics


def build_cmd(script: str, mode: str, ckp_name: str, ticker: str = TICKER) -> list:
    """Build the subprocess command for the given script and mode."""
    ckp_path = os.path.join(CHECKPOINT_DIR, f"{ticker}_{ckp_name}")

    if mode == "train":
        start, end = TRAIN_START, TRAIN_END
        extra = ["--save-checkpoint", ckp_path]
        dataset = TRAIN_DATASET
    else:
        start, end = TEST_START, TEST_END
        extra = ["--checkpoint", ckp_path]
        dataset = TEST_DATASET

    # Add dataset if file exists (skips Yahoo Finance download)
    dataset_args = []
    if os.path.exists(dataset):
        dataset_args = ["--dataset", dataset]

    # run_obj3.py uses --tickers instead of --ticker
    if script == "run_obj3.py":
        cmd = [PYTHON, script, "--tickers", ticker, "--mode", mode,
               "--start-date", start, "--end-date", end] + dataset_args + extra
    # run_obj4.py doesn't use --checkpoint or --dataset (uses yf.download directly)
    elif script == "run_obj4.py":
        cmd = [PYTHON, script, "--ticker", ticker, "--mode", mode,
               "--start-date", start, "--end-date", end]
    else:
        cmd = [PYTHON, script, "--ticker", ticker, "--mode", mode,
               "--start-date", start, "--end-date", end] + dataset_args + extra

    return cmd


def run_single_config(cfg: dict, skip_train: bool = False) -> dict:
    """Run a single configuration (train + test) and return parsed metrics."""
    name = cfg["name"]
    script = cfg["script"]
    ckp_name = cfg["ckp"]
    env_vars = cfg["env"]

    # Build env: copy current env + override feature flags + enable LLM cache
    run_env = os.environ.copy()
    run_env["LLM_CACHE"] = "true"
    for k, v in env_vars.items():
        run_env[k] = v

    result = {"Configuration": name, "Status": "OK"}
    raw_outputs = {}

    # ── Phase 1: Train ──
    if not skip_train:
        print(f"\n{'─'*60}")
        print(f"  🏋️ TRAIN: {name} ({script})")
        print(f"  Period: {TRAIN_START} → {TRAIN_END}")
        print(f"{'─'*60}")

        cmd = build_cmd(script, "train", ckp_name)
        try:
            proc = subprocess.run(
                cmd, env=run_env,
                capture_output=True, text=True, check=False,
                timeout=1800,  # 30 min max
            )
            raw_outputs["train"] = proc.stdout + "\n" + proc.stderr
            if proc.returncode != 0:
                print(f"  ⚠️ Train exited with code {proc.returncode}")
                # Show last few lines
                for line in raw_outputs["train"].strip().splitlines()[-10:]:
                    print(f"     | {line}")
            else:
                print(f"  ✅ Training complete")
        except subprocess.TimeoutExpired:
            print(f"  ⏰ Training timed out")
            result["Status"] = "TRAIN_TIMEOUT"
            return result
        except Exception as e:
            print(f"  💥 Training exception: {e}")
            result["Status"] = f"TRAIN_ERROR: {e}"
            return result

    # ── Phase 2: Test ──
    print(f"\n{'─'*60}")
    print(f"  🧪 TEST:  {name} ({script})")
    print(f"  Period: {TEST_START} → {TEST_END}")
    print(f"{'─'*60}")

    cmd = build_cmd(script, "test", ckp_name)
    try:
        proc = subprocess.run(
            cmd, env=run_env,
            capture_output=True, text=True, check=False,
            timeout=1800,
        )
        output = proc.stdout + "\n" + proc.stderr
        raw_outputs["test"] = output

        # Save raw output for debugging
        os.makedirs("artifacts", exist_ok=True)
        safe_name = name.replace(" ", "_").replace("+", "_")
        raw_path = f"artifacts/raw_output_{safe_name}.txt"
        with open(raw_path, "w") as f:
            f.write(f"=== TRAIN OUTPUT ===\n{raw_outputs.get('train', 'SKIPPED')}\n\n")
            f.write(f"=== TEST OUTPUT ===\n{output}\n")

        # Parse metrics
        metrics = parse_simulation_output(output)

        if metrics.get("Final Value ($)") or metrics.get("Total Return (%)"):
            result.update(metrics)
            print(f"  ✅ Done | Final: ${metrics.get('Final Value ($)', '?')} | "
                  f"Return: {metrics.get('Total Return (%)', '?')}% | "
                  f"Sharpe: {metrics.get('Sharpe Ratio', '?')}")
        else:
            result["Status"] = "PARSE_FAIL"
            print(f"  ❌ Could not parse metrics from output.")
            print(f"     Raw output saved → {raw_path}")
            for line in output.strip().splitlines()[-15:]:
                print(f"     | {line}")

    except subprocess.TimeoutExpired:
        result["Status"] = "TEST_TIMEOUT"
        print(f"  ⏰ Test timed out")
    except Exception as e:
        result["Status"] = f"TEST_ERROR: {e}"
        print(f"  💥 Test exception: {e}")

    return result


# ── Main ───────────────────────────────────────────────────────

def main():
    global TRAIN_START, TRAIN_END, TEST_START, TEST_END, TICKER

    parser = argparse.ArgumentParser(
        description="FinMEM Ablation Study — Train + Test all configurations"
    )
    parser.add_argument("--parallel", action="store_true",
                        help="Run all configs in parallel (caution: high API load)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, use existing checkpoints")
    parser.add_argument("--only", type=str, default=None,
                        help='Run only a specific config, e.g. --only "Base FinMEM"')
    parser.add_argument("--ticker", type=str, default=TICKER,
                        help=f"Ticker to test (default: {TICKER})")
    parser.add_argument("--train-start", type=str, default=None,
                        help="Override train start date (YYYY-MM-DD)")
    parser.add_argument("--train-end", type=str, default=None,
                        help="Override train end date (YYYY-MM-DD)")
    parser.add_argument("--test-start", type=str, default=None,
                        help="Override test start date (YYYY-MM-DD)")
    parser.add_argument("--test-end", type=str, default=None,
                        help="Override test end date (YYYY-MM-DD)")
    parser.add_argument("--label", type=str, default=None,
                        help="Label for this run (used in output filenames, e.g. 'paper', '3mo', '6mo')")
    args = parser.parse_args()

    # Override module-level globals from CLI
    if args.train_start:
        TRAIN_START = args.train_start
    if args.train_end:
        TRAIN_END = args.train_end
    if args.test_start:
        TEST_START = args.test_start
    if args.test_end:
        TEST_END = args.test_end
    ticker = args.ticker
    TICKER = ticker

    run_label = args.label or f"{TEST_START}_to_{TEST_END}"

    # Filter configs if --only is specified
    configs = CONFIGS
    if args.only:
        configs = [c for c in CONFIGS if args.only.lower() in c["name"].lower()]
        if not configs:
            print(f"❌ No config matching '{args.only}'. Available:")
            for c in CONFIGS:
                print(f"   - {c['name']}")
            sys.exit(1)

    print("=" * 60)
    print(f"  📊 FinMEM Ablation Study")
    print(f"  Ticker:       {ticker}")
    print(f"  Train Period: {TRAIN_START} → {TRAIN_END}")
    print(f"  Test Period:  {TEST_START} → {TEST_END}")
    print(f"  Label:        {run_label}")
    print(f"  Configs:      {len(configs)}")
    print(f"  Mode:         {'Parallel' if args.parallel else 'Sequential'}")
    print(f"  Skip Train:   {args.skip_train}")
    print(f"  Python:       {PYTHON}")
    print("=" * 60)

    os.makedirs("artifacts", exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    start_time = time.time()

    results = []

    if args.parallel:
        # Run all configs in parallel
        with ProcessPoolExecutor(max_workers=len(configs)) as executor:
            futures = {
                executor.submit(run_single_config, cfg, args.skip_train): cfg
                for cfg in configs
            }
            for future in as_completed(futures):
                cfg = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({"Configuration": cfg["name"], "Status": f"ERROR: {e}"})
        # Sort results back to original order
        name_order = {c["name"]: i for i, c in enumerate(configs)}
        results.sort(key=lambda r: name_order.get(r["Configuration"], 99))
    else:
        # Run sequentially
        for cfg in configs:
            result = run_single_config(cfg, args.skip_train)
            results.append(result)

    elapsed = time.time() - start_time

    # ── Column order for the output ────────────────────────────
    columns = [
        "Configuration",
        "Final Value ($)",
        "Total Return (%)",
        "Cumulative Return (%)",
        "Sharpe Ratio",
        "Ann. Volatility",
        "Daily Volatility",
        "Max Drawdown (%)",
        "BH Cum. Return (%)",
        "BH Sharpe Ratio",
        "BH Max Drawdown (%)",
        "Days",
        "Status",
    ]

    # ── Save CSV ───────────────────────────────────────────────
    csv_path = f"artifacts/ablation_results_{run_label}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"\n📄 CSV saved → {csv_path}")

    # ── Save Markdown ──────────────────────────────────────────
    md_path = f"artifacts/ablation_results_{run_label}.md"
    with open(md_path, "w") as f:
        f.write(f"# 📊 FinMEM Ablation Study Results — {run_label}\n\n")
        f.write(f"**Ticker:** {ticker}\n")
        f.write(f"**Train Period:** {TRAIN_START} → {TRAIN_END}\n")
        f.write(f"**Test Period:** {TEST_START} → {TEST_END}\n")
        f.write(f"**Run Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Elapsed:** {elapsed:.0f}s\n\n")

        # Build markdown table
        f.write("## Results\n\n")
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in results:
            vals = [str(row.get(c, "")) for c in columns]
            f.write("| " + " | ".join(vals) + " |\n")

        f.write("\n\n## Configurations\n\n")
        for cfg in configs:
            f.write(f"- **{cfg['name']}** → `{cfg['script']}` with flags: `{cfg['env']}`\n")

        f.write("\n\n## LLM Model\n\n")
        f.write(f"- Provider: `{os.getenv('LLM_PROVIDER', 'bedrock')}`\n")
        f.write(f"- Base Model: `{os.getenv('BEDROCK_MODEL_ID', 'unknown')}`\n")
        f.write(f"- Obj4 Models: All set to `{os.getenv('FUNDAMENTAL_MODEL', 'unknown')}`\n")

    print(f"📝 Markdown saved → {md_path}")

    # ── Print summary table ────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  ABLATION STUDY COMPLETE — {len(results)} configs in {elapsed:.0f}s")
    print(f"{'='*80}")
    print(f"  {'Configuration':<25} {'Return (%)':<12} {'Sharpe':<10} {'MaxDD (%)':<10} {'Status':<10}")
    print(f"  {'─'*70}")
    for r in results:
        print(f"  {r.get('Configuration', '?'):<25} "
              f"{r.get('Total Return (%)', 'N/A'):<12} "
              f"{r.get('Sharpe Ratio', 'N/A'):<10} "
              f"{r.get('Max Drawdown (%)', 'N/A'):<10} "
              f"{r.get('Status', '?'):<10}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
