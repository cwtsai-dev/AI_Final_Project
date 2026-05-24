# SmartFolio

Implementation of the IJCAI-25 paper **"Enhancing Portfolio Optimization via Heuristic-Guided Inverse Reinforcement Learning with Multi-Objective Reward and Graph-based Policy Learning."**

The method generates synthetic "expert" portfolios from finance heuristics (sector-diversified, decorrelated stock selection), learns a reward function from them via **Maximum-Entropy Inverse RL**, and optimizes a stock-selection policy with **PPO**. The policy is either a plain MLP or a **heterogeneous graph attention network (HGAT)** over three stock-relation graphs (industry, positive-correlation, negative-correlation).

The full paper is `1054.pdf` in the repo root. Architecture/internals for developers are documented in `CLAUDE.md`.

---

## 1. Requirements

This project uses a **legacy stack** (old `gym` API). The versions are pinned and matched to each other — don't upgrade them:

- `python 3.8`, `torch 1.12.1`, `torch-geometric 2.0.4`
- `gym 0.21.0`, `stable-baselines3 1.6.2`

A **GPU with CUDA 11.3+** is recommended (the Dockerfile builds a CUDA 11.3 image). Training runs on CPU too, just slowly.

---

## 2. Setup

Requires the host to have the **NVIDIA Container Toolkit** installed (for `--gpus all`).

```bash
docker build -t smartfolio .
```

---

## 3. Data

Raw OHLCV data for all four markets is included in `dataset/`. **Before running training you must generate the preprocessed files** (see §6 — takes a few minutes per market on GPU).

| Market (`-mkt`) | Index | Stocks | Region |
|---|---|---|---|
| `hs300` | CSI 300 | 102 | China |
| `zz500` | CSI 500 | 80  | China |
| `nd100` | NASDAQ 100 | 84 | US |
| `sp500` | S&P 500 | 472 | US |

Stock counts are the subset of each index that traded on **every single day** from 2018–2024 (no suspensions, delistings, or gaps). Train/validation/test split (fixed in `main.py`): **train 2019–2022, validation 2023, test 2024.**

---

## 4. Run training + testing

**Mount the repo as a volume (`-v "$(pwd)":/app`)** so the `results/` files survive after the container exits — otherwise, with `--rm`, everything written inside the container is lost. Mounting also makes the live host code take effect without rebuilding.

```bash
# from the repo root. default: hs300, MLP policy
docker run --rm --gpus all -v "$(pwd)":/app smartfolio python main.py -mkt hs300

# pick a market and/or policy
docker run --rm --gpus all -v "$(pwd)":/app smartfolio python main.py -mkt sp500 -p HGAT

# interactive shell (to run several configs, inspect output)
docker run --rm --gpus all -v "$(pwd)":/app -it smartfolio bash
```

The run trains the IRL reward network and PPO policy, then evaluates on the 2024 test set.

### Reading the output

At the end of the test episode it prints:

- **`net_values: [...]`** — the cumulative-wealth curve over the test period (1.0 = starting capital; e.g. 1.43 = +43%). This is your equity curve / "return over time."
- **ARR** — Annualized Return Rate
- **AVol** — Annualized Volatility
- **Sharpe** — Sharpe Ratio
- **MDD** — Maximum Drawdown
- **CR** — Calmar Ratio
- **IR** — Information Ratio (reports **0** unless benchmark index files are present — see §7)

The test episode uses **real portfolio returns** (not the learned reward), so these reflect genuine performance.

### Result files

At the end of every run, results are written to a `results/` folder (persisted to the host **only if you mounted the volume** as shown above):

- **`results/summary.csv`** — one row **appended per run** with the config (market, policy, ablation flags) and all metrics (ARR, AVol, Sharpe, MDD, CR, IR, final net value). This is the master table to compare runs.
- **`results/{market}_{policy}_{timestamp}_equity.csv`** — the full equity curve for that run (`net_value` and `daily_return` per test day), e.g. for plotting cumulative wealth.

So to check performance after a run, open `results/summary.csv`. Each run adds a row (timestamped), so nothing is overwritten.

---

## 5. Ablation studies (paper Table 2)

The ablation toggles are CLI flags taking `y`/`n` (default `y` = full model). Run them from an interactive shell (started as shown in §4) or substitute the `python main.py ...` part into a full `docker run` command:

| Configuration | Command (inside container) |
|---|---|
| Full model (graph policy) | `python main.py -mkt hs300 -p HGAT` |
| w/o reward network / multi-objective reward | `python main.py -mkt hs300 -mr n` |
| w/o industry diversification | `python main.py -mkt hs300 -ind n` |
| w/o correlation control | `python main.py -mkt hs300 -pos n -neg n` |
| w/o HGAT (use MLP) | `python main.py -mkt hs300 -p MLP` |

> Note: the policy **defaults to `-p MLP`**, so the paper's "Full Model" requires explicitly passing `-p HGAT`. Training length, dates, batch size, and seed are fixed in the debug block of `main.py` — edit there to change them (e.g. set `max_epochs = 1` for a quick smoke test).

---

## 6. Generating market data (required before first run)

The preprocessed files are not in the repo. Run the one-shot builder for each market you want to use. Results are written to the mounted host directory, so they persist after the container exits.

```bash
# from the repo root — repeat for each market you need
docker run --rm --gpus all -v "$(pwd)":/app smartfolio \
    bash -c "cd gen_data && python build_us_market.py hs300"

docker run --rm --gpus all -v "$(pwd)":/app smartfolio \
    bash -c "cd gen_data && python build_us_market.py zz500"

docker run --rm --gpus all -v "$(pwd)":/app smartfolio \
    bash -c "cd gen_data && python build_us_market.py nd100"

docker run --rm --gpus all -v "$(pwd)":/app smartfolio \
    bash -c "cd gen_data && python build_us_market.py sp500"
```

Or run all four back-to-back inside one interactive shell:

```bash
docker run --rm --gpus all -v "$(pwd)":/app -it smartfolio bash
# inside the container:
cd gen_data
for mkt in hs300 zz500 nd100 sp500; do python build_us_market.py $mkt; done
```

For each market this builds: the industry graph (US: `dataset/{market}/industry.npy` from `dataset/us_sectors.csv`; China: sliced from `dataset/A_stock_industry_matrx.csv`), the monthly correlation matrices (`dataset/corr/{market}/`), and the per-day `.pkl` samples (`dataset/data_train_predict_{market}/1_hy/`).

---

## 7. Known limitations

- **Information Ratio (IR) reports 0.** It needs benchmark index daily-return series at `dataset/index_data/{market}_index_2024.csv`, which are not included. All other metrics work without them.
- **Only SmartFolio is implemented.** The paper's baseline models (LSTM, Transformer, AlphaStock, DeepTrader, GPT4TS, TIME-LLM, etc.) are not in this repo, so the head-to-head comparison (Table 1) cannot be reproduced here — only this method and its ablations.
- **Some details diverge from the paper** (e.g. the synthetic expert ranks by forward returns; Chinese industry graphs are graded rather than binary). See `CLAUDE.md` for the full list.
