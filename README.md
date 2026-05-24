# SmartFolio

Implementation of the IJCAI-25 paper **"Enhancing Portfolio Optimization via Heuristic-Guided Inverse Reinforcement Learning with Multi-Objective Reward and Graph-based Policy Learning."**

The method generates synthetic "expert" portfolios from finance heuristics (sector-diversified, decorrelated stock selection), learns a reward function from them via **Maximum-Entropy Inverse RL**, and optimizes a stock-selection policy with **PPO**. The policy is either a plain MLP or a **heterogeneous graph attention network (HGAT)** over three stock-relation graphs (industry, positive-correlation, negative-correlation).

The full paper is `paper.pdf` in the repo root. Architecture/internals for developers are documented in `CLAUDE.md`.

---

## 1. Setup

Requires the host to have the **NVIDIA Container Toolkit** installed (for `--gpus all`).

```bash
docker build -t smartfolio .
```

---

## 2. Generate preprocessed data (required before training)

Raw `*_org.csv` files for all four markets are committed. The generated files — correlation matrices (`dataset/corr/{market}/`) and per-day `.pkl` samples (`dataset/data_train_predict_{market}/1_hy/`) — must be built before training.

**hs300 (CSI 300):**
```bash
docker run --rm --gpus all -v "$(pwd)":/app smartfolio bash -c "cd gen_data && python build_market.py hs300"
```

**zz500 (CSI 500):**
```bash
docker run --rm --gpus all -v "$(pwd)":/app smartfolio bash -c "cd gen_data && python build_market.py zz500"
```

**nd100 (NASDAQ 100):**
```bash
docker run --rm --gpus all -v "$(pwd)":/app smartfolio bash -c "cd gen_data && python build_market.py nd100"
```

**sp500 (S&P 500):**
```bash
docker run --rm --gpus all -v "$(pwd)":/app smartfolio bash -c "cd gen_data && python build_market.py sp500"
```

**What each step does per market:**

| Step | hs300 / zz500 (China) | nd100 / sp500 (US) |
|---|---|---|
| Industry graph | Skipped — slices pre-committed `dataset/A_stock_industry_matrx.csv` | Builds `dataset/{market}/industry.npy` from `dataset/us_sectors.csv` (binary GICS-sector matrix) |
| Correlation matrices | `dataset/corr/{market}/*.csv` (monthly Pearson) | Same |
| Per-day samples | `dataset/data_train_predict_{market}/1_hy/*.pkl` | Same |

---

## 3. Data

| Market (`-mkt`) | Index | Stocks | Region |
|---|---|---|---|
| `hs300` | CSI 300 | 102 | China |
| `zz500` | CSI 500 | 80  | China |
| `nd100` | NASDAQ 100 | 84 | US |
| `sp500` | S&P 500 | 472 | US |

Stock counts are the subset of each index that traded on **every single day** from 2018–2024 (no suspensions, delistings, or gaps). Train/validation/test split (fixed in `main.py`): **train 2019–2022, validation 2023, test 2024.**

---

## 4. Run training + testing

**Mount the repo as a volume (`-v "$(pwd)":/app`)** so the `results/` files survive after the container exits.

```bash
# default: hs300, MLP policy
docker run --rm --gpus all -v "$(pwd)":/app smartfolio python main.py -mkt hs300

# pick a market and/or policy
docker run --rm --gpus all -v "$(pwd)":/app smartfolio python main.py -mkt sp500 -p HGAT

# interactive shell (to run several configs, inspect output)
docker run --rm --gpus all -v "$(pwd)":/app -it smartfolio bash
```

The run trains the IRL reward network and PPO policy, then evaluates on the 2024 test set.

### Reading the output

At the end of the test episode it prints:

- **`net_values: [...]`** — cumulative-wealth curve (1.0 = starting capital; e.g. 1.43 = +43%)
- **ARR** — Annualized Return Rate
- **AVol** — Annualized Volatility
- **Sharpe** — Sharpe Ratio
- **MDD** — Maximum Drawdown
- **CR** — Calmar Ratio
- **IR** — Information Ratio (reports **0** unless benchmark index files are present — see §6)

### Result files

Results are written to `results/` (persisted to the host only if you mounted the volume):

- **`results/summary.csv`** — one row appended per run with config and all metrics.
- **`results/{market}_{policy}_{timestamp}_equity.csv`** — full equity curve for that run.

---

## 5. Ablation studies (paper Table 2)

Ablation toggles are CLI flags taking `y`/`n` (default `y` = full model):

| Configuration | Command |
|---|---|
| Full model (graph policy) | `python main.py -mkt hs300 -p HGAT` |
| w/o reward network / multi-objective reward | `python main.py -mkt hs300 -mr n` |
| w/o industry diversification | `python main.py -mkt hs300 -ind n` |
| w/o correlation control | `python main.py -mkt hs300 -pos n -neg n` |
| w/o HGAT (use MLP) | `python main.py -mkt hs300 -p MLP` |

> Note: the policy **defaults to `-p MLP`**, so the paper's "Full Model" requires explicitly passing `-p HGAT`. Training length, dates, batch size, and seed are fixed in the `main.py` debug block — edit there to change them.
