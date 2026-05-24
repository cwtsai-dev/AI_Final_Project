# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Implementation of the IJCAI-25 paper **"Enhancing Portfolio Optimization via Heuristic-Guided Inverse Reinforcement Learning with Multi-Objective Reward and Graph-based Policy Learning"** (SmartFolio). The paper PDF is `1054.pdf` in the repo root and is the source of truth for intended behavior.

The method: generate synthetic "expert" portfolios from finance heuristics (greedy sector-diversified, decorrelated stock selection), learn a reward function from them via **MaxEnt Inverse RL**, and optimize a stock-selection policy with **PPO** (Stable-Baselines3). The policy can be a plain MLP or a **heterogeneous graph attention network (HGAT)** over three stock-relation graphs (industry, positive-correlation, negative-correlation).

## Environment

Legacy stack — do not "upgrade" these casually; the code uses the old `gym` API (`reset()->obs`, `step()->4-tuple`):
- `gym==0.21.0`, `stable-baselines3==1.6.2`, `torch==1.12.1`, `torch-geometric==2.0.4`
- See `Dockerfile` (GPU/CUDA 11.3 build) and `requirements.txt`. `torch-scatter`/`torch-sparse`/`torch` install from dedicated index URLs (handled in the Dockerfile).
- `torch_geometric` is treated as **optional** in the data-gen scripts (guarded import); training requires it.

## Commands

All commands run **inside the Docker container**. Start one with:
```bash
docker run --rm --gpus all -v "$(pwd)":/app -it smartfolio bash
```

Run training + test (no test suite exists in this repo):
```bash
python main.py -mkt hs300        # market: hs300 | zz500 | nd100 | sp500
python main.py -mkt sp500 -p HGAT  # -p selects policy: MLP (default) or HGAT
```
Output: ARR / AVol / Sharpe / MDD / Calmar printed at end of the test episode (IR = 0 unless benchmark index files are present — see below).

Ablation flags (paper Table 2) take `y`/`n`; default `y` = full model:
```bash
python main.py -ind n     # w/o industry diversification (ind_yn)
python main.py -pos n -neg n   # w/o correlation control (pos/neg graphs)
python main.py -mr n      # w/o reward network / multi-objective reward (multi_reward_yn)
python main.py -p HGAT    # full model uses HGAT; -p MLP (default) = "w/o HGAT"
```

Regenerate preprocessed data (run from the `gen_data/` directory; paths are relative to `../dataset`):
```bash
cd gen_data
python generate_relation.py hs300              # monthly Pearson corr matrices -> dataset/corr/{market}/
python train_predict_data.py                   # builds .pkl samples (market hardcoded in __main__)
python build_us_market.py nd100|sp500          # one-shot: industry.npy + corr + .pkl for a US market
```

## Pipeline architecture (read these together)

Data flows: raw CSV → preprocessed per-day `.pkl`s → gym env (time-stepped) → IRL reward learning ↔ PPO policy.

1. **Data generation** (`gen_data/`): `train_predict_data.py` turns `dataset/{market}_org.csv` (OHLCV) into one `.pkl` **per trading day** in `dataset/data_train_predict_{market}/1_hy/`. Each `.pkl` is a dict bundling: `features` (N×1×6), `ts_features` (N×20×6), `labels` (N forward returns), `corr`, `industry_matrix`, `pos_matrix`, `neg_matrix` (all N×N), `mask`. Correlations come from `generate_relation.py`; pos/neg graphs are `corr` thresholded at ±0.2 (binary, self-loops removed). `build_us_market.py` is the orchestrator for US markets (also builds `industry.npy`).

2. **Loading** (`dataloader/data_loader.py`): `AllGraphDataSampler` loads all `.pkl`s and slices train/val/test by **date strings** (matched against filenames). PyG's `DataLoader` uses its own `Collater` (any `collate_fn=` is ignored), which collates a batch of per-day dicts into a dict of stacked tensors with a leading time/batch dimension.

3. **Environment** (`env/portfolio_env.py`): `StockPortfolioEnv` is a gym env where one **step = one trading day**. Action space is `MultiDiscrete` = pick `top_k = 10%` of stocks; selected stocks get equal weight. Observation is built in `load_observation` by concatenating per-stock: `[features | industry_matrix | pos_matrix | neg_matrix]` (each relation matrix gated by `ind_yn`/`pos_yn`/`neg_yn`). **Raw `corr` is NOT in the observation** (commented out). At test time the env uses real portfolio return as reward; during training it uses the learned `reward_net`.

4. **Reward + IRL** (`trainer/irl_trainer.py`): `MaxEntIRL` trains a reward network from expert vs. agent trajectories. `MultiRewardNetwork` (the `multi_reward=True` path) splits the observation into base/ind/pos/neg streams with softmax-weighted fusion. `train_model_and_predict` runs the **alternation loop**: each epoch trains the reward net one pass, then rebuilds the env with the new reward and runs PPO, finally predicting on the test set. Expert trajectories come from `gen_data/generate_expert.py` (Algorithm 1 in the paper).

5. **Policy + GNN** (`policy/policy.py`, `model/model.py`): `HGATActorCriticPolicy` plugs a custom extractor into SB3 PPO. `HGATNetwork._pack` decodes the **flattened** observation back into the three adjacency matrices `[ind; pos; neg; features]` that `HGAT` (in `model.py`) expects, runs a multi-head GAT per graph, fuses them with attention (`HeteFusionAttn`), and generates per-stock scores. With `-p MLP` the graphs are still in the observation but only seen as flat features, not as graph structure.

## Critical, non-obvious details

- **`main.py` hardcodes the training schedule** in a debug block (dates 2019–2024 splits, `batch_size`, `max_epochs`, `seed`, `input_dim`) — to change those you edit the block. The market (`-mkt`), policy (`-p`), and ablation flags (`-ind/-pos/-neg/-mr`, each `y`/`n`) **are honored from the CLI**; the debug block converts the `y`/`n` strings to the booleans downstream code expects (`args.ind_yn`/`pos_yn`/`neg_yn`/`multi_reward`). `num_stocks` is hardcoded per market here (hs300=102, zz500=80, nd100=84, sp500=472) and must match the data.

- **Industry graph differs by region.** hs300/zz500 (Chinese A-shares) slice the shared `dataset/A_stock_industry_matrx.csv`, which is **graded** (0 / 0.8 / 0.9 / 1.0, a 3-level industry hierarchy). nd100/sp500 (US) use `dataset/{market}/industry.npy`, built **binary** (same-GICS-sector = 1) from `dataset/us_sectors.csv`. The paper's written definition is binary; the graded A-share matrix is undocumented in the paper and has no generation code in the repo.

- **Market data status:** raw `*_org.csv` files for all four markets are committed. The generated files (`dataset/corr/`, `dataset/data_train_predict_*/`, US `industry.npy`) are **not committed** — run `gen_data/build_us_market.py <market>` (inside Docker, with volume mounted) for each market before training. See README §6. Note: regenerated hs300 results may differ slightly from the paper due to KMeans non-determinism across sklearn versions.

- **`dataset/index_data/{market}_index_2024.csv` is absent.** It's the benchmark index daily-return series, used **only** to compute the Information Ratio (IR). The read is guarded, so IR just reports 0; all other metrics work without it.

- **`trainer/trainer.py` is the non-IRL alternative trainer and is unused** (`main.py` imports `trainer/irl_trainer.py`).

- **`pyg_data` is optional**: shipped `.pkl`s don't contain it and the env never uses it; `process_data` tolerates its absence.

## Known divergences from the paper (intentional to flag, not yet reconciled)

- Expert generation (`generate_expert.py`) ranks stocks by `labels` (**forward** returns) rather than the paper's *historical* returns — effectively an oracle demonstrator.
- Reward weights use a learnable softmax, not the Lagrangian-duality scheme described in the paper.
- Baseline models from the paper's comparison (LSTM, Transformer, AlphaStock, DeepTrader, GPT4TS, etc.) are **not in this repo** — only SmartFolio itself, so Table 1's head-to-head cannot be reproduced here.
