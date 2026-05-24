import gym
import pandas as pd
import torch
from gym import spaces
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

class StockPortfolioEnv(gym.Env):
    def __init__(self, args, corr=None, ts_features=None, features=None,
                 ind=None, pos=None, neg=None, returns=None, pyg_data=None,
                 benchmark_return=None, mode="train", reward_net=None, device='cuda:0',
                 ind_yn=False, pos_yn=False, neg_yn=False):
        super(StockPortfolioEnv, self).__init__()
        self.current_step = 0
        self.max_step = returns.shape[0] - 1
        self.done = False
        self.reward = 0.0
        self.net_value = 1.0
        self.net_value_s = [1.0]
        self.daily_return_s = [0.0]
        self.num_stocks = returns.shape[-1]
        self.benchmark_return = benchmark_return

        self.corr_tensor = corr
        self.ts_features_tensor = ts_features
        self.features_tensor = features
        self.ind_tensor = ind
        self.pos_tensor = pos
        self.neg_tensor = neg
        self.pyg_data_batch = pyg_data
        self.ror_batch = returns
        self.ind_yn = ind_yn
        self.pos_yn = pos_yn
        self.neg_yn = neg_yn

        # # action space: continuous values in [0, 1]
        # # meaning: per-stock score
        # self.action_space = spaces.Box(low=0,
        #                                high=1,
        #                                shape=(self.num_stocks,),
        #                                dtype=np.float32)

        # select a fixed number of stocks (e.g. 10%)
        self.top_k = max(1, int(0.1 * self.num_stocks))  # number of stocks selected each step
        # action space: discrete, the selected stock indices
        self.action_space = spaces.MultiDiscrete([self.num_stocks] * self.top_k)

        # partially observable
        # observation space: per-stock features plus the relation graphs
        obs_len = args.input_dim
        if self.ind_yn:
            obs_len += self.num_stocks
        if self.pos_yn:
            obs_len += self.num_stocks
        if self.neg_yn:
            obs_len += self.num_stocks
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(self.num_stocks, obs_len),
                                            dtype=np.float32)
        self.mode = mode
        self.reward_net = reward_net  # injected IRL reward network
        self.device = device
        self.args = args
        self.market = getattr(args, 'market', 'unknown')
        self.policy = getattr(args, 'policy', 'MLP')

    def load_observation(self, ts_yn=False, ind_yn=False, pos_yn=False, neg_yn=False):
        # SB3's DummyVecEnv requires the env observation (obs) to be stored as a NumPy array
        if torch.isnan(self.features_tensor).any():
            print("nan!!!")
        features = self.features_tensor[self.current_step].cpu().numpy()
        corr_matrix = self.corr_tensor[self.current_step].cpu().numpy()
        ind_matrix = self.ind_tensor[self.current_step].cpu().numpy()
        pos_matrix = self.pos_tensor[self.current_step].cpu().numpy()
        neg_matrix = self.neg_tensor[self.current_step].cpu().numpy()
        obs = features
        # obs = np.concatenate([features, corr_matrix], axis=1)
        if ind_yn:
            obs = np.concatenate([obs, ind_matrix], axis=1)
        if pos_yn:
            obs = np.concatenate([obs, pos_matrix], axis=1)
        if neg_yn:
            obs = np.concatenate([obs, neg_matrix], axis=1)
        self.observation = obs
        self.ror = self.ror_batch[self.current_step].cpu()


    def reset(self):
        self.current_step = 0
        self.done = False
        self.reward = 0.0
        self.net_value = 1.0
        self.net_value_s = [1.0]
        self.daily_return_s = [0.0]
        self.load_observation(ind_yn=self.ind_yn, pos_yn=self.pos_yn, neg_yn=self.neg_yn)
        return self.observation

    def seed(self, seed):
        return np.random.seed(seed)

    def step(self, actions):
        self.done = self.current_step == self.max_step
        if self.done:
            if self.mode == "test":
                print("=================================")
                print(f"net_values:{self.net_value_s}")
                arr, avol, sharpe, mdd, cr, ir = self.evaluate()

                print("ARR: ", arr)
                print("AVOL: ", avol)
                print("Sharpe: ", sharpe)
                print("MDD: ", mdd)
                print("CR: ", cr)
                print("IR: ", ir)
                print("=================================")
                # save results before DummyVecEnv auto-resets and clears net_value_s
                try:
                    self._save_results(arr, avol, sharpe, mdd, cr, ir)
                except Exception as e:
                    print(f"[results] could not save results file: {e}")
        else:
            # load s'
            self.current_step += 1
            self.load_observation(ind_yn=self.ind_yn, pos_yn=self.pos_yn, neg_yn=self.neg_yn)
            # under MultiDiscrete, pick stocks from actions
            selected_indices = list(set(actions))  # dedupe
            if self.mode == "test":
                print(self.current_step)
                print(selected_indices)
            weights = np.zeros(self.num_stocks)
            weights[selected_indices] = 1.0 / len(selected_indices)  # equal weight across selected stocks

            # use the IRL reward network instead of the raw reward
            if self.reward_net is not None:
                state_tensor = torch.FloatTensor(np.expand_dims(self.observation, 1)).to(self.device)  # current state
                # convert the action to multi-hot encoding for the reward network
                action_multi_hot = np.zeros(self.num_stocks)
                action_multi_hot[selected_indices] = 1
                action_tensor = torch.FloatTensor(action_multi_hot).to(self.device)  # action (weight vector)
                with torch.no_grad():
                    self.reward = self.reward_net(state_tensor, action_tensor).mean().cpu().item()
            else:
                self.reward = np.dot(weights, np.array(self.ror))

            self.net_value *= (1 + self.reward)
            self.daily_return_s.append(self.reward)
            self.net_value_s.append(self.net_value)

        return self.observation, self.reward, self.done, {}

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

    def get_df_net_value(self):
        df_net_value = pd.DataFrame(self.net_value_s)
        df_net_value.columns = ["net_value"]
        return df_net_value

    def get_df_daily_return(self):
        df_daily_return = pd.DataFrame(self.daily_return_s)
        df_daily_return.columns = ["daily_return"]
        return df_daily_return

    def evaluate(self):
        arr, avol, sp, mdd, cr, ir = (0, 0, 0, 0, 0, 0)
        df_daily_return = self.get_df_daily_return()
        if df_daily_return["daily_return"].std() != 0:
            # annualized return (ARR)
            # assume 252 trading days per year
            arr = (1 + df_daily_return['daily_return'].mean()) ** 252 - 1
            # annualized volatility (AVol)
            avol = df_daily_return["daily_return"].std() * (252 ** 0.5)
            sp = (
                    (252 ** 0.5)
                    * df_daily_return["daily_return"].mean()
                    / df_daily_return["daily_return"].std()
            )
            # cumulative return
            df_daily_return['cumulative_return'] = (1 + df_daily_return['daily_return']).cumprod()
            # the running maximum
            running_max = df_daily_return['cumulative_return'].cummax()
            # drawdown
            drawdown = df_daily_return['cumulative_return'] / running_max - 1
            # Maximum Drawdown (MDD)
            mdd = drawdown.min()
            # Calmar Ratio (CR)
            if mdd != 0:
                cr = arr / abs(mdd)
            # Information Ratio (IR) (requires a benchmark return series)
            if self.benchmark_return is not None:
                if len(self.benchmark_return) == len(df_daily_return):
                    ex_return = df_daily_return["daily_return"] -\
                                self.benchmark_return.reset_index(drop=True)
                    if ex_return.std() != 0:
                        ir = ex_return.mean() / ex_return.std() * (252 ** 0.5)
        return arr, avol, sp, mdd, cr, ir

    def _save_results(self, arr, avol, sharpe, mdd, cr, ir):
        """Persist test-episode results: per-day equity curve + a row appended
        to a master summary table. Written when the test episode ends."""
        import os
        import datetime
        out_dir = "results"
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = f"{self.market}_{self.policy}_{ts}"

        # equity curve over the test period (cumulative wealth + daily return)
        pd.DataFrame({
            "net_value": self.net_value_s,
            "daily_return": self.daily_return_s,
        }).to_csv(os.path.join(out_dir, f"{tag}_equity.csv"), index_label="step")

        # one-row summary appended to results/summary.csv (master performance table)
        row = {
            "timestamp": ts, "market": self.market, "policy": self.policy,
            "ind": self.ind_yn, "pos": self.pos_yn, "neg": self.neg_yn,
            "multi_reward": getattr(self.args, "multi_reward", None),
            "ARR": arr, "AVol": avol, "Sharpe": sharpe,
            "MDD": mdd, "CR": cr, "IR": ir,
            "final_net_value": self.net_value_s[-1] if self.net_value_s else None,
        }
        summary = os.path.join(out_dir, "summary.csv")
        pd.DataFrame([row]).to_csv(summary, mode="a",
                                   header=not os.path.exists(summary), index=False)
        print(f"[results] equity -> {out_dir}/{tag}_equity.csv | summary appended -> {summary}")
