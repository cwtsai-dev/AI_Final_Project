import os
import time
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import pandas as pd
import torch
print(torch.cuda.is_available())
from dataloader.data_loader import *
from policy.policy import *
# from trainer.trainer import *
from trainer.irl_trainer import *
from torch_geometric.loader import DataLoader

PATH_DATA = f'./dataset/'

def train_predict(args, predict_dt):
    # GPU selection is controlled by docker --gpus / the device arg; do not set CUDA_VISIBLE_DEVICES here
    # (str(args.device)=="cuda:0" is an invalid value for that env var; it expects an index like "0")
    data_dir = f'dataset/data_train_predict_{args.market}/{args.horizon}_{args.relation_type}/'
    train_dataset = AllGraphDataSampler(base_dir=data_dir, date=True,
                                        train_start_date=args.train_start_date, train_end_date=args.train_end_date,
                                        mode="train")
    val_dataset = AllGraphDataSampler(base_dir=data_dir, date=True,
                                      val_start_date=args.val_start_date, val_end_date=args.val_end_date,
                                      mode="val")
    test_dataset = AllGraphDataSampler(base_dir=data_dir, date=True,
                                       test_start_date=args.test_start_date, test_end_date=args.test_end_date,
                                       mode="test")
    train_loader_all = DataLoader(train_dataset, batch_size=len(train_dataset), pin_memory=True, collate_fn=lambda x: x,
                                  drop_last=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, collate_fn=lambda x: x,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), pin_memory=True)
    print(len(train_loader), len(val_loader), len(test_loader))

    # create model
    env_init = create_env_init(args, dataset=train_dataset)
    if args.policy == 'MLP':
        model = PPO(policy='MlpPolicy',
                    env=env_init,
                    **PPO_PARAMS,
                    seed=args.seed,
                    device=args.device)
    elif args.policy == 'HGAT':
        model = PPO(policy=HGATActorCriticPolicy,
                    env=env_init,
                    **PPO_PARAMS,
                    seed=args.seed,
                    device=args.device)
    train_model_and_predict(model, args, train_loader, val_loader, test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transaction ..")
    parser.add_argument("-device", "-d", default="cuda:0", help="gpu")
    parser.add_argument("-model_name", "-nm", default="SmartFolio", help="model name")
    parser.add_argument("-market", "-mkt", default="hs300", help="stock market")
    parser.add_argument("-horizon", "-hrz", default="1", help="prediction horizon")
    parser.add_argument("-relation_type", "-rt", default="hy", help="stock relation type")
    parser.add_argument("-ind_yn", "-ind", default="y", help="whether to include the industry relation graph")
    parser.add_argument("-pos_yn", "-pos", default="y", help="whether to include the positive-correlation (momentum) graph")
    parser.add_argument("-neg_yn", "-neg", default="y", help="whether to include the negative-correlation (reversal) graph")
    parser.add_argument("-multi_reward_yn", "-mr", default="y", help="whether to use multi-objective reward learning")
    parser.add_argument("-policy", "-p", default="MLP", help="policy network")
    args = parser.parse_args()

    # debug parameter settings
    args.model_name = 'SmartFolio'
    # args.market comes from the CLI -mkt/-market (default hs300); no longer hardcoded
    args.relation_type = 'hy'
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.train_start_date = '2019-01-02'
    args.train_end_date = '2022-12-30'
    args.val_start_date = '2023-01-03'
    args.val_end_date = '2023-12-29'
    args.test_start_date = '2024-01-02'
    args.test_end_date = '2024-12-30'
    args.batch_size = 32
    args.max_epochs = 60
    args.seed = 123
    args.input_dim = 6
    # ablation switches come from the CLI ("y"/"n"); default all on (= full model)
    args.ind_yn = (args.ind_yn == 'y')
    args.pos_yn = (args.pos_yn == 'y')
    args.neg_yn = (args.neg_yn == 'y')
    args.multi_reward = (args.multi_reward_yn == 'y')

    if args.market == 'hs300':
        args.num_stocks = 102
    elif args.market == 'zz500':
        args.num_stocks = 80
    elif args.market == 'nd100':
        args.num_stocks = 84
    elif args.market == 'sp500':
        args.num_stocks = 472

    train_predict(args, predict_dt='2025-02-05')

    print(1)




