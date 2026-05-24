import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler


def generate_expert_strategy(returns,
                             industry_relation_matrix,
                             correlation_matrix,
                             top_k=0.1,
                             max_industry_ratio=0.3):
    """
    Generate the expert strategy: diversification via the industry relation matrix, correlation control, and momentum.
    :param returns: historical return data (DataFrame; columns = stock codes, rows = dates).
    :param industry_relation_matrix: industry relation matrix (num_stocks x num_stocks; weights = industry relatedness).
    :param correlation_matrix: stock correlation matrix (DataFrame indexed by stock code on both axes).
    :param top_k: fraction of stocks to select.
    :param max_industry_ratio: max fraction per industry cluster.
    :return: expert strategy (binary array; 1 = selected, 0 = not).
    """
    num_stocks = len(returns)
    # Step 1: rank by return descending to form the candidate queue
    candidate_indices = np.argsort(-returns).tolist()  # indices sorted descending
    target_k = int(num_stocks * top_k)  # target number of stocks to pick
    expert_actions = np.zeros(num_stocks, dtype=int)
    selected_stocks = []
    industry_counts = {}  # count selected per industry
    # Step 2: iterate the candidate queue until full or empty
    while len(selected_stocks) < target_k and candidate_indices:
        idx = candidate_indices.pop(0)  # pop the current highest-return candidate
        # define this stock's industry cluster
        industry_cluster = np.where(industry_relation_matrix[idx] > 0)[0].tolist()
        industry_cluster.append(idx)  # include itself
        # count how many already selected in this industry
        selected_in_cluster = sum(expert_actions[industry_cluster])
        max_allowed = int(target_k * max_industry_ratio)
        # industry cap check
        if selected_in_cluster >= max_allowed:
            continue  # skip this stock, continue to the next candidate
        # correlation check
        if selected_stocks:
            avg_corr = correlation_matrix[idx, selected_stocks].mean()
            if avg_corr >= 0.5:
                continue  # correlation too high, skip
        # select this stock
        expert_actions[idx] = 1
        selected_stocks.append(idx)
        # update industry counts
        for stock in industry_cluster:
            industry_counts[stock] = industry_counts.get(stock, 0) + 1
    return expert_actions


def generate_expert_trajectories(args, dataset, num_trajectories=100):
    """
    Generate expert trajectories (state-action pairs) directly from the preprocessed time-series features.
    :param args: CLI arguments (market, industry classification, etc.).
    :param dataset: dataset (each sample already contains time-series features and a correlation matrix).
    :param num_trajectories: number of trajectories to generate.
    :return: list of expert trajectories, each a (state, action) pair.
    """
    expert_trajectories = []

    for _ in range(num_trajectories):
        # randomly pick a data point; each already has full time-series features
        idx = np.random.randint(0, len(dataset))
        data = dataset[idx]

        # extract time-series features and the correlation matrix
        time_series_features = data['ts_features'].numpy()  # shape [num_stocks, time_steps, feature_dim]
        features = data['features'].numpy()  # shape [num_stocks, feature_dim]
        correlation_matrix = data['corr'].numpy()  # shape [num_stocks, num_stocks]
        ind_matrix = data['industry_matrix'].numpy()  # shape [num_stocks, num_stocks]
        pos_matrix = data['pos_matrix'].numpy()  # shape [num_stocks, num_stocks]
        neg_matrix = data['neg_matrix'].numpy()  # shape [num_stocks, num_stocks]

        # extract returns (here from the labels field)
        returns = data['labels'].numpy()  # returns array

        # generate the expert action (using this sample's own industry relation matrix)
        expert_actions = generate_expert_strategy(
            returns=returns,
            industry_relation_matrix=ind_matrix,
            correlation_matrix=correlation_matrix
        )

        state = features.squeeze()
        if args.ind_yn:
            state = np.concatenate([state, ind_matrix], axis=1)
        if args.pos_yn:
            state = np.concatenate([state, pos_matrix], axis=1)
        if args.neg_yn:
            state = np.concatenate([state, neg_matrix], axis=1)
        expert_trajectories.append((state, expert_actions))
    return expert_trajectories


def load_industry_relation_matrix(market):
    """
    Load the industry relation matrix.
    :param market: market name (e.g. 'hs300').
    :return: industry relation matrix (num_stocks x num_stocks).
    """
    with open(f"dataset_default/data_train_predict_{market}/industry.npy", 'rb') as f:
        industry_relation_matrix = np.load(f)
    return industry_relation_matrix


def process_state(features):
    """
    Process state features (e.g. standardization).
    :param features: raw feature data (numpy array).
    :return: processed state features.
    """
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    return features_scaled


def save_expert_trajectories(trajectories, save_path):
    """
    Save expert trajectories to a file.
    :param trajectories: list of expert trajectories.
    :param save_path: output path.
    """
    with open(save_path, 'wb') as f:
        pickle.dump(trajectories, f)


def load_expert_trajectories(load_path):
    """
    Load expert trajectories from a file.
    :param load_path: input path.
    :return: list of expert trajectories.
    """
    with open(load_path, 'rb') as f:
        trajectories = pickle.load(f)
    return trajectories


if __name__ == '__main__':
    # test params: generate and save expert trajectories
    class Args:
        market = 'hs300'
        input_dim = 6


    args = Args()
    from dataloader.data_loader import AllGraphDataSampler

    # load dataset
    data_dir = f'../dataset/data_train_predict_{args.market}/1_hy/'
    train_dataset = AllGraphDataSampler(base_dir=data_dir, date=True,
                                        train_start_date='2019-01-02', train_end_date='2022-12-30',
                                        mode="train")

    # generate expert trajectories
    expert_trajectories = generate_expert_trajectories(args, train_dataset, num_trajectories=100)

    # save expert trajectories
    save_path = f'..dataset/expert_trajectories_{args.market}.pkl'
    save_expert_trajectories(expert_trajectories, save_path)
    print(f"expert trajectories saved to {save_path}")