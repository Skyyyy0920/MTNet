import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class TrajectoryTrainDataset(Dataset):
    def __init__(self, data_df, map_set):
        user_id2idx_dict, POI_id2idx_dict, POI_idx2cat_idx_dict = map_set
        self.trajectories = {}

        data_df = data_df.groupby(['trajectory_id']).filter(lambda x: len(x) > 2)
        data_df['local_time'] = pd.to_datetime(data_df['local_time'])  # convert time column to datetime format
        data_df = data_df.sort_values(['user_id', 'local_time'])  # sort

        for trajectory_id, trajectory in tqdm(data_df.groupby('trajectory_id'), desc=f"Prepare train dataset"):
            user_id = trajectory_id.split('_')[0]
            user_idx = user_id2idx_dict[user_id]
            traj_idx = len(self.trajectories)
            self.trajectories[traj_idx] = []

            for index in range(len(trajectory) - 1):
                _, pid, _, _, _, _, _, _, _, local_tim, _, _, _, _, _, coo = trajectory.iloc[index]
                _, next_pid, _, _, _, _, _, _, _, _, _, _, _, _, _, next_coo = trajectory.iloc[index + 1]
                POI_idx = POI_id2idx_dict[pid]
                cat_idx = POI_idx2cat_idx_dict[POI_idx]
                next_POI_idx = POI_id2idx_dict[next_pid]
                next_cat_idx = POI_idx2cat_idx_dict[next_POI_idx]
                features = [user_idx, POI_idx, cat_idx, local_tim.hour, coo]
                labels = [next_POI_idx, next_cat_idx, next_coo]
                checkin = {'features': features, 'labels': labels}
                self.trajectories[traj_idx].append(checkin)

        print(f"Train dataset length: ", len(self.trajectories))

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, item):
        return self.trajectories[item]


class TrajectoryValDataset(Dataset):
    def __init__(self, data_df, map_set):
        user_id2idx_dict, POI_id2idx_dict, POI_idx2cat_idx_dict = map_set
        self.trajectories = {}

        data_df['user_id'] = data_df['user_id'].astype(str)
        data_df = data_df[data_df['user_id'].isin(user_id2idx_dict.keys())]
        data_df = data_df[data_df['POI_id'].isin(POI_id2idx_dict.keys())]  # Do the same as GETNext
        data_df = data_df.groupby(['trajectory_id']).filter(lambda x: len(x) > 2)

        data_df['local_time'] = pd.to_datetime(data_df['local_time'])  # convert time column to datetime format
        data_df = data_df.sort_values(['user_id', 'local_time'])  # sort

        for trajectory_id, trajectory in tqdm(data_df.groupby('trajectory_id'), desc=f"Prepare val dataset"):
            user_id = trajectory_id.split('_')[0]
            user_idx = user_id2idx_dict[user_id]
            traj_idx = len(self.trajectories)
            self.trajectories[traj_idx] = []

            for index in range(len(trajectory) - 1):
                _, pid, _, _, _, _, _, _, _, local_tim, _, _, _, _, _, coo = trajectory.iloc[index]
                _, next_pid, _, _, _, _, _, _, _, _, _, _, _, _, _, next_coo = trajectory.iloc[index + 1]
                POI_idx = POI_id2idx_dict[pid]
                cat_idx = POI_idx2cat_idx_dict[POI_idx]
                next_POI_idx = POI_id2idx_dict[next_pid]
                next_cat_idx = POI_idx2cat_idx_dict[next_POI_idx]
                features = [user_idx, POI_idx, cat_idx, local_tim.hour, coo]
                labels = [next_POI_idx, next_cat_idx, next_coo]
                checkin = {'features': features, 'labels': labels}
                self.trajectories[traj_idx].append(checkin)

        print(f"Val dataset length: ", len(self.trajectories))

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, item):
        return self.trajectories[item]


class TrajectoryTestDataset(Dataset):
    def __init__(self, data_df, map_set):
        user_id2idx_dict, POI_id2idx_dict, POI_idx2cat_idx_dict = map_set
        self.trajectories = {}

        data_df['user_id'] = data_df['user_id'].astype(str)
        data_df = data_df[data_df['user_id'].isin(user_id2idx_dict.keys())]
        data_df = data_df[data_df['POI_id'].isin(POI_id2idx_dict.keys())]  # Do the same as GETNext
        data_df = data_df.groupby(['trajectory_id']).filter(lambda x: len(x) > 2)

        data_df['local_time'] = pd.to_datetime(data_df['local_time'])  # convert time column to datetime format
        data_df = data_df.sort_values(['user_id', 'local_time'])  # sort

        for trajectory_id, trajectory in tqdm(data_df.groupby('trajectory_id'), desc=f"Prepare test dataset"):
            user_id = trajectory_id.split('_')[0]
            user_idx = user_id2idx_dict[user_id]
            traj_idx = len(self.trajectories)
            self.trajectories[traj_idx] = []

            for index in range(len(trajectory) - 1):
                _, pid, _, _, _, _, _, _, _, local_tim, _, _, _, _, _, coo = trajectory.iloc[index]
                _, next_pid, _, _, _, _, _, _, _, _, _, _, _, _, _, next_coo = trajectory.iloc[index + 1]
                POI_idx = POI_id2idx_dict[pid]
                cat_idx = POI_idx2cat_idx_dict[POI_idx]
                next_POI_idx = POI_id2idx_dict[next_pid]
                next_cat_idx = POI_idx2cat_idx_dict[next_POI_idx]
                features = [user_idx, POI_idx, cat_idx, local_tim.hour, coo]
                if index == len(trajectory) - 2:
                    labels = [next_POI_idx, next_cat_idx, next_coo]
                else:
                    labels = [-1, -1, -1]
                checkin = {'features': features, 'labels': labels}
                self.trajectories[traj_idx].append(checkin)

        print(f"Test dataset length: ", len(self.trajectories))

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, item):
        return self.trajectories[item]
