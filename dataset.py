import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class TrajectoryTrainDataset(Dataset):
    def __init__(self, data_df, map_set):
        user_id2idx_dict, POI_id2idx_dict, cat_id2idx_dict = map_set
        self.trajectories = {}

        data_df = data_df.groupby(['trajectory_id']).filter(lambda x: len(x) > 2)
        data_df['local_time'] = pd.to_datetime(data_df['local_time'])  # convert time column to datetime format
        data_df = data_df.sort_values(['user_id', 'local_time'])  # sort

        max_ = 0

        for trajectory_id, trajectory in tqdm(data_df.groupby('trajectory_id'), desc=f"Prepare training dataset"):
            user_id = trajectory_id.split('_')[0]
            user_idx = user_id2idx_dict[user_id]
            traj_idx = len(self.trajectories)
            self.trajectories[traj_idx] = []
            start_date = trajectory.iloc[0]['local_time']
            cur_day_of_year = start_date.day_of_year
            self.trajectories[traj_idx].append([])
            record = {}
            record[0] = 0
            record[1] = 0
            record[2] = 0
            record[3] = 0
            record[4] = 0

            for index in range(len(trajectory) - 1):
                _, pid, cid, _, _, _, _, _, _, tim, _, _, _, _, _, coo = trajectory.iloc[index]
                _, next_pid, next_cid, _, _, _, _, _, _, next_tim, _, _, _, _, _, next_coo = trajectory.iloc[index + 1]
                POI_idx, cat_idx = POI_id2idx_dict[pid], cat_id2idx_dict[cid]
                next_POI_idx, next_cat_idx = POI_id2idx_dict[next_pid], cat_id2idx_dict[next_cid]
                features = [user_idx, POI_idx, cat_idx, coo]
                tim_info = int((tim - start_date).days * 24 + (tim - start_date).seconds / 60 / 60)
                labels = [next_POI_idx, next_cat_idx, next_coo]
                checkin = {'features': features, 'time': tim_info, 'labels': labels, 't': tim}
                if tim.day_of_year == cur_day_of_year:
                    self.trajectories[traj_idx][len(self.trajectories[traj_idx]) - 1].append(checkin)
                    if tim.hour <= 6:
                        record[0] += 1
                    elif tim.hour <= 12:
                        record[1] += 1
                    elif tim.hour <= 18:
                        record[2] += 1
                    elif tim.hour <= 24:
                        record[3] += 1
                else:
                    for i in record:
                        if record[i] < max_:
                            max_ = max_
                        else:
                            max_ = record[i]
                            for i in self.trajectories[traj_idx][len(self.trajectories[traj_idx]) - 1]:
                                print(i['t'], i['features'][1])
                            print("--------------------")
                    record = {}
                    record[0] = 0
                    record[1] = 0
                    record[2] = 0
                    record[3] = 0
                    record[4] = 0
                    # max_ = max_ if len(
                    #     self.trajectories[traj_idx][len(self.trajectories[traj_idx]) - 1]) < max_ else len(
                    #     self.trajectories[traj_idx][len(self.trajectories[traj_idx]) - 1])
                    cur_day_of_year = tim.day_of_year
                    self.trajectories[traj_idx].append([checkin])

        print(f"Train dataset length: ", len(self.trajectories))
        print(max_)
        exit()

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, item):
        return self.trajectories[item]


class TrajectoryValDataset(Dataset):
    def __init__(self, data_df, map_set):
        user_id2idx_dict, POI_id2idx_dict, cat_id2idx_dict = map_set
        self.trajectories = {}

        data_df['user_id'] = data_df['user_id'].astype(str)
        data_df = data_df[data_df['user_id'].isin(user_id2idx_dict.keys())]
        data_df = data_df[data_df['POI_id'].isin(POI_id2idx_dict.keys())]  # Do the same as GETNext
        data_df = data_df[data_df['POI_catid'].isin(cat_id2idx_dict.keys())]
        data_df = data_df.groupby(['trajectory_id']).filter(lambda x: len(x) > 2)

        data_df['local_time'] = pd.to_datetime(data_df['local_time'])  # convert time column to datetime format
        data_df = data_df.sort_values(['user_id', 'local_time'])  # sort

        for trajectory_id, trajectory in tqdm(data_df.groupby('trajectory_id'), desc=f"Prepare validation dataset"):
            user_id = trajectory_id.split('_')[0]
            user_idx = user_id2idx_dict[user_id]
            traj_idx = len(self.trajectories)
            self.trajectories[traj_idx] = []
            start_date = trajectory.iloc[0]['local_time']

            for index in range(len(trajectory) - 1):
                _, pid, cid, _, _, _, _, _, _, tim, _, _, _, _, _, coo = trajectory.iloc[index]
                _, next_pid, next_cid, _, _, _, _, _, _, next_tim, _, _, _, _, _, next_coo = trajectory.iloc[index + 1]
                POI_idx, cat_idx = POI_id2idx_dict[pid], cat_id2idx_dict[cid]
                next_POI_idx, next_cat_idx = POI_id2idx_dict[next_pid], cat_id2idx_dict[next_cid]
                features = [user_idx, POI_idx, cat_idx, coo]
                tim_info = int((tim - start_date).days * 24 + (tim - start_date).seconds / 60 / 60)
                if index == len(trajectory) - 2:
                    labels = [next_POI_idx, next_cat_idx, next_coo]
                else:
                    labels = [-1, -1, -1]
                checkin = {'features': features, 'time': tim_info, 'labels': labels}
                self.trajectories[traj_idx].append(checkin)

        print(f"Validation dataset length: ", len(self.trajectories))

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, item):
        return self.trajectories[item]


class TrajectoryTestDataset(Dataset):
    def __init__(self, data_df, map_set):
        user_id2idx_dict, POI_id2idx_dict, cat_id2idx_dict = map_set
        self.trajectories = {}

        data_df['user_id'] = data_df['user_id'].astype(str)
        data_df = data_df[data_df['user_id'].isin(user_id2idx_dict.keys())]
        data_df = data_df[data_df['POI_id'].isin(POI_id2idx_dict.keys())]  # Do the same as GETNext
        data_df = data_df[data_df['POI_catid'].isin(cat_id2idx_dict.keys())]
        data_df = data_df.groupby(['trajectory_id']).filter(lambda x: len(x) > 2)

        data_df['local_time'] = pd.to_datetime(data_df['local_time'])  # convert time column to datetime format
        data_df = data_df.sort_values(['user_id', 'local_time'])  # sort

        for trajectory_id, trajectory in tqdm(data_df.groupby('trajectory_id'), desc=f"Prepare testing dataset"):
            user_id = trajectory_id.split('_')[0]
            user_idx = user_id2idx_dict[user_id]
            traj_idx = len(self.trajectories)
            self.trajectories[traj_idx] = []
            start_date = trajectory.iloc[0]['local_time']

            for index in range(len(trajectory) - 1):
                _, pid, cid, _, _, _, _, _, _, tim, _, _, _, _, _, coo = trajectory.iloc[index]
                _, next_pid, next_cid, _, _, _, _, _, _, next_tim, _, _, _, _, _, next_coo = trajectory.iloc[index + 1]
                POI_idx, cat_idx = POI_id2idx_dict[pid], cat_id2idx_dict[cid]
                next_POI_idx, next_cat_idx = POI_id2idx_dict[next_pid], cat_id2idx_dict[next_cid]
                features = [user_idx, POI_idx, cat_idx, coo]
                tim_info = int((tim - start_date).days * 24 + (tim - start_date).seconds / 60 / 60)
                if index == len(trajectory) - 2:
                    labels = [next_POI_idx, next_cat_idx, next_coo]
                else:
                    labels = [-1, -1, -1]
                checkin = {'features': features, 'time': tim_info, 'labels': labels}
                self.trajectories[traj_idx].append(checkin)

        print(f"Test dataset length: ", len(self.trajectories))

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, item):
        return self.trajectories[item]
