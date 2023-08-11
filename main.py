import yaml
import time
import pickle
import zipfile
import logging
from sklearn.cluster import KMeans
from model import *
from utils import *
from config import *
from dataset import *

if __name__ == '__main__':
    # ==================================================================================================
    # 1. Get experiment args and seed
    # ==================================================================================================
    print('\n' + '=' * 36 + ' Get experiment args ' + '=' * 36)
    args = get_args()
    print(f"Using device: {args.device}")
    setup_seed(args.seed)  # make the experiment repeatable

    # ==================================================================================================
    # 2. Setup logger
    # ==================================================================================================
    print('\n' + '=' * 36 + ' Setup logger ' + '=' * 36)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging_time = time.strftime('%m-%d_%H-%M', time.localtime())
    save_dir = os.path.join(args.save_path, f"{args.dataset}_{args.nary}-ary_{logging_time}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(f"Saving path: {save_dir}")
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s %(levelname)s]%(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=os.path.join(save_dir, f'{args.dataset}_{args.nary}.log'))
    console = logging.StreamHandler()  # Simultaneously output to console
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(fmt='[%(asctime)s %(levelname)s]%(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logging.getLogger('').addHandler(console)
    logging.getLogger('matplotlib.font_manager').disabled = True

    # ==================================================================================================
    # 3. Save codes and settings
    # ==================================================================================================
    print('\n' + '=' * 36 + ' Save codes and settings ' + '=' * 36)
    zipf = zipfile.ZipFile(file=os.path.join(save_dir, 'codes.zip'), mode='a', compression=zipfile.ZIP_DEFLATED)
    zipdir(Path().absolute(), zipf, include_format=['.py'])
    zipf.close()
    with open(os.path.join(save_dir, 'args.yml'), 'a') as f:
        yaml.dump(vars(args), f, sort_keys=False)

    # ==================================================================================================
    # 4. Prepare data
    # ==================================================================================================
    print('\n' + '=' * 36 + ' Prepare data ' + '=' * 36)
    # Load data
    train_df = pd.read_csv(f'dataset/{args.dataset}/{args.dataset}_train.csv')
    val_df = pd.read_csv(f'dataset/{args.dataset}/{args.dataset}_val.csv')
    test_df = pd.read_csv(f'dataset/{args.dataset}/{args.dataset}_test.csv')

    # with open('dataset/NYC/NYC_train.csv', 'r') as f:
    #     checkin = f.readlines()
    # checkin = checkin[1:]
    # # poi_data = {}
    # traj_data = []
    # for i in checkin:
    #     data = i.replace('\n', '').split(',')
    #     traj = str(data[12])
    #     traj_data.append(traj.split('_'))
    # uesr = {}
    # for i in range(len(traj_data)):
    #     if not (int(traj_data[i][0]) in uesr):
    #         uesr[int(traj_data[i][0])] = int(traj_data[i][1])
    #     elif uesr[int(traj_data[i][0])] < int(traj_data[i][1]):
    #         uesr[int(traj_data[i][0])] = int(traj_data[i][1])
    # count = 0
    # user_ids = []
    # for k in uesr.keys():
    #     if uesr[k] < 13:
    #         count += 1
    #         user_ids.append(k)
    # print(count)
    # print(len(user_ids))
    # print(user_ids)
    # test_df = test_df[test_df['user_id'].isin(user_ids)]

    # grouped_train_df = train_df.groupby('user_id')
    # user_ids = []
    # for user_id, group in grouped_train_df:
    #     # if 4 <= len(group.groupby('trajectory_id')) <= 20:
    #     # if len(group.groupby('trajectory_id')) > 25:
    #     # if len(group.groupby('trajectory_id')) < 6:
    #     # if len(group.groupby('trajectory_id')) > 41:
    #     # if 13 <= len(group.groupby('trajectory_id')) <= 41:
    #     if len(group.groupby('trajectory_id')) < 13:
    #     # if len(group) > 41:
    #         user_ids.append(user_id)
    # print(len(user_ids))
    # print(user_ids)
    # test_df = test_df[test_df['user_id'].isin(user_ids)]

    if args.dataset == 'Gowalla-CA':
        train_df = process_for_GowallaCA(train_df)
        val_df = process_for_GowallaCA(val_df)
        test_df = process_for_GowallaCA(test_df)

    # User id to index
    uid_list = [str(uid) for uid in list(set(train_df['user_id'].to_list()))]
    user_id2idx_dict = dict(zip(uid_list, range(len(uid_list))))
    # POI id to index
    POI_list = list(set(train_df['POI_id'].tolist()))
    POI_list.sort()
    POI_id2idx_dict = dict(zip(POI_list, range(len(POI_list))))
    # Cat id to index
    cat_list = list(set(train_df['POI_catid'].tolist()))
    cat_list.sort()
    cat_id2idx_dict = dict(zip(cat_list, range(len(cat_list))))

    user_counts = train_df['user_id'].value_counts()
    top_ten_users = user_counts.head(10)
    for user, counts in top_ten_users.items():
        print(user_id2idx_dict[str(user)], counts)

    data_train = np.column_stack((train_df['longitude'], train_df['latitude']))
    kmeans_train = KMeans(n_clusters=args.K_cluster)
    kmeans_train.fit(data_train)
    train_df['coo_label'] = kmeans_train.labels_
    data_val = np.column_stack((val_df['longitude'], val_df['latitude']))
    val_df['coo_label'] = kmeans_train.predict(data_val)
    data_test = np.column_stack((test_df['longitude'], test_df['latitude']))
    test_df['coo_label'] = kmeans_train.predict(data_test)

    num_users = len(user_id2idx_dict)
    num_POIs = len(POI_id2idx_dict)
    num_cats = len(cat_id2idx_dict)
    print(f"users: {num_users}, POIs: {num_POIs}, cats: {num_cats}, coos: {args.K_cluster}")

    # Build dataset
    map_set = (user_id2idx_dict, POI_id2idx_dict, cat_id2idx_dict)
    train_dataset = TrajectoryTrainDataset(train_df, map_set)
    val_dataset = TrajectoryValDataset(val_df, map_set)
    test_dataset = TrajectoryTestDataset(test_df, map_set)
    batch_size = int(args.batch_size / args.accumulation_steps)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                                  pin_memory=True, num_workers=args.workers, collate_fn=lambda x: x)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                                pin_memory=True, num_workers=args.workers, collate_fn=lambda x: x)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                                 pin_memory=True, num_workers=args.workers, collate_fn=lambda x: x)

    # ==================================================================================================
    # 5. Build models, define overall loss and optimizer
    # ==================================================================================================
    print('\n' + '=' * 36 + ' Build models ' + '=' * 36)

    TreeLSTM_model = TreeLSTM(h_size=args.h_size, nary=args.nary,
                              embed_dropout=args.embed_dropout, model_dropout=args.model_dropout,
                              num_users=num_users, user_embed_dim=args.user_embed_dim,
                              num_POIs=num_POIs, POI_embed_dim=args.POI_embed_dim,
                              num_cats=num_cats, cat_embed_dim=args.cat_embed_dim,
                              num_coos=args.K_cluster, coo_embed_dim=args.coo_embed_dim,
                              device=args.device).to(device=args.device)
    multi_task_loss = MultiTaskLoss(3).to(device=args.device)

    criterion_POI = nn.CrossEntropyLoss(ignore_index=-1)  # -1 is ignored
    criterion_cat = nn.CrossEntropyLoss(ignore_index=-1)
    criterion_coo = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(params=list(TreeLSTM_model.parameters())
                                        + list(multi_task_loss.parameters()),
                                 lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    # ==================================================================================================
    # 6. Load pre-trained model
    # ==================================================================================================
    if args.load_path:
        print('\nLoad pre-trained model...')
        checkpoint = torch.load(os.path.join(args.load_path, f"checkpoint_50.pth"))
        TreeLSTM_model.load_state_dict(checkpoint['model_state'])
        multi_task_loss.load_state_dict(checkpoint['multi_task_loss_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state'])

    logging.info(f"\n{TreeLSTM_model}")
    logging.info(f"\n{optimizer}")

    # ==================================================================================================
    # 7. Training
    # ==================================================================================================
    print('\n' + '=' * 36 + ' Start training ' + '=' * 36)

    current_patience = 0
    best_validation_loss = float('inf')
    early_stopping_flag = False

    # Training loop
    for epoch in range(args.epochs):
        TreeLSTM_model.train()
        TreeLSTM_model.cell_IAC.train()
        TreeLSTM_model.cell_IRC.train()
        multi_task_loss.train()

        loss_list = []

        h_2_save, label_save = {}, {}
        for i in range(3):
            h_2_save[i] = np.empty((0, 512), dtype=np.float32)
            label_save[i] = []

        for b_idx, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Training"):
            MT_batcher = []
            for trajectory, label in batch:
                mobility_tree = construct_MobilityTree(trajectory, label, args.nary, args.plot_tree)
                MT_batcher.append(mobility_tree.to(args.device))

            MT_batch = dgl.batch(MT_batcher).to(args.device)
            MT_input = SSTBatch(graph=MT_batch,
                                features=MT_batch.ndata["x"].to(args.device),
                                time=MT_batch.ndata["time"].to(args.device),
                                label=MT_batch.ndata["y"].to(args.device),
                                mask=MT_batch.ndata["mask"].to(args.device),
                                mask2=MT_batch.ndata["mask2"].to(args.device),
                                type=MT_batch.ndata["type"].to(args.device))

            y_pred_POI, y_pred_cat, y_pred_coo, h_2_out, label_out = TreeLSTM_model(MT_input, epoch)
            y_POI, y_cat, y_coo = MT_input.label[:, 0], MT_input.label[:, 1], MT_input.label[:, 2]

            for i in range(3):
                if i in h_2_out.keys():
                    h_2_save[i] = np.vstack((h_2_save[i], h_2_out[i]))
                    label_save[i].append(label_out[i])

            loss_POI = criterion_POI(y_pred_POI, y_POI.long())
            loss_cat = criterion_cat(y_pred_cat, y_cat.long())
            loss_coo = criterion_coo(y_pred_coo, y_coo.long())
            loss = multi_task_loss(loss_POI, loss_cat, loss_coo)
            loss_list.append(loss.item())
            loss.backward()

            # Gradient accumulation to solve the GPU memory problem
            if (b_idx + 1) % args.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        lr_scheduler.step()  # update learning rate

        if epoch >= 0:
            for i in range(3):
                label_save[i] = np.concatenate(label_save[i])
                np.save(rf'{i}_h2.npy', h_2_save[i])
                np.save(rf'{i}_category.npy', label_save[i])

        # Logging
        logging.info(f"************************  Training epoch: {epoch + 1}/{args.epochs}  ************************")
        logging.info(f"Current epoch's mean loss: {np.mean(loss_list)}\t\tLr: {optimizer.param_groups[0]['lr']}"
                     f"\t\tMulti-loss weight: {multi_task_loss.params}")

        # ==================================================================================================
        # 8. Validation and Testing
        # ==================================================================================================
        TreeLSTM_model.eval()
        TreeLSTM_model.cell_IAC.eval()
        TreeLSTM_model.cell_IRC.eval()
        multi_task_loss.eval()

        with torch.no_grad():
            # ==================================================================================================
            # 8.1 Validation
            # ==================================================================================================
            loss_list = []

            for batch in val_dataloader:
                MT_batcher = []
                for trajectory, label in batch:
                    mobility_tree = construct_MobilityTree(trajectory, label, args.nary, args.plot_tree)
                    MT_batcher.append(mobility_tree.to(args.device))

                MT_batch = dgl.batch(MT_batcher).to(args.device)
                MT_input = SSTBatch(graph=MT_batch,
                                    features=MT_batch.ndata["x"].to(args.device),
                                    time=MT_batch.ndata["time"].to(args.device),
                                    label=MT_batch.ndata["y"].to(args.device),
                                    mask=MT_batch.ndata["mask"].to(args.device),
                                    mask2=MT_batch.ndata["mask2"].to(args.device),
                                    type=MT_batch.ndata["type"].to(args.device))

                y_pred_POI, y_pred_cat, y_pred_coo, _, _ = TreeLSTM_model(MT_input)
                y_POI, y_cat, y_coo = MT_input.label[:, 0], MT_input.label[:, 1], MT_input.label[:, 2]

                loss = criterion_POI(y_pred_POI, y_POI.long())
                loss_list.append(loss.item())

            validation_loss = np.mean(loss_list)
            logging.info(f"-------------------------------- Validation --------------------------------")
            logging.info(f"Current epoch's mean loss: {validation_loss}, best validation loss: {best_validation_loss}")
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                current_patience = 0
            else:
                current_patience += 1
                if current_patience >= args.patience:
                    if args.save_model:
                        # Save model
                        checkpoint = {
                            'model_state': TreeLSTM_model.state_dict(),
                            'multi_task_loss_state': multi_task_loss.state_dict(),
                            'optimizer_state': optimizer.state_dict(),
                            'lr_scheduler_state': lr_scheduler.state_dict()
                        }
                        torch.save(checkpoint, os.path.join(save_dir, f"checkpoint_{epoch + 1}.pth"))
                    logging.info(f"Early stopping at epoch {epoch + 1}...")
                    early_stopping_flag = True

            # ==================================================================================================
            # 8.2 Testing
            # ==================================================================================================
            y_pred_POI_list, y_label_POI_list = [], []
            y_pred_cat_list, y_label_cat_list = [], []
            y_pred_coo_list, y_label_coo_list = [], []
            # Start testing
            for batch in test_dataloader:
                MT_batcher = []
                for trajectory, label in batch:
                    mobility_tree = construct_MobilityTree(trajectory, label, args.nary, args.plot_tree)
                    MT_batcher.append(mobility_tree.to(args.device))

                MT_batch = dgl.batch(MT_batcher).to(args.device)
                MT_input = SSTBatch(graph=MT_batch,
                                    features=MT_batch.ndata["x"].to(args.device),
                                    time=MT_batch.ndata["time"].to(args.device),
                                    label=MT_batch.ndata["y"].to(args.device),
                                    mask=MT_batch.ndata["mask"].to(args.device),
                                    mask2=MT_batch.ndata["mask2"].to(args.device),
                                    type=MT_batch.ndata["type"].to(args.device))

                y_pred_POI, y_pred_cat, y_pred_coo, _, _ = TreeLSTM_model(MT_input)
                y_POI, y_cat, y_coo = MT_input.label[:, 0], MT_input.label[:, 1], MT_input.label[:, 2]

                y_pred_POI_list.append(y_pred_POI.detach().cpu().numpy())
                y_label_POI_list.append(y_POI.detach().cpu().numpy())
                y_pred_cat_list.append(y_pred_cat.detach().cpu().numpy())
                y_label_cat_list.append(y_cat.detach().cpu().numpy())
                y_pred_coo_list.append(y_pred_coo.detach().cpu().numpy())
                y_label_coo_list.append(y_coo.detach().cpu().numpy())

            y_label_POI_numpy, y_pred_POI_numpy = get_pred_label(y_label_POI_list, y_pred_POI_list)
            y_label_cat_numpy, y_pred_cat_numpy = get_pred_label(y_label_cat_list, y_pred_cat_list)
            y_label_coo_numpy, y_pred_coo_numpy = get_pred_label(y_label_coo_list, y_pred_coo_list)

            # Logging
            logging.info(f"================================ Testing ================================")
            acc1, acc5, acc10, acc20, mrr = get_performance(y_label_POI_numpy, y_pred_POI_numpy)
            logging.info(f" <POI> acc@1: {acc1}\tacc@5: {acc5}\tacc@10: {acc10}\tacc@20: {acc20}\tmrr: {mrr}")
            acc1, acc5, acc10, acc20, mrr = get_performance(y_label_cat_numpy, y_pred_cat_numpy)
            logging.info(f" <cat> acc@1: {acc1}\tacc@5: {acc5}\tacc@10: {acc10}\tacc@20: {acc20}\tmrr: {mrr}")
            acc1, acc5, acc10, acc20, mrr = get_performance(y_label_coo_numpy, y_pred_coo_numpy)
            logging.info(f" <coo> acc@1: {acc1}\tacc@5: {acc5}\tacc@10: {acc10}\tacc@20: {acc20}\tmrr: {mrr}")

            if early_stopping_flag:
                if args.save_data:
                    pickle.dump(y_pred_POI_numpy, open(os.path.join(save_dir, f"recommend_list_{epoch + 1}"), 'wb'))
                    pickle.dump(y_label_POI_numpy, open(os.path.join(save_dir, f"ground_truth_{epoch + 1}"), 'wb'))
                break
