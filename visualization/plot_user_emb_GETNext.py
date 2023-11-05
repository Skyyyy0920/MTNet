from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

h, label = {}, {}
colors = ['#6c7197', '#d92405', '#3563eb', '#bf55ec', '#fbbf45', '#67f2d1', '#cf2f74']
plt.rcParams['font.size'] = 22
epoch = 62
path = rf'user_emb_period_GETNext_{epoch}'
time_period = 5
user_list = [811, 1128, 1889, 200, 1223, 548, 1078]

for user in range(len(user_list)):
    label[user] = np.load(rf'./{path}/{user_list[user]}_label.npy')
    h[user] = np.load(rf'./{path}/{user_list[user]}.npy')
    h[user] = h[user].reshape(len(label[user]), int(len(h[user]) / len(label[user])))
    print(len(h[user]), len(label[user]))

h_test, label_test = [], []
for i in range(len(user_list)):
    for j in range(len(h[i])):
        h_test.append(h[i][j])
        label_test.append(i)

X_tsne = TSNE(n_components=2, random_state=33).fit_transform(h_test)

plt.figure(figsize=(12, 12))

for user in range(7):
    cluster_indices = np.where(np.array(label_test) == user)
    plt.scatter(X_tsne[cluster_indices, 0], X_tsne[cluster_indices, 1], c=colors[user],
                label=rf"{user_list[user]}")

plt.xticks([])
plt.yticks([])

lg = plt.legend(loc='upper right')
lg.set_title(title="User ID", prop={'family': 'Times New Roman', 'size': 24})

plt.savefig(rf'./7users_{time_period}_GETNext_{epoch}.pdf')
plt.show()
