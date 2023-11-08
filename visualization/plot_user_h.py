# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# import numpy as np
#
# h, label = {}, {}
# colors = ['#6c7197', '#d92405', '#3563eb', '#bf55ec', '#fbbf45', '#67f2d1', '#cf2f74']
# plt.rcParams['font.size'] = 22
# path = 'h_user'
# time_period = 5
#
# for user in range(7):
#     h[user] = np.load(rf'./{path}/{user}_h2.npy')
#     label[user] = np.load(rf'./{path}/{user}_category.npy')
#     print(len(h[user]), len(label[user]))
#
# h_test, label_test = [], []
# user_list = [811, 1128, 1889, 200, 1223, 548, 1078]
# for i in range(7):
#     for j in range(len(h[i])):
#         if label[i][j] == time_period:
#             h_test.append(h[i][j])
#             label_test.append(i)
#
# X_tsne = TSNE(n_components=2, random_state=33).fit_transform(h_test)
#
# plt.figure(figsize=(12, 12))
#
# for user in range(7):
#     cluster_indices = np.where(np.array(label_test) == user)
#     plt.scatter(X_tsne[cluster_indices, 0], X_tsne[cluster_indices, 1], c=colors[user],
#                 label=rf"{user_list[user]}")
#
# plt.xticks([])
# plt.yticks([])
#
# lg = plt.legend(loc='upper right')
# lg.set_title(title="User ID", prop={'family': 'Times New Roman', 'size': 24})
#
# plt.savefig(rf'./7users_{time_period}.pdf')
# plt.show()

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

num = 500
h, label = {}, {}
# colors = ['#6c7197', '#d92405', '#3563eb', '#bf55ec', '#fbbf45', '#67f2d1', '#cf2f74']
colors = plt.cm.jet(np.linspace(0, 1, num))
print(colors)
plt.rcParams['font.size'] = 22
path = 'h'
time_period = 5

for user in range(num):
    h[user] = np.load(rf'./{path}/{user}_h2.npy')
    label[user] = np.load(rf'./{path}/{user}_category.npy')
    print(len(h[user]), len(label[user]))

h_test, label_test = [], []
user_list = range(num)
for i in range(num):
    for j in range(len(h[i])):
        if label[i][j] == time_period:
            h_test.append(h[i][j])
            label_test.append(i)

X_tsne = TSNE(n_components=2, random_state=33).fit_transform(h_test)

plt.figure(figsize=(48, 48))

for user in range(num):
    cluster_indices = np.where(np.array(label_test) == user)
    plt.scatter(X_tsne[cluster_indices, 0], X_tsne[cluster_indices, 1], c=colors[user],
                label=rf"{user_list[user]}")

plt.xticks([])
plt.yticks([])

# lg = plt.legend(loc='upper right')
# lg.set_title(title="User ID", prop={'family': 'Times New Roman', 'size': 24})

plt.savefig(rf'./{num}users_{time_period}.pdf')
plt.show()
