from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.size'] = 12
h, label = {}, {}
# path = 'h_trajectory'
# path = 'concat_embedding'
# path = 'concat_embedding_trajectory'
path = 'period_h'
# path = 'period_h_trajectory'
colors = ['#6c7197', '#d92405', '#3563eb', '#bf55ec', '#fbbf45', '#67f2d1']
colors = plt.cm.jet(np.linspace(0, 1, 2000))
for user in range(10):
    h[user] = np.load(rf'./{path}/{user}_h2.npy')
    label[user] = np.load(rf'./{path}/{user}_category.npy')
    print(len(h[user]), len(label[user]))

for user in range(10):
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(h[user])
    color_label = []
    for i in range(len(label[user])):
        color_label.append(colors[label[user][i] - 1])
    plt.figure(figsize=(6, 6))

    for category in range(6):
        cluster_indices = np.where(label[user] == (category + 1))
        plt.scatter(X_tsne[cluster_indices, 0], X_tsne[cluster_indices, 1], c=colors[category],
                    label=rf"{4 * category}:00 - {4 * (category + 1)}:00")

    plt.xticks([])
    plt.yticks([])

    lg = plt.legend(loc='upper right')
    lg.set_title(title="time slot", prop={'family': 'Times New Roman', 'size': 16})

    plt.savefig(rf'./{path}/user_{user}.pdf')
    plt.show()

# digits = load_digits()
# X_tsne = TSNE(n_components=2, random_state=33).fit_transform(digits.data)
# X_pca = PCA(n_components=2).fit_transform(digits.data)
#
# ckpt_dir = "images"
# if not os.path.exists(ckpt_dir):
#     os.makedirs(ckpt_dir)
#
# plt.figure(figsize=(10, 5))
# plt.subplot(121)
# plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=digits.target, label="t-SNE")
# plt.legend()
# plt.subplot(122)
# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=digits.target, label="PCA")
# plt.legend()
# plt.savefig('images/digits_tsne-pca.png', dpi=120)
# plt.show()
