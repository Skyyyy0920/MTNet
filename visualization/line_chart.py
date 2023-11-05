import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

figure = plt.figure(figsize=(18, 10))
sns.set(style='whitegrid', color_codes=True)
plt.gcf().subplots_adjust(bottom=0.28, wspace=0.3)

plt.subplot(121)
x = ['0', '0.1', '0.5', '1', '2', '5']
# data = [[0.1441, 0.3198, 0.3961, 0.4753, 0.2283],
#         [0.1435, 0.3198, 0.3961, 0.4747, 0.2277],
#         [0.1459, 0.321, 0.399, 0.4777, 0.2288],
#         [0.1477, 0.3228, 0.3967, 0.4699, 0.2319],
#         [0.1429, 0.3198, 0.402, 0.4759, 0.2271],
#         [0.14, 0.3192, 0.3985, 0.4723, 0.2249]]
data = [[0.3961, 0.4753],
        [0.3961, 0.4747],
        [0.399, 0.4777],
        [0.3967, 0.4699],
        [0.402, 0.4759],
        [0.3985, 0.4723]]
data = np.array(data)
# wide_df = pd.DataFrame(data, x, ['Acc@1', 'Acc@5', 'Acc@10', 'Acc@20', 'MRR'])
wide_df = pd.DataFrame(data, x, ['Acc@10', 'Acc@20'])
ax2 = sns.lineplot(data=wide_df, markers=True, markersize=15, palette=["blue", "red"])
plt.xlabel('η', fontdict={'family': 'Times New Roman', 'size': 43})
ax2.set_ylabel('Accuracy', fontsize=32)
plt.yticks(fontproperties='Times New Roman', size=30)
plt.xticks(fontproperties='Times New Roman', size=30)
plt.legend(ncol=1, loc="lower right", columnspacing=0.5, prop={'family': 'Times New Roman', 'size': 24})
plt.ylim(0.35, 0.5)

plt.subplot(122)
x = ['0', '0.1', '0.5', '1', '2', '5']
data = [[0.3919, 0.4675],
        [0.3925, 0.4687],
        [0.3961, 0.4735],
        [0.3967, 0.4699],
        [0.3996, 0.4771],
        [0.3961, 0.4777]]
data = np.array(data)
wide_df = pd.DataFrame(data, x, ['Acc@10', 'Acc@20'])
ax1 = sns.lineplot(data=wide_df, markers=True, markersize=15, palette=["blue", "red"])
ax1.set_ylabel('Accuracy', fontsize=32)
plt.xlabel('δ', fontdict={'family': 'Times New Roman', 'size': 43})
plt.yticks(fontproperties='Times New Roman', size=30)
plt.xticks(fontproperties='Times New Roman', size=30)
plt.legend(ncol=1, loc="lower right", columnspacing=0.5, prop={'family': 'Times New Roman', 'size': 24})
plt.ylim(0.35, 0.5)

plt.show()
figure.savefig('CA.pdf')

figure = plt.figure(figsize=(18, 10))
sns.set(style='whitegrid', color_codes=True)
plt.gcf().subplots_adjust(bottom=0.28, wspace=0.3)

plt.subplot(121)
x = ['0', '0.1', '0.5', '1', '2', '5']
data = [[0.6092, 0.6876],
        [0.6151, 0.6884],
        [0.6195, 0.6876],
        [0.6321, 0.6936],
        [0.6262, 0.6862],
        [0.6284, 0.6876]]
data = np.array(data)
wide_df = pd.DataFrame(data, x, ['Acc@10', 'Acc@20'])
ax2 = sns.lineplot(data=wide_df, markers=True, markersize=15, palette=["blue", "red"])
plt.xlabel('η', fontdict={'family': 'Times New Roman', 'size': 43})
ax2.set_ylabel('Accuracy', fontsize=32)
plt.yticks(fontproperties='Times New Roman', size=30)
plt.xticks(fontproperties='Times New Roman', size=30)
plt.legend(ncol=1, loc="lower right", columnspacing=0.5, prop={'family': 'Times New Roman', 'size': 24})
plt.ylim(0.6, 0.7)

plt.subplot(122)
x = ['0', '0.1', '0.5', '1', '2', '5']
data = [[0.6225, 0.6899],
        [0.6218, 0.6899],
        [0.6232, 0.6869],
        [0.6321, 0.6936],
        [0.624, 0.6876],
        [0.621, 0.6854]]
data = np.array(data)
wide_df = pd.DataFrame(data, x, ['Acc@10', 'Acc@20'])
ax1 = sns.lineplot(data=wide_df, markers=True, markersize=15, palette=["blue", "red"])
ax1.set_ylabel('Accuracy', fontsize=32)
plt.xlabel('δ', fontdict={'family': 'Times New Roman', 'size': 43})
plt.yticks(fontproperties='Times New Roman', size=30)
plt.xticks(fontproperties='Times New Roman', size=30)
plt.legend(ncol=1, loc="lower right", columnspacing=0.5, prop={'family': 'Times New Roman', 'size': 24})
plt.ylim(0.6, 0.7)

plt.show()
figure.savefig('NYC.pdf')

figure = plt.figure(figsize=(18, 10))
sns.set(style='whitegrid', color_codes=True)
plt.gcf().subplots_adjust(bottom=0.28, wspace=0.3)

plt.subplot(121)
x = ['0', '0.1', '0.5', '1', '2', '5']
data = [[0.5608, 0.6374],
        [0.5635, 0.6389],
        [0.5728, 0.6446],
        [0.5739, 0.6416],
        [0.5756, 0.6467],
        [0.5718, 0.6452]]
data = np.array(data)
# wide_df = pd.DataFrame(data, x, ['Acc@1', 'Acc@5', 'Acc@10', 'Acc@20', 'MRR'])
wide_df = pd.DataFrame(data, x, ['Acc@10', 'Acc@20'])
ax2 = sns.lineplot(data=wide_df, markers=True, markersize=15, palette=["blue", "red"])
plt.xlabel('η', fontdict={'family': 'Times New Roman', 'size': 43})
ax2.set_ylabel('Accuracy', fontsize=32)
plt.yticks(fontproperties='Times New Roman', size=30)
plt.xticks(fontproperties='Times New Roman', size=30)
plt.legend(ncol=1, loc="best", columnspacing=0.5, prop={'family': 'Times New Roman', 'size': 24})
plt.ylim(0.55, 0.65)

plt.subplot(122)
x = ['0', '0.1', '0.5', '1', '2', '5']
data = [[0.5764, 0.6458],
        [0.5737, 0.6437],
        [0.5737, 0.6448],
        [0.5739, 0.6416],
        [0.5707, 0.6452],
        [0.5678, 0.6433]]
data = np.array(data)
wide_df = pd.DataFrame(data, x, ['Acc@10', 'Acc@20'])
ax1 = sns.lineplot(data=wide_df, markers=True, markersize=15, palette=["blue", "red"])
ax1.set_ylabel('Accuracy', fontsize=32)
plt.xlabel('δ', fontdict={'family': 'Times New Roman', 'size': 43})
plt.yticks(fontproperties='Times New Roman', size=30)
plt.xticks(fontproperties='Times New Roman', size=30)
plt.legend(ncol=1, loc="best", columnspacing=0.5, prop={'family': 'Times New Roman', 'size': 24})
plt.ylim(0.55, 0.65)

plt.show()
figure.savefig('TKY.pdf')
