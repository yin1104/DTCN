import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

Pretrain = pd.read_csv('pretrain.csv')
Retrain = pd.read_csv('retrain.csv')

ground_truth = Pretrain['Ground Truth']
pretrain_predict = Pretrain['Predict Answer']
retrain_truth = Retrain['Ground Truth']
retrain_predict = Retrain['Predict Answer']

relabel = pd.read_excel('0430relabel.xlsx')
true_label = relabel['true_label']
re_label = relabel['re_label']

new_ground_truth = np.zeros(2800)
new_retrain_truth = np.zeros(2800)
new_pretrain_predict = np.zeros(2800)
new_retrain_predict = np.zeros(2800)
for i in range(2800):
    new_ground_truth[i] = re_label[int(ground_truth[i])]
    new_retrain_truth[i] = re_label[int(retrain_truth[i])]
    new_pretrain_predict[i] = re_label[int(pretrain_predict[i])]
    new_retrain_predict[i] = re_label[int(retrain_predict[i])]

_FREQS = [8.0, 8.2, 8.4, 8.6, 8.8,
          9.0, 9.2, 9.4, 9.6, 9.8,
          10.0, 10.2, 10.4, 10.6, 10.8,
          11.0, 11.2, 11.4, 11.6, 11.8,
          12.0, 12.2, 12.4, 12.6, 12.8,
          13.0, 13.2, 13.4, 13.6, 13.8,
          14.0, 14.2, 14.4, 14.6, 14.8,
          15.0, 15.2, 15.4, 15.6, 15.8]

map = sns.color_palette("icefire", as_cmap=True)
# # 校准前
# matrix = confusion_matrix(new_ground_truth, new_pretrain_predict)
# x_tick = _FREQS
# y_tick = _FREQS
# pd_data = pd.DataFrame(matrix, index=y_tick, columns=x_tick)
# f, ax = plt.subplots(figsize=(10, 8))
# F2 = sns.heatmap(pd_data, annot=False, linewidths=0.5, cmap=map, fmt=".0f", ax=ax,
#                  xticklabels=False, yticklabels=False, vmax=70, vmin=0)
# cbar = F2.collections[0].colorbar
# cbar.ax.tick_params(labelsize=20)
# # plt.rcParams['font.sans-serif']=['SimHei','Times New Roman']
# plt.rcParams['axes.unicode_minus'] = False
# plt.title("UI", fontdict={'size': 20, 'color': 'black'})  # 标题
# plt.ylabel("Ground Truth", fontsize=20, color='k')
# plt.xlabel("Predict Answer", fontsize=20, color='k')
# plt.savefig("pretrain.svg", dpi=300, format="svg", transparent=True, bbox_inches='tight')
# # plt.show()

# 校准后
matrix = confusion_matrix(new_retrain_truth, new_retrain_predict)
print(sum(new_retrain_truth == new_retrain_predict))
x_tick = _FREQS
y_tick = _FREQS
pd_data = pd.DataFrame(matrix, index=y_tick, columns=x_tick)
f, ax = plt.subplots(figsize=(10, 8))
F2 = sns.heatmap(pd_data, annot=False, linewidths=0.5, cmap=map, fmt=".0f", ax=ax,
                 xticklabels=False, yticklabels=False, vmax=70, vmin=0)
cbar = F2.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)
# plt.rcParams['font.sans-serif']=['SimHei','Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.title("UD", fontdict={'size': 20, 'color': 'black'})  # 标题
plt.ylabel("Ground Truth", fontsize=20, color='k')
plt.xlabel("Predict Answer", fontsize=20, color='k')
plt.savefig("retrain.svg", dpi=300, format="svg", transparent=True, bbox_inches='tight')
# plt.show()