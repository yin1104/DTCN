import torch
import numpy as np
from algorithm.Ablation.DTCN import DTCNet
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random

seed = 1234
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

duration = 1
block_index = 5  # 用最后一块进行测试
step_num = 1
s_len = 25
all_block = [4, 5]
num_bands = 1
fs = 250


def test_all(test_length):
    all_data = []  # (1680, 11, 250)
    all_labels = []  # (1680,)
    for ts in range(test_length):
        subject_file = '../rawData/Benchmark/S'+str(ts+1)+'.mat'
        rawdata = loadmat(subject_file)
        rawdata = rawdata['eeg']
        for j in all_block:  # 用第六个全都没训练过的块测试效果
            block_index = j
            for k in range(40):  # 40个target
                target_index = k
                channel_data = rawdata[:, int(160):int(160 + duration * fs), target_index, block_index]
                channel_data = np.array(channel_data)
                all_labels.append(target_index)
                all_data.append(channel_data)

    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    return all_data, all_labels


if __name__ == "__main__":

    all_data, all_labels = test_all(35)
    # print(all_data.shape)  # (2800, 3, 11, 250),(1350, 3, 8, 250), (360, 3, 8, 256)
    data_length = len(all_data)
    choose = 3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DTCNet(
        num_channel=11,
        num_classes=40,
        signal_length=int(fs * duration),
    )
    model_path = '../model/DTCNet_pretrain_' + str(duration) + 's_S' + str(choose) + '.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    all_data = torch.Tensor(all_data)
    all_labels = torch.Tensor(all_labels)
    # t-SNE可视化层输出
    model.eval()
    with torch.no_grad():
        outputs = model(all_data)
    target = outputs.argmax(axis=1)
    # 层输出可视化
    accuracy = (target == all_labels).sum()
    all_acc = float(accuracy / data_length * 100)
    # _______________T-SNE____________________________________
    print("accuracy: {}% \n\t".format(all_acc))
    # 层输出可视化
    conv1_out = np.array(model.features[0]).reshape(data_length, -1)
    tsn = TSNE(n_components=2, init='pca', random_state=1024)

    X_tsne = tsn.fit_transform(conv1_out)
    # MinMax 归一化
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # Normalize

    # plt.style.use('seaborn-white')
    plt.figure(figsize=(12, 12))
    fig, ax = plt.subplots()
    # plt.subplot(121)
    # scatter = ax.scatter(X_norm[:, 0], X_norm[:, 1], c=target, label="out", cmap='Spectral_r', s=5)
    scatter = ax.scatter(X_norm[:, 0], X_norm[:, 1], c=target, label="out", cmap='jet', s=8)
    # plt.title('Model_Input', fontdict={'size': 15, 'color': 'black'})
    # fig.colorbar(scatter, ticks=np.linspace(1, 40, 40), fontdict={'size': 5, 'color': 'black'})
    fig.colorbar(scatter)
    # plt.savefig("model/out.jpg", dpi=300)  # 保存图像结果
    plt.savefig("tsne0.svg", transparent=True,  bbox_inches='tight', format="svg")
    plt.show()
    # _______________T-SNE____________________________________
