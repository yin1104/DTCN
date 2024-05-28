import numpy as np
import torch
import pandas as pd
from algorithm.DATCNet import DoubleScaleATCNet
from algorithm.Ablation.DTCN import DTCNet
from algorithm.Ablation.non_TCN import NoTCNet
from algorithm.Ablation.non_atten import NoAttenNet
from algorithm.Ablation.Singlescale_local import LocalNet
from algorithm.Ablation.Singlescale_global import GlobalNet
import random
from scipy.io import loadmat
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, accuracy_score
seed = 1234
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

duration = 1
s_len = 25
step = 1
fs = 250


def test_single(test_id):
    all_data = []
    all_labels = []
    subject_file = './rawData/Benchmark/S' + str(test_id) + '.mat'
    rawdata = loadmat(subject_file)
    rawdata = rawdata['eeg']
    for j in range(4, 6):  # 6个block
        block_index = j
        for k in range(40):  # 40个target
            target_index = k
            for s in range(step):
                channel_data = rawdata[:, 160: (160 + int(fs * duration)), target_index, block_index]
                all_labels.append(target_index)
                all_data.append(channel_data)

    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    return all_data, all_labels


if __name__ == "__main__":
    choose = 3
    all_predict = []
    all_label = []
    # all_data, all_labels = test_single(1)  # (80, 11, 250)
    # print(all_data.shape)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model = DoubleScaleATCNet(
    #     num_channel=11,
    #     num_classes=40,
    #     signal_length=int(fs*duration),
    # )

    # model = NoAttenNet(
    #     num_channel=11,
    #     num_classes=40,
    #     signal_length=int(fs * duration),
    # )

    # model = NoTCNet(
    #     num_channel=11,
    #     num_classes=40,
    #     signal_length=int(fs * duration),
    # )

    # model = GlobalNet(
    #     num_channel=11,
    #     num_classes=40,
    #     signal_length=int(fs * duration),
    # )

    model = DTCNet(
        num_channel=11,
        num_classes=40,
        signal_length=int(fs * duration),
    )

    # model_path = 'model/DATCN_pretrain_'+str(duration)+'s_S' + str(choose) + '.pth'
    model_path = 'model/DTCNet_pretrain_' + str(duration) + 's_S' + str(choose) + '.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))

    for s in range(35):
        subject = s+1
        all_data, all_labels = test_single(subject)
        all_data = torch.Tensor(all_data)
        all_labels = torch.Tensor(all_labels)
        data_length = len(all_data)

        model.eval()
        with torch.no_grad():
            outputs = model(all_data)
        target = outputs.argmax(axis=1)
        all_predict.append(target.detach().numpy())
        all_label.append(all_labels.detach().numpy())
        accuracy = (target == all_labels).sum()
        all_acc = float(accuracy / data_length * 100)
        print("Subject {},".format(subject))
        print("accuracy: {}% \n\t".format(all_acc))

    all_predict = np.array(all_predict).flatten()
    all_label = np.array(all_label).flatten()
    print(classification_report(all_label, all_predict))
    macro_f1 = f1_score(all_label, all_predict, average='macro')
    print("F1 SCORE:")
    print(macro_f1 * 100)
    all_acc = accuracy_score(all_label, all_predict)
    print("ACC: ")
    print(all_acc * 100)  #
    # output = pd.DataFrame({'Predict Answer': all_predict,
    #                        'Ground Truth': all_label})
    # output.to_csv('pretrain.csv', index=False)