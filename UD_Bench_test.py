import numpy as np
import torch
import pandas as pd
from algorithm.DATCNet import DoubleScaleATCNet
from algorithm.Ablation.DTCN import DTCNet
from algorithm.Ablation.non_TCN import NoTCNet
from algorithm.Ablation.non_atten import NoAttenNet
from algorithm.Ablation.Singlescale_local import LocalNet
from algorithm.Ablation.Singlescale_global import GlobalNet
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score,accuracy_score
import random
from scipy.io import loadmat

seed = 1234
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

duration = 1
step_num = 1
all_block = [4, 5]
fs = 250

def test_single(test_id):
    all_data = []
    all_labels = []
    subject_file = './rawData/Benchmark/S' + str(test_id) + '.mat'
    rawdata = loadmat(subject_file)
    rawdata = rawdata['eeg']
    for k in range(40):
        target_index = k
        for block_index in all_block:
            channel_data = rawdata[:, 160 : (160 + int(fs * duration)), target_index,
                           block_index]
            all_labels.append(target_index)
            all_data.append(channel_data)


    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    return all_data, all_labels


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # all_data, all_labels = test_single(1)
    # print(all_data.shape)
    all_predict = []
    all_label = []
    all_truth = [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                 8.2, 9.2, 10.2, 11.2, 12.2, 13.2, 14.2, 15.2,
                 8.4, 9.4, 10.4, 11.4, 12.4, 13.4, 14.4, 15.4,
                 8.6, 9.6, 10.6, 11.6, 12.6, 13.6, 14.6, 15.6,
                 8.8, 9.8, 10.8, 11.8, 12.8, 13.8, 14.8, 5.8]
    for s in range(35):
        subject = s + 1
        model = DTCNet(
            num_channel=11,
            num_classes=40,
            signal_length=int(fs * duration),
        )

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

        # model = LocalNet(
        #     num_channel=11,
        #     num_classes=40,
        #     signal_length=int(fs * duration),
        # )

        model_path = 'model/Benchmark/DTCNet_b4_' + str(duration) + 's_S' + str(subject) + '.pth'
        # model_path = 'model/Benchmark/ablation/LocalNet_b4_' + str(duration) + 's_S' + str(subject) + '.pth'
        model.load_state_dict(torch.load(model_path, map_location=device))
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
    print(all_predict.shape)
    print(all_label.shape)
    print(classification_report(all_label, all_predict))

    # kappa = cohen_kappa_score(all_label, all_predict)
    macro_f1 = f1_score(all_label, all_predict, average='macro')
    print("F1 SCORE:")
    print(macro_f1 * 100)
    all_acc = accuracy_score(all_label, all_predict)
    print("ACC: ")
    print(all_acc * 100)
    # output = pd.DataFrame({'Predict Answer': all_predict,
    #                        'Ground Truth': all_label})
    # output.to_csv('retrain.csv', index=False)
