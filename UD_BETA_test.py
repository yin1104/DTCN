import numpy as np
import torch

from algorithm.Ablation.DTCN import DTCNet
from algorithm.DATCNet import DoubleScaleATCNet
from algorithm.Ablation.non_atten import NoAttenNet
import random
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score,accuracy_score

seed = 1234
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

duration = 1
step_num = 1
all_block = [3]
fs = 250

def test_single(test_id):
    all_data = []
    all_labels = []
    subject_file = './rawData/BETA/S' + str(test_id) + '.mat'
    rawdata = loadmat(subject_file)
    rawdata = rawdata['eeg']
    for k in range(40):  # 40个target
        target_index = k
        for block_index in all_block:
            channel_data = rawdata[:, int(160):int(160+fs*duration), target_index, block_index]
            all_labels.append(target_index)
            all_data.append(channel_data)

    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    return all_data, all_labels


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # all_data, all_labels = test_single(1)
    # print(all_data.shape)
    all_predict = []
    all_label = []
    for s in range(15):
        subject = s + 1

        # model = DoubleScaleATCNet(
        #     num_channel=11,
        #     num_classes=40,
        #     signal_length=int(fs * duration),
        # )

        # model = NoAttenNet(
        #     num_channel=11,
        #     num_classes=40,
        #     signal_length=int(fs * duration),
        # )

        model = DTCNet(
            num_channel=11,
            num_classes=40,
            signal_length=int(fs * duration),
        )

        # model_path = 'model/BETA/DATCN_b2_' + str(duration) + 's_S' + str(subject) + '.pth'
        model_path = 'model/BETA/DTCN_b1_' + str(duration) + 's_S' + str(subject) + '.pth'
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

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
        accuracy = sum(np.array(target == all_labels))
        all_acc = float(accuracy / data_length * 100)
        print("Subject {},".format(subject))
        print("accuracy: {}% \n\t".format(all_acc))

    all_predict = np.array(all_predict).flatten()
    all_label = np.array(all_label).flatten()
    print(all_predict.shape)
    print(all_label.shape)
    # label = np.zeros(40)
    # for i in range(40):
    #     label[i] = i
    # y1_label = label_binarize(all_label, classes=label)  # Ground Truth
    # y1_pred = label_binarize(all_predict, classes=label)
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
    # output.to_csv('result/2ATCN.csv', index=False)