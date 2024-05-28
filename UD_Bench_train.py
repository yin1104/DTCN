import numpy as np
import torch
import torch.nn as nn
from algorithm.DATCNet import DoubleScaleATCNet
from algorithm.Ablation.DTCN import DTCNet
from algorithm.Ablation.non_TCN import NoTCNet
from algorithm.Ablation.non_atten import NoAttenNet
from algorithm.Ablation.Singlescale_local import LocalNet
from algorithm.Ablation.Singlescale_global import GlobalNet
import random
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
# from sklearn.model_selection import StratifiedKFold
from pytorchtools import EarlyStopping


seed = 1234
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

duration = 1.2
w_len = 25
step_num = 1
fs = 250


def train_data(subject):
    all_data = []
    all_labels = []
    train_block = [0, 1, 2, 3]
    subject_file = './rawData/Benchmark/S' + str(subject) + '.mat'
    rawdata = loadmat(subject_file)
    rawdata = rawdata['eeg']
    for k in range(40):
        target_idx = k
        for ts in range(len(train_block)):  # 共4块
            block_idx = train_block[ts]
            for step in range(step_num):
                channel_data = rawdata[:, int(160 + w_len * step):int(160 + w_len * step + duration * fs), target_idx,
                               block_idx]
                all_labels.append(target_idx)
                all_data.append(channel_data)

    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    return all_data, all_labels


def valid_data(subject):
    all_data = []
    all_labels = []
    subject_file = './rawData/Benchmark/S' + str(subject) + '.mat'
    rawdata = loadmat(subject_file)
    rawdata = rawdata['eeg']
    valid_b = [4, 5]
    for k in range(40):
        target_idx = k
        for ts in range(len(valid_b)):  # 共4块
            block_idx = valid_b[ts]
            for step in range(step_num):
                channel_data = rawdata[:, int(160 + w_len * step):int(160 + w_len * step + duration * fs), target_idx,
                               block_idx]
                all_labels.append(target_idx)
                all_data.append(channel_data)

    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    return all_data, all_labels


class TrainDataset(Dataset):

    def __init__(self, test_subject, transformer=None):
        super(TrainDataset, self).__init__()
        self.data = []
        self.label = []
        data_list, all_label = train_data(test_subject)
        self.data = data_list
        self.label = all_label
        self.transform = transformer

    def __len__(self):
        assert len(self.data) == len(self.label)
        return len(self.data)

    def __getitem__(self, item):
        s_data = self.data[item]
        s_label = self.label[item]
        if self.transform is not None:
            s_data = self.transform(s_data)
        return s_data, s_label


class ValidDataset(Dataset):

    def __init__(self, test_subject, transformer=None):
        super(ValidDataset, self).__init__()
        self.data = []
        self.label = []
        data_list, all_label = valid_data(test_subject)
        self.data = data_list
        self.label = all_label
        self.transform = transformer

    def __len__(self):
        assert len(self.data) == len(self.label)
        return len(self.data)

    def __getitem__(self, item):
        s_data = self.data[item]  # array(1,1,10,45,6)
        s_label = self.label[item]
        if self.transform is not None:
            s_data = self.transform(s_data)
        return s_data, s_label


class ToTensor(object):
    def __call__(self, seq):
        return torch.tensor(seq, dtype=torch.float)


if __name__ == "__main__":
    choose = 3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(new_model)
    lr, num_epochs, batch_size = 0.0005, 150, 64
    patience = 10  # 早停为7步
    delta = 1e-5
    n_splits = 5
    foldperf = {}
    # all_data, all_labels = valid_data(1)
    # print(all_data.shape)
    for subject_id in range(35):  # 8-subject-fold cross validation, 5-fold CV training
        subject = subject_id + 1
        print('————————————————————————————————————————\n')
        print('Subject-Fold for validation: S{} \n'.format(subject))
        for valid_block in range(1):
            train_dataset = TrainDataset(subject, ToTensor())
            test_dataset = ValidDataset(subject, ToTensor())
            train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
            test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
            train_size = len(train_dataset)
            test_size = len(test_dataset)

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

            # model = GlobalNet(
            #     num_channel=11,
            #     num_classes=40,
            #     signal_length=int(fs * duration),
            # )

            # model_path = 'model/DATCN_pretrain_' + str(duration) + 's_S' + str(choose) + '.pth'
            model_path = 'model/DTCNet_pretrain_' + str(duration) + 's_S' + str(choose) + '.pth'
            model.load_state_dict(torch.load(model_path))
            new_model = model.cuda()

            loss_fn = nn.CrossEntropyLoss()
            if torch.cuda.is_available():
                loss_fn = loss_fn.cuda()

            optimizer = torch.optim.Adam(new_model.parameters(), lr=lr, weight_decay=0.05)
            clr = CosineAnnealingLR(optimizer, T_max=10)

            one_fold_path = 'model/Benchmark/DTCNet_' + str(duration) + 's_S' + str(subject) + '.pth'
            # one_fold_path = 'model/Benchmark/ablation/GlobalNet_b4_' + str(duration) + 's_S' + str(subject) + '.pth'
            # early_stopping = EarlyStopping(patience=patience, verbose=True, path=one_fold_path)

            total_train_step = 0
            total_test_step = 0
            history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
            for epoch in range(num_epochs):
                new_model.train()
                total_train_loss = 0
                total_train_accuracy = 0
                avg_train_acc = 0
                for data in train_dataloader:
                    s_data, s_label = data
                    info = torch.Tensor(s_data)
                    info = info.to(device)  # 将张量存入GPU
                    s_label = s_label.to(torch.long)
                    s_label = s_label.to(device)
                    outputs = new_model(info)
                    loss = loss_fn(outputs, s_label)
                    total_train_loss = total_train_loss + loss.sum().item()
                    accuracy = (outputs.argmax(axis=1) == s_label).sum()
                    total_train_accuracy = total_train_accuracy + accuracy

                    # 优化器优化模型
                    optimizer.zero_grad()
                    loss.sum().backward()
                    optimizer.step()

                    total_train_step = total_train_step + 1
                avg_train_acc = float(total_train_accuracy / train_size * 100)
                clr.step()

                new_model.eval()
                total_test_loss = 0
                total_accuracy = 0
                with torch.no_grad():
                    for data in test_dataloader:
                        s_data, s_label = data
                        info = torch.Tensor(s_data)
                        info = info.to(device)
                        s_label = s_label.to(torch.long)
                        s_label = s_label.to(device)
                        outputs = new_model(info)
                        loss = loss_fn(outputs, s_label)
                        total_test_loss = total_test_loss + loss.sum().item()
                        accuracy = (outputs.argmax(axis=1) == s_label).sum()  # 很怪，检查一下
                        total_accuracy = total_accuracy + accuracy

                if (epoch + 1) % 25 == 0:
                    print("-------第 {} 轮训练开始-------".format(epoch + 1))
                    # print("output.shape is {}".format(outputs.shape))
                    print("整体测试集上的Loss: {}".format(total_test_loss))
                    print("整体测试集上的正确率: {}%".format(total_accuracy / test_size * 100))
                    # print('当前学习率：%f' % optimizer.state_dict()['param_groups'][0]['lr'])
                total_test_step = total_test_step + 1

                if (epoch + 1) == num_epochs:
                    history['train_loss'].append(total_train_loss)
                    history['train_acc'].append(avg_train_acc)
                    history['test_loss'].append(total_test_loss)
                    history['test_acc'].append(float(total_accuracy / test_size * 100))
                    torch.save(model.state_dict(), one_fold_path)  # 保存模型参数
                    print("模型已保存 Subject:", subject)

                # early_stopping(total_test_loss, new_model)
                # if early_stopping.early_stop:
                #     history['train_loss'].append(total_train_loss)
                #     history['train_acc'].append(avg_train_acc)
                #     history['test_loss'].append(total_test_loss)
                #     history['test_acc'].append(float(total_accuracy / test_size * 100))
                #     print("Early stopping")
                #     break

            foldperf['fold_S{}'.format(subject)] = history
            print(history)