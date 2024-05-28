import numpy as np
import torch
import torch.nn as nn
from algorithm.DATCNet import DoubleScaleATCNet
from algorithm.Ablation.DTCN import DTCNet
from algorithm.Ablation.non_TCN import NoTCNet
# from algorithm.Ablation.non_atten import NoAttenNet
from algorithm.Ablation.Singlescale_local import LocalNet
from algorithm.Ablation.Singlescale_global import GlobalNet
import random
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
# from sklearn.model_selection import StratifiedKFold
from pytorchtools import EarlyStopping
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

# torch.cuda.empty_cache()
seed = 1234
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

duration = 1.2  # 用于训练的信号长度。1表示1秒
s_len = 25  # 滑窗步数
step = 1  # 1: 不滑窗
fs = 250


def train_data(test_subject):
    all_data = []
    all_labels = []
    all_subject = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                   11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                   21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                   31, 32, 33, 34, 35]
    train_subject = []  # 除验证集以外的受试
    for s in all_subject:
        if s != test_subject:
            train_subject.append(s)
    # print(train_subject)
    for ts in range(len(train_subject)):
        subject_file = './rawData/Benchmark/S' + str(train_subject[ts]) + '.mat'
        rawdata = loadmat(subject_file)
        rawdata = rawdata['eeg']
        for j in range(4):  # 前5块个block
            block_index = j
            for k in range(40):  # 40个target
                target_index = k
                for s in range(step):
                    channel_data = rawdata[:, int(160 + s_len * s):int(160 + s_len * s + duration * fs), target_index,
                                   block_index]
                    all_labels.append(target_index)
                    all_data.append(channel_data)

    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    return all_data, all_labels


def valid_data(test_subject):
    all_data = []
    all_labels = []
    valid_file = './rawData/Benchmark/S' + str(test_subject) + '.mat'
    valid_data = loadmat(valid_file)
    valid_data = valid_data['eeg']
    for j in range(4):  # 6个block
        block_index = j
        for k in range(40):  # 40个target
            target_index = k
            for s in range(step):
                channel_data = valid_data[:, int(160 + s_len * s):int(160 + s_len * s + duration * fs), target_index,
                               block_index]
                all_labels.append(target_index)
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
        s_data = self.data[item]  # array(1,1,10,45,6)
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


# # 用于权重调优
# class weightConstraint(object):
#     def __init__(self):
#         pass
#
#     def __call__(self, module):
#         if hasattr(module, 'weight'):
#             print("Entered")
#             w = module.weight.data
#             w = w.clamp(0, 1.0)  # 将参数范围限制到0.5-0.7之间
#             module.weight.data = w
#
#
# def init_weights(m):
#     if type(m) == nn.Conv2d:
#         torch.nn.init.normal_(m.weight,mean=0,std=0.5)
#     if type(m) == nn.Linear:
#         nn.init.uniform_(m.weight, a=-0.1, b=0.1)
#         m.bias.data.fill_(0.01)


if __name__ == "__main__":
    # test()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr, num_epochs, batch_size = 0.001, 100, 64
    patience = 10  # 早停为7步
    delta = 1e-5
    n_splits = 5
    foldperf = {}
    # data_list, all_label = train_data(1)
    # print(data_list.shape)  # (5440, 11, 250)
    # data_v_list, all_v_label = valid_data(1)
    # print(data_v_list.shape)  # (160, 11, 250)

    for subject_id in range(2, 3):  # 8-subject-fold cross validation, 5-fold CV training
        subject = subject_id + 1
        print('————————————————————————————————————————\n')
        print('Subject-Fold for validation: S{} \n'.format(subject))
        train_dataset = TrainDataset(subject, ToTensor())
        test_dataset = ValidDataset(subject, ToTensor())
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        train_size = len(train_dataset)
        test_size = len(test_dataset)
        # kf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)
        # constraints = weightConstraint()  # 权重约束

        model = DTCNet(
            num_channel=11,
            num_classes=40,
            signal_length=int(fs*duration),
        )

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

        # model.apply(init_weights)
        # model = torch.nn.DataParallel(model,device_ids=list(range(torch.cuda.device_count())))
        model = model.to(device)

        loss_fn = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            loss_fn = loss_fn.cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        clr = CosineAnnealingLR(optimizer, T_max=10)

        # one_fold_path = 'model/DATCN_pretrain_' + str(duration) + 's_S' + str(subject) + '.pth'
        one_fold_path = 'model/DTCNet_pretrain_' + str(duration) + 's_S' + str(subject) + '.pth'
        early_stopping = EarlyStopping(patience=patience, verbose=True, path=one_fold_path)

        total_train_step = 0
        total_test_step = 0
        history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0
            total_train_accuracy = 0
            avg_train_acc = 0
            for data in train_dataloader:
                s_data, s_label = data
                info = torch.Tensor(s_data)
                info = info.to(device)  # 将张量存入GPU
                s_label = s_label.to(torch.long)
                s_label = s_label.to(device)
                outputs = model(info)
                loss = loss_fn(outputs, s_label)
                total_train_loss = total_train_loss + loss.cpu().detach().numpy().item()
                accuracy = (outputs.argmax(axis=1) == s_label).sum()
                total_train_accuracy = total_train_accuracy + accuracy

                # 优化器优化模型
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # model._modules['l3'].apply(constraints)

                total_train_step = total_train_step + 1
            avg_train_acc = float(total_train_accuracy / train_size * 100)
            clr.step()
            #
            model.eval()
            total_test_loss = 0
            total_accuracy = 0
            with torch.no_grad():
                for data in test_dataloader:
                    s_data, s_label = data
                    info = torch.Tensor(s_data)
                    info = info.to(device)
                    s_label = s_label.to(torch.long)
                    s_label = s_label.to(device)
                    outputs = model(info)
                    loss = loss_fn(outputs, s_label)
                    total_test_loss = total_test_loss + loss.cpu().detach().numpy().item()
                    accuracy = (outputs.argmax(axis=1) == s_label).sum()  # 很怪，检查一下
                    total_accuracy = total_accuracy + accuracy

            # if (epoch + 1) % 25 == 0:
            #     print("-------第 {} 轮训练开始-------".format(epoch + 1))
            #     # print("output.shape is {}".format(outputs.shape))
            #     print("整体测试集上的Loss: {}".format(total_test_loss))
            #     print("整体测试集上的正确率: {}%".format(total_accuracy / test_size * 100))
            #     # print('当前学习率：%f' % optimizer.state_dict()['param_groups'][0]['lr'])
            # total_test_step = total_test_step + 1

            if (epoch + 1) == num_epochs:
                history['train_loss'].append(total_train_loss)
                history['train_acc'].append(avg_train_acc)
                history['test_loss'].append(total_test_loss)
                history['test_acc'].append(float(total_accuracy / test_size * 100))

            early_stopping(total_test_loss, model)
            if early_stopping.early_stop:
                history['train_loss'].append(total_train_loss)
                history['train_acc'].append(avg_train_acc)
                history['test_loss'].append(total_test_loss)
                history['test_acc'].append(float(total_accuracy / test_size * 100))
                print("Early stopping")
                break

        foldperf['fold_S{}'.format(subject)] = history
        print(history)
        # torch.cuda.empty_cache()
