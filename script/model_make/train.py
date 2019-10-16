# coding=utf-8
import os
import sys
import torch
import numpy as np
from script.model_make.datasets import FiveSecDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from script.model_make.models import resnet18_2D
import torch.optim as optim
import torch.nn as nn
root_dir = os.path.join(os.getcwd(), '../../')
sys.path.append(root_dir)


def main():
    # config setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = os.path.join('/usr', 'local', 'wk', 'data', 'fortnite')
    annotation_file = os.path.join(data_dir, 'annotation.list')
    batch_size = 4
    num_classes = 2
    n_images = 5
    ngf = 64
    lr = 0.001
    weight_decay = 0.0
    epoch = 200
    save_epoch_interval = 5
    save_model_file = './saved/trained_weight.pth'

    # datasetの作成
    train = FiveSecDataset(annotation_file, 'train', None)
    valid = FiveSecDataset(annotation_file, 'val', None)

    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=1)
    val_dataloader = DataLoader(valid, batch_size=batch_size, shuffle=True, num_workers=1)

    # modelの作成
    model = resnet18_2D(num_classes, n_images * 3, ngf).to(device)

    # optimizer 定義
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss().to(device)

    # training
    train_costs_list = []
    valid_costs_list = []
    for i in range(epoch + 1):
        model.train()
        train_metrics = []
        for j, train_datas in enumerate(train_dataloader):
            train_data, train_label = train_datas
            train_data = train_data.to(device)
            train_label = train_label.to(device)
            train_data = Variable(train_data)
            train_label = Variable(train_label)
            optimizer.zero_grad()

            train_outputs = model(train_data)
            _, pred = train_outputs.topk(k=1, dim=1, largest=True)
            pred = pred.t()
            train_loss = criterion(train_outputs, train_label)
            train_loss.backward()
            optimizer.step()
            train_metrics.append([train_loss.item()])
            # print('[train]epoch:{}-iteration:{} loss:{}'.format(i, j, train_loss.item()))

        # validation
        model.eval()
        valid_metrics = []
        with torch.no_grad():
            for valid_data, valid_label in val_dataloader:
                valid_data = valid_data.to(device)
                valid_label = valid_label.to(device)
                criterion = nn.BCEWithLogitsLoss()
                valid_outputs = model(valid_data)
                valid_loss = criterion(valid_outputs, valid_label)
                valid_metrics.append([valid_loss.item()])

        train_metrics_mean = np.mean(train_metrics, axis=0)
        valid_metrics_mean = np.mean(valid_metrics, axis=0)
        train_costs_list.append(train_metrics_mean[0])
        valid_costs_list.append(valid_metrics_mean[0])
        print('[train]epoch:{}-end loss:{}'.format(i, train_metrics_mean[0]))
        print('[valid]epoch:{}-end loss:{}'.format(i, valid_metrics_mean[0]))

        if i % save_epoch_interval == 0:
            torch.save(model.state_dict(), save_model_file.replace('.pth', '+' + str(i) + '.pth'))


if __name__ == '__main__':
    main()
