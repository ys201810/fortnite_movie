# coding=utf-8
import os
import sys
import torch
import time
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from script.model_make.datasets import FiveSecDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from script.model_make.models import resnet18_2D
import torch.optim as optim
import torch.nn as nn
from statistics import mean
root_dir = os.path.join(os.getcwd(), '../../')
sys.path.append(root_dir)


def draw_history(acc_log_file, loss_log_file, model_save_dir, model_type):
    # make training history on image file.
    acc_df = pd.read_csv(acc_log_file)
    val_x = acc_df['epoch'].values
    val_y_train_acc = acc_df['train_acc'].values
    val_y_val_acc = acc_df['val_acc'].values

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(val_x, val_y_train_acc, linestyle='-', color='blue', label='train')
    ax.plot(val_x, val_y_val_acc, linestyle='-', color='orange', label='val')
    ax.legend(loc='best')
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    plt.title("accuracy by each epoch")
    plt.savefig(os.path.join(model_save_dir, model_type + '_acc_history.png'))

    acc_df = pd.read_csv(loss_log_file)
    val_x = acc_df['epoch'].values
    val_y_train_loss = acc_df['train_loss'].values
    val_y_val_loss = acc_df['val_loss'].values

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(val_x, val_y_train_loss, linestyle='-', color='blue', label='train')
    ax.plot(val_x, val_y_val_loss, linestyle='-', color='orange', label='val')  # linestyle = '--'
    ax.legend(loc='best')
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    plt.title("loss by each epoch")
    plt.savefig(os.path.join(model_save_dir, model_type + '_loss_history.png'))


def main():
    # config setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = os.path.join('/usr', 'local', 'wk', 'data', 'fortnite')
    annotation_file = os.path.join(data_dir, 'annotation.list')
    batch_size = 64
    num_classes = 2
    n_images = 5
    ngf = 64
    lr = 0.001
    weight_decay = 0.0
    epoch = 200
    save_epoch_interval = 5
    save_model_file = './saved/trained_weight.pth'
    acc_log_file, loss_log_file = './saved/acc.log', './saved/loss.log'
    weights = torch.FloatTensor([0.5, 1]).to(device)  # [not in battle, in battle]

    # datasetの作成
    train = FiveSecDataset(annotation_file, 'train', None)
    valid = FiveSecDataset(annotation_file, 'val', None)

    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    val_dataloader = DataLoader(valid, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

    # modelの作成
    model = resnet18_2D(num_classes, n_images * 3, ngf).to(device)

    # optimizer 定義
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(weight=weights).to(device)

    # training
    train_loss_list, train_acc_list = [], []
    valid_loss_list, valid_acc_list = [], []
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
            train_correct = pred.eq(train_label.argmax(dim=1))
            n_train_correct = train_correct.float().sum().item()
            train_acc = n_train_correct / batch_size
            train_acc_list.append(train_acc)
            train_loss = criterion(train_outputs, train_label)
            train_loss.backward()
            optimizer.step()
            train_metrics.append([train_loss.item()])
            if j != 0:
                elapsed_time = time.time() - iteration_start_time
                print('[train]epoch:{}-iteration:{} loss:{} acc:{} elapsed_time:{}[s]'.format(
                    i, j, train_loss.item(), train_acc, elapsed_time))
            iteration_start_time = time.time()

        # validation
        model.eval()
        valid_metrics = []
        with torch.no_grad():
            for valid_data, valid_label in val_dataloader:
                valid_data = valid_data.to(device)
                valid_label = valid_label.to(device)
                criterion = nn.BCEWithLogitsLoss()
                valid_outputs = model(valid_data)

                _, pred = valid_outputs.topk(k=1, dim=1, largest=True)
                pred = pred.t()
                valid_correct = pred.eq(valid_label.argmax(dim=1))
                n_valid_correct = valid_correct.float().sum().item()
                valid_acc = n_valid_correct / batch_size
                valid_acc_list.append(valid_acc)

                valid_loss = criterion(valid_outputs, valid_label)
                valid_metrics.append([valid_loss.item()])

        train_metrics_mean = np.mean(train_metrics, axis=0)
        valid_metrics_mean = np.mean(valid_metrics, axis=0)
        train_loss_list.append(train_metrics_mean[0])
        valid_loss_list.append(valid_metrics_mean[0])
        print('[train]epoch:{}-end loss:{} acc:{}'.format(i, train_metrics_mean[0], mean(train_acc_list)))
        print('[valid]epoch:{}-end loss:{} acc:{}'.format(i, valid_metrics_mean[0], mean(valid_acc_list)))

        with open(acc_log_file, 'a') as acc_logf:
            if i == 0:
                acc_logf.write('epoch,train_acc,val_acc\n')
            text = ','.join([str(i + 1), str(mean(train_acc_list)), str(mean(valid_acc_list))]) + '\n'
            acc_logf.write(text)

        with open(loss_log_file, 'a') as loss_logf:
            if i == 0:
                loss_logf.write('epoch,train_loss,val_loss\n')
            text = ','.join([str(i + 1), str(mean(train_loss_list)), str(mean(valid_loss_list))]) + '\n'
            loss_logf.write(text)

        if i % save_epoch_interval == 0:
            torch.save(model.state_dict(), save_model_file.replace('.pth', '_' + str(i) + '.pth'))


if __name__ == '__main__':
    main()
