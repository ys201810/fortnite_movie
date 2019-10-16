
import time
from collections import defaultdict
from abc import abstractmethod
from tqdm import tqdm
import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
from tqdm import tqdm


def worker_init_fn(worker_id):
    """
    自作Dataset内で乱数を使うときには、
    torch.utils.data.DataLoaderのworker_init_fnにこの関数を指定する。
    ds = DataLoader(ds, 10, shuffle=False, num_workers=4, worker_init_fn=worker_init_fn)
    のように。
    そうしないと、num_workersに1以上を指定した時にbatch内で同じ乱数が生成される。
    参考 : https://github.com/pytorch/pytorch/issues/5059
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def rand_init(seed=0):
    """
    pytorchでGPUを使っている時でも再現性を得るための関数
    計算を実行する前にまず実行しておく
    参考 : https://pytorch.org/docs/stable/notes/randomness.html
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

                            
def create_dataset(data_list, dtype_list, device):
    """
    torch.tensorより、pytorchのdatasetオブジェクトを作成する
    """
    assert len(data_list) == len(dtype_list)
    tensors = []
    for data, dtype in zip(data_list, dtype_list):
        tensor = torch.Tensor(data).to(device, dtype=dtype)
        tensors.append(tensor)

    dataset = torch.utils.data.TensorDataset(*tensors)
    return dataset



class BasicTrainerBase:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.train_dataloader = None
        self.valid_dataloader = None
        self.train_costs_lst = None
        self.valid_costs_lst = None

    @abstractmethod
    def loss_forward(self, data):
        raise NotImplementedError("損失関数の値を計算する関数が定義されておりません")

    def after_each_epoch(self):
        """
        エポック終了ごとに実行させたい処理を記述する。
        例えば学習率スケジューラによる学習率の更新など
        """
        pass

    def training(self, train_dataloader, valid_dataloader,
                 n_epochs, y_upper_bound=None, 
                 callbacks_dct_using_model=None, clip=None):
        """

        * train_dataloader
        * valid_dataloader
        はいずれも、dataloaderを与える

        callbacks_dct_using_modelは、
        * key : 正の整数
        * value : model, epochを引数に取る関数のリスト
        であるような辞書。
        エポック数がkeyで割り切れる時に、その関数が呼ばれる。
        例えば、5エポックに1回モデルの重みをカレントに保存したい場合には、
        def create_save_weights_func(dir_):
            def save_weights(model, epoch):
                torch.save(model.state_dict(), os.path.join(
                    dir_, "epoch{0:04d}.pth".format(epoch + 1)))
            return save_weights

        callbacks_dct_using_model = {5 : [create_save_weights_func("./")]}
        とすればよい。
        """
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        mb = range(n_epochs)
        def pb(): return tqdm(train_dataloader, total=len(train_dataloader))

        history = defaultdict(list)

        # 学習曲線描画のための前準備
        self.train_costs_lst = []
        self.valid_costs_lst = []
        x_bounds = [1, n_epochs]
        y_bounds = None

        for epoch in mb:
            # Train
            start = time.time()
            self.model.train()
            train_metrics = []

            for data in pb():
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                loss = self.loss_forward(data)
                loss.backward()
                if clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), clip)
                self.optimizer.step()

                train_metrics.append([loss.item()])

            # Valid
            self.model.eval()
            valid_metrics = []
            with torch.no_grad():
                for data in valid_dataloader:
                    # forward
                    loss = self.loss_forward(data)
                    valid_metrics.append([loss.item()])

            # 損失関数の値の計算
            train_metrics_mean = np.mean(train_metrics, axis=0)
            valid_metrics_mean = np.mean(valid_metrics, axis=0)
            self.train_costs_lst.append(train_metrics_mean[0])
            self.valid_costs_lst.append(valid_metrics_mean[0])

            # learning curveの図示
            if y_bounds is None:
                # 1エポック目
                y_bounds = [0, max(self.train_costs_lst + self.valid_costs_lst)
                            * 1.1 if y_upper_bound is None else y_upper_bound]

            t = np.arange(len(self.train_costs_lst)) + \
                1  # 1エポック目の値が横軸の1の部分に表示されるように+1
            graphs = [[t, self.train_costs_lst], [t, self.valid_costs_lst]]

            # 学習過程の出力・historyへの追加
            history['epoch'].append(epoch + 1)
            message_output = 'EPOCH: {0:02d},'.format(epoch+1)
            for name, val in zip(['loss'], train_metrics_mean):
                message_output += " Training {0}: {1:10.5f}".format(name, val)
                history[name].append(val)

            message_output += "  "  # trainとvalidとの間に空白があったほうが見やすい
            for name, val in zip(['loss'], valid_metrics_mean):
                message_output += " Validation {0}: {1:10.5f}".format(
                    name, val)
                history['val_' + name].append(val)

            elapsed_time = time.time() - start
            message_output += '  Elapsed time:{0:10.4f}'.format(
                elapsed_time)
            history['elapsed_time'].append(elapsed_time)
            print(message_output)

            if callbacks_dct_using_model is not None:
                for key, callbacks in callbacks_dct_using_model.items():
                    if (epoch + 1) % key == 0:
                        for callback in callbacks:
                            callback(self.model, epoch)

            self.after_each_epoch()

        return history


def model_outputs_with_ground_truth(model, dataloader, load_func=None):
    """
    modelのdataloaderに対する予測結果と教師値の2つを出力する
    """
    model.eval()
    outputs_all = []
    ground_truth_all = []

    # device取得
    device = next(model.parameters()).device

    with torch.no_grad():
        for _, data in enumerate(dataloader, 0):
            # get the x
            if load_func is not None:
                x, _y = load_func(data)
            else:
                x, _y = data

            # forward
            _outputs = model(x)
            outputs_all.append(_outputs.cpu())
            ground_truth_all.append(_y.cpu())

    y_true = np.concatenate(ground_truth_all, axis=0)
    y_pred = np.concatenate(outputs_all, axis=0)
    return y_pred, y_true


def mixup_data(x, y, device, alpha=0.4):
    '''
    Compute the mixup data. Return mixed inputs, pairs of targets, and lambda
    https://github.com/hongyi-zhang/mixup/blob/master/cifar/utils.py
    を参考に実装
    '''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y
