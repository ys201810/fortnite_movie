
import os
import glob
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    roc_auc_score, f1_score

# モデルの予測結果と教師値を用いて、予測結果を評価するスクリプト
# AUCやaccuracyなどを計算する
# TPが1件もないと、F1 scoreを算出する際に警告が出るが、
# その場合はF1 scoreの値に0.0が入るだけであり、特に例外処理する必要はないと思われる


def visualize(df):
    """
    epochごとのaucを可視化する
    """
    plt.rcParams['font.size'] = 12; plt.rcParams['figure.figsize'] = 15, 10
    plt.figure(figsize=(10, 4))
    for hemorrhage, v in df.groupby("hemorrhage"):
        plt.plot(v['auc'].values, 'o-', label=hemorrhage)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        plt.grid(True)
        plt.ylim(0, 1)
        plt.xticks(np.arange(len(v)), sorted(df['model'].unique()), rotation=30)
        plt.ylabel("auc")
        plt.title("AUC score for each epoch")

    plt.tight_layout()
    return plt
    


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--predict_dir", type=str,
                        default="../../models/predict_results/init",
                        help="モデルの予測結果が保存されているディレクトリ")
    parser.add_argument("--save_dir", type=str,
                        default="../../reports/csvs/models/evaluate_results/init",
                        help="評価結果(csv)を保存するディレクトリ")
    parser.add_argument("--save_fig_dir", type=str, default="",
                        help="評価結果(aucの図)を保存するディレクトリ")
    parser.add_argument('--classif_hemorrhage_or_not', action='store_true',
                        help='出血しているかどうかだけの判別とするか')
    parser.add_argument("--threshold", type=float,
                        default=0.0, help="モデルの予測結果(sigmoid変換前)を1/0に変換する際のthreshold")
    args = parser.parse_args()
    print(args)
    if args.classif_hemorrhage_or_not:
        HEMORRHAGES = ['HEMORRHAGE']
    else:
        HEMORRHAGES = ['ICH', 'IPH', 'IVH', 'SDH', 'EDH', 'SAH']

    # 予測結果・教師値のロード
    pred_paths = sorted(glob.glob(args.predict_dir + "/*_pred.npy"))
    true_paths = sorted(glob.glob(args.predict_dir + "/*_true.npy"))
    # 各モデル・各データに対する指標の算出
    lst = []
    for pred_path, true_path in zip(pred_paths, true_paths):
        print(pred_path)
        model_name = os.path.basename(pred_path)[:os.path.basename(pred_path).find("_pred")]
        pred = np.load(pred_path, allow_pickle=True)
        true = np.load(true_path, allow_pickle=True).astype(int)
        for h, p, t in zip(HEMORRHAGES, pred.T, true.T):
            p_aslabel = (p > args.threshold).astype(int)
            auc = roc_auc_score(t, p)
            acc = accuracy_score(t, p_aslabel)
            precision = precision_score(t, p_aslabel)
            recall = recall_score(t, p_aslabel)
            f1 = f1_score(t, p_aslabel)
            lst.append([model_name, h, auc, acc, precision, recall, f1])

    df = pd.DataFrame(lst).rename(columns={0: "model", 1: "hemorrhage", 2: "auc", 3: "acc",
                                           4: "precison", 5: "recall", 6: "f1"})
    os.makedirs(args.save_dir, exist_ok=True)
    df.to_csv(os.path.join(args.save_dir, "out.csv"), index=False)

    # epochごとのaucの値を図示して保存する
    plt = visualize(df)
    os.makedirs(args.save_fig_dir, exist_ok=True)
    plt.savefig(os.path.join(args.save_fig_dir, "auc.png"))
