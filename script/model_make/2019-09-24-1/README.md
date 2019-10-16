Qure.aiデータを用いた出血判別モデル
==============================

## ディレクトリ構成

------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │   │
    │   ├── weights             <- weights of trained models
    │   │   
    │   │
    │   └── prediction_results  <- model prediction results
    │                        
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │                     predictions
    │   │
    │   └── visualization  <- Scripts to visualize prediction results
    │
    └── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
                              generated with `pip freeze > requirements.txt`

--------

## 各ファイルの説明

プロジェクト内の各ファイルについて説明いたします。

### data/processed/csvs_for_scandataset_example/

src/models/datasets.pyで定義されているScanDatasetを用いてデータセットオブジェクトを作成する際に、
引数として指定するcsvファイル(例)が含まれているディレクトリです。

### data/processed/csvs_for_slicedataset_example/

src/models/datasets.pyで定義されているSliceDatasetを用いてデータセットオブジェクトを作成する際に、
引数として指定するcsvファイル(例)が含まれているディレクトリです。

### src/models/datasets.py

データセットオブジェクトが含まれているスクリプトです。

### src/models/models.py

pytorchで実装したモデルが含まれているスクリプトです。

全てのモデルクラスはこのスクリプト内で定義されております。

### src/models/train_model.py

モデルの学習を行うスクリプトです。
3Dモデルと2Dモデルの両方に対応しております。

### src/models/predict_model.py

train_model.pyで学習したモデルの、入力データに対する予測結果を算出して保存するスクリプトです。
3Dモデルと2Dモデルの両方に対応しております。

### src/models/create_cam_heatmap_od_slicemodel.py

train_model.pyで学習したモデルの、入力データに対するCAM(Class Activation Mapping)を算出して保存するスクリプトです。
2Dモデルにのみ対応しております。

### src/models/evaluate.py

predict_model.pyで得られた予測結果を評価するためのスクリプトです。


### src/models/utils.py

汎用関数を記述したスクリプトです。

### src/models/pytorch_utils.py

pytorchを用いたモデルの学習や評価などのための関数を記述したスクリプトです。


### src/visualization/visualize_each_instance_with_cam_heatmap_of_slicemodel.py

create_cam_heatmap_od_slicemodel.pyで得られたCAMの算出結果を、元のスライス画像と重ね合わせた画像ファイルとして保存するスクリプトです。