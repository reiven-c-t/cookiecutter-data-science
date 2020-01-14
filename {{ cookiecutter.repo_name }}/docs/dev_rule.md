# 開発規則

1つのプロジェクトで複数のモデルを作成することが最近よくあるので、そのような時のpractice.

例で示す。

新たに作成するモデルの名前を`bert2bert`とする。

- `/models/bert2bert/` :モデル訓練済みパラメータを保持
- `/src/models/bert2bert/`: 内部にpredict_model.py, train_model.pyをもつ
- `/src/data/bert2bert/`: 内部にpredict_model.pyやtrain_model.pyで使用するDataManagerをもつ。