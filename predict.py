import pandas as pd
from simple_model import build_simple_model
import numpy as np

#モデルの読み込み
model = build_simple_model()

#学習した重みを読み込み
#先にget_data_and_train.pyを実行していないと、param.hdf5が存在しないのでエラーになる
model.load_weights('param.hdf5')

#学習したデータを使って今日の終値予測

#予測に使うデータを準備
df = pd.read_csv("USDJPY.csv")

#入力データ
input_data = np.array([[df["Close"].iloc[-(i+1)] for i in range(16)]])

#推論値
prediction = model.predict(input_data).flatten()
print("AIが予測する次のUSDJPYの終値は以下です。")
print(prediction)