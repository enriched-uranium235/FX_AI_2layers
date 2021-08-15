from simple_model import build_simple_model
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data as DataReader

start_day = "2000-1-1"

#start_dayから今日までのデータをダウンロードし、学習データとする。
df=DataReader.get_data_yahoo("JPY=X",start=start_day)

#ドル円データをcsvにして保存しておく
df.to_csv("USDJPY.csv")

#モデルの作成
forex_model = build_simple_model()

#深層学習モデルの形状を表示
#print(model.summary())

#modelをコンパイル（最小化しようとする損失関数--ここでは自乗誤差--などを指定）する
forex_model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['mae'])

df["Close"] = df["Close"].astype(float)

#後のコードが書きやすいように訓練データを加工
#よくわからない時はdfをprintで見てみると良い

#同じ行に過去16時点の価格(予想に使う説明変数)と今の価格（目的変数）が並ぶようにする
for i in range(16):
    df["Close-"+str(i+1)]=df["Close"].shift(i+1)

df = df.dropna(how="any")

# print(df.head())
# print(df.tail())
# print(df.columns)

#dfをnumpy配列に変換
#訓練データ(train_x)は　過去16時点までの価格*行数　となっている
train_x=df[["Close-"+str(i+1) for i in range(16)]].values
train_y=df["Close"].values

#注意）今回はコードの簡単さを重視して正規化などは無し
#リターンを取ってみる、といった工夫もなし

# 学習の実行
history = forex_model.fit(train_x, train_y, epochs=400, validation_split=0.2)

#学習結果をファイルに保存
#成功するとディレクトリ内にparam.hdf5というファイルが生成される
forex_model.save_weights('param.hdf5')

#学習中の評価値の推移
#右肩下がりのグラフになっていると嬉しい
plt.plot(history.history['mae'], label='train mae')
plt.plot(history.history['val_mae'], label='val mae')
plt.xlabel('epoch')
plt.ylabel('mae')
plt.legend(loc='best')
plt.ylim([0,5])
plt.savefig("train.png")