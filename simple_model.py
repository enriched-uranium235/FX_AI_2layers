from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.models import Sequential

def build_simple_model():

    simple_model=Sequential()
    simple_model.add(Dense(8,activation="relu",input_shape=(16,)))
    simple_model.add(Dense(8,activation="relu"))
    simple_model.add(Dense(8,activation="relu"))
    # 2層目のNNを追加
    simple_model.add(Dense(16,activation="relu"))
    simple_model.add(Dense(1))

    return simple_model