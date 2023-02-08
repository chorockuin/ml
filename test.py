import logging
import os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.datasets.mnist as mnist
from tensorflow.keras import Input
from tensorflow.keras.layers import LSTM, Conv2D, Dense, Dropout, Flatten, MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # 간단한 실습을 위해 데이터의 일부만 사용합니다.
    x_train, y_train = x_train[:10000], y_train[:10000]
    x_test, y_test = x_test[:1000], y_test[:1000]

    # 지시사항 1번에 따라 코드를 작성하세요.
    # 이미지 픽셀 값을 0~1 사이의 값으로 normalize 합니다.
    x_train = x_train / 255.0# None
    x_test = x_test / 255.0# None

    # 이미지 shape을 (N_samples, 28, 28) -> (N_samples, 28, 28, 1) 로 변경합니다.
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    # 정수형의 라벨을 categorical한 형태로 바꿔줍니다.
    y_train = to_categorical(y_train)# None
    y_test = to_categorical(y_test)#None

    return x_train, y_train, x_test, y_test


def CNN_Model():
    # 지시사항 2번에 따라 CNN의 구조를 작성하세요.
    # ※ 지시사항과 다른 경우, 오답 처리가 될 수 있습니다.
    model = Sequential()
    model.add(Input((28, 28, 1)))
    model.add(Conv2D(32, (5, 5), padding='same', activation='relu'))#None)
    model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))#None)
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))#None)
    model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))#None)
    model.add(Flatten())#None)
    model.add(Dropout(0.5))#None)
    model.add(Dense(10, activation='softmax'))#None)
    # 모델 compile 진행
    opt = Adam(learning_rate=0.0002)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])#None
    
    return model


def FC_Model():
    # 지시사항 3번에 따라 fully-connected Network의 구조를 작성하세요.
    # ※ 지시사항과 다른 경우, 오답 처리가 될 수 있습니다.
    model = Sequential()
    model.add(Input((28, 28, 1)))
    model.add(Flatten()) #None)
    model.add(Dense(32, activation='relu'))#None)
    model.add(Dropout(0.5))#None)
    model.add(Dense(64, activation='relu'))#None)
    model.add(Dropout(0.5))#None)
    model.add(Dense(10, activation='softmax'))#None)
    # 모델 compile 진행
    opt = Adam(learning_rate=0.0002)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])#None
    
    return model


def main():
    x_train, y_train, x_test, y_test = load_data()

    # cnn_model과 fc_model에 대해서 훈련 진행
    # 에폭(epochs)은 5, 배치사이즈(batch_size)는 32로 설정.
    # 훈련 데이터셋의 25%는 검증(validation) 셋으로 이용
    cnn_model = CNN_Model()
    cnn_model.summary()
    cnn_model.fit(
        x_train, y_train, epochs=5, batch_size=32, validation_split=0.25, verbose=2
    )

    fc_model = FC_Model()
    fc_model.summary()
    fc_model.fit(
        x_train, y_train, epochs=5, batch_size=32, validation_split=0.25, verbose=2
    )

    # 테스트 데이터셋에 적용
    cnn_test_loss, cnn_test_acc = cnn_model.evaluate(x_test, y_test, verbose=2)
    fc_test_loss, fc_test_acc = fc_model.evaluate(x_test, y_test, verbose=2)

    # cnn_model과 fc_model의 성능차이 확인
    print(
        "CNN_model acc: {}% / FC_model acc: {}%".format(
            round(cnn_test_acc * 100, 2), round(fc_test_acc * 100, 2)
        )
    )


if __name__ == "__main__":
    main()
