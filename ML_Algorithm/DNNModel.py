import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Activation, Dropout
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler


class DNNModel:
    @classmethod
    def run(cls, training_feature_array, training_label_array, testing_feature_array, testing_label_array):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(training_feature_array)
        X_test_scaled = scaler.transform(testing_feature_array)

        y_train = to_categorical(training_label_array)
        y_test = to_categorical(testing_label_array)  # 테스트 레이블도 원-핫 인코딩으로 변환

        n_features = X_train_scaled.shape[1]
        epochs = 100
        n_classes = 2  # 이진 분류

        model = Sequential()

        # 모델 구성
        model.add(Dense(64, input_dim=n_features))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(128))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(n_classes))
        model.add(Activation('softmax'))

        model.summary()

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        history = model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=10, validation_split=0.2)

        # 학습 손실률 시각화
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        plt.show()

        pred_x = model.predict(X_test_scaled)
        pred_labels = np.argmax(pred_x, axis=1)
        true_labels = np.argmax(y_test, axis=1)  # 원-핫 인코딩에서 가장 큰 값을 가지는 인덱스 추출
        class_report = classification_report(true_labels, pred_labels, zero_division=0, output_dict=True)

        return class_report
