import matplotlib.pyplot as plt
import Constant
from keras.optimizers import Adam, Nadam
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Activation, Dropout
from keras.callbacks import EarlyStopping  # 추가된 부분
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler


class DNNModel:
    @classmethod
    def run(cls, training_feature_array, training_label_array, testing_feature_array, testing_label_array, param_type,
            param_dir) -> dict:
        results = {}
        min_epoch = 400
        epochs_list = [min_epoch, 800, 1600, 3200]

        for epochs in epochs_list:
            print(f"Training with {epochs} epochs...")

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(training_feature_array)
            X_test_scaled = scaler.transform(testing_feature_array)

            y_train = training_label_array
            y_test = testing_label_array

            n_features = X_train_scaled.shape[1]
            n_classes = 1  # For binary classification with sigmoid, we only need 1 output neuron

            model = Sequential()

            model.add(Dense(256, input_dim=n_features))
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

            model.add(Dense(32))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(0.5))

            model.add(Dense(16))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(0.5))

            model.add(Dense(n_classes))
            model.add(Activation('sigmoid'))

            # model.summary()

            optimizer = Nadam(learning_rate=0.0001)
            model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

            # EarlyStopping 콜백 설정
            early_stopping = EarlyStopping(monitor='val_loss', patience=min_epoch, verbose=1, mode='min',
                                           restore_best_weights=True)

            history = model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=32, validation_split=0.2,
                                callbacks=[early_stopping], verbose=0)

            # Plot training & validation loss values
            plt.figure(figsize=(10, 4))
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title(param_type + '_' + str(epochs))
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(loc='upper right')

            file_path \
                = Constant.FileSave.DNN_LOSS_RATE_FILE + '/' + param_dir + '/' + param_type + '_' + str(epochs) + '.png'
            plt.savefig(file_path)
            plt.close()

            pred_x = model.predict(X_test_scaled)
            pred_labels = (pred_x > 0.5).astype(int).flatten()
            true_labels = y_test
            class_report = classification_report(true_labels, pred_labels, zero_division=0, output_dict=True)

            results[epochs] = class_report

        return results
