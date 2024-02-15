import matplotlib.pyplot as plt
import Constant
from keras.optimizers import Nadam
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Activation, Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler


class DNNModel:
    @classmethod
    def run(cls, training_feature_array, training_label_array, testing_feature_array, testing_label_array,
            epoch_step, learning_rate, loss_rate_file_path) -> dict:
        epochs = Constant.DNNParameters.epochStep[epoch_step]

        patience = epochs  # 추후 조정 필요

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(training_feature_array)
        X_test_scaled = scaler.transform(testing_feature_array)

        y_train = training_label_array
        y_test = testing_label_array

        n_features = X_train_scaled.shape[1]
        n_classes = 1

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

        optimizer = Nadam(learning_rate=learning_rate)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='min',
                                       restore_best_weights=True)

        history = model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=32, validation_split=0.2,
                            callbacks=[early_stopping], verbose=0)

        actual_epochs = len(history.history['loss'])

        print(f"Training stopped after {actual_epochs} epochs.")

        plt.figure(figsize=(10, 4))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss over Epochs at ' + str(actual_epochs) + ', Learning Rate at ' + str(learning_rate))
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')

        plt.savefig(loss_rate_file_path)
        plt.close()

        pred_x = model.predict(X_test_scaled)
        pred_labels = (pred_x > 0.5).astype(int).flatten()
        true_labels = y_test
        class_report = classification_report(true_labels, pred_labels, zero_division=0, output_dict=True)

        results = {
            Constant.CLASSIFICATION_REPORT: class_report,
            Constant.EPOCHS: actual_epochs,
            Constant.LEARNING_RATE: learning_rate
        }

        return results

    def grid_run(self):
        pass