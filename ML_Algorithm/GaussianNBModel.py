from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler


class GaussianNBModel:
    @classmethod
    def run(cls, training_feature_array, training_label_array, testing_feature_array, testing_label_array) -> dict:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(training_feature_array)
        X_test_scaled = scaler.transform(testing_feature_array)

        clf_gnb = GaussianNB()
        clf_gnb.fit(X_train_scaled, training_label_array.ravel())

        predictions = clf_gnb.predict(X_test_scaled)

        report = classification_report(testing_label_array, predictions, output_dict=True, zero_division=0)

        return report
    
    def grid_run(self):
        pass