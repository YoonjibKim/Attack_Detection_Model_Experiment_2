from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler


class RandomForestModel:
    @classmethod
    def run(cls, X_train, y_train, X_test, y_test) -> dict:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        classifier = RandomForestClassifier(random_state=0)
        classifier.fit(X_train_scaled, y_train.ravel())

        predictions = classifier.predict(X_test_scaled)

        report = classification_report(y_test, predictions, zero_division=0, output_dict=True)

        return report
