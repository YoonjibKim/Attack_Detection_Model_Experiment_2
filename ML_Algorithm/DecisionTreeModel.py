from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler


class DecisionTreeModel:
    @classmethod
    def run(cls, X_train, y_train, X_test, y_test) -> dict:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        classifier = tree.DecisionTreeClassifier(random_state=0)
        classifier.fit(X_train_scaled, y_train)

        predictions = classifier.predict(X_test_scaled)

        report = classification_report(y_test, predictions, output_dict=True, zero_division=0)

        return report

    def grid_run(self):
        pass