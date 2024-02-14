from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler


class GradientBoostModel:
    @classmethod
    def run(cls, X_train, y_train, X_test, y_test) -> dict:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf_gbt = GradientBoostingClassifier(random_state=0)
        clf_gbt.fit(X_train_scaled, y_train.ravel())
        predictions = clf_gbt.predict(X_test_scaled)
        report = classification_report(y_test, predictions, output_dict=True, zero_division=0)

        return report
