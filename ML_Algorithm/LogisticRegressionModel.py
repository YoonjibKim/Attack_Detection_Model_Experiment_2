from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler


class LogisticRegressionModel:
    @classmethod
    def run(cls, X_train, y_train, X_test, y_test):
        # 데이터 표준화
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 로지스틱 회귀 모델 초기화 및 학습
        classifier = LogisticRegression(random_state=0)
        classifier.fit(X_train_scaled, y_train.ravel())

        # 테스트 데이터에 대한 예측 수행
        predictions = classifier.predict(X_test_scaled)

        # 성능 평가
        report = classification_report(y_test, predictions, zero_division=0, output_dict=True)

        return report
