from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler


class DecisionTreeModel:
    @classmethod
    def run(cls, X_train, y_train, X_test, y_test) -> dict:
        # 데이터 표준화
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 결정 트리 분류기 초기화 및 학습
        classifier = tree.DecisionTreeClassifier(random_state=0)
        classifier.fit(X_train_scaled, y_train)

        # 테스트 데이터에 대한 예측 수행
        predictions = classifier.predict(X_test_scaled)

        # 성능 평가
        report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
        return report
