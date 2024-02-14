from sklearn.metrics import classification_report
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler  # 표준화를 위해 StandardScaler import


class GaussianMixtureModel:
    @classmethod
    def run(cls, X, y) -> dict:
        # 표준화를 위한 StandardScaler 인스턴스화
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)  # 데이터 표준화

        gmm = GaussianMixture(n_components=2, random_state=0)

        gmm.fit(X_scaled)
        label_gmm = gmm.predict(X_scaled)  # 표준화된 데이터로 예측
        class_report = classification_report(y, label_gmm, output_dict=True, zero_division=0)

        return class_report
