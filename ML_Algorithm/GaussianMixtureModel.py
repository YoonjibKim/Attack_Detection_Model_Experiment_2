from sklearn.metrics import classification_report
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


class GaussianMixtureModel:
    @classmethod
    def run(cls, X, y) -> dict:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        gmm = GaussianMixture(n_components=2, random_state=0)

        gmm.fit(X_scaled)
        label_gmm = gmm.predict(X_scaled)
        class_report = classification_report(y, label_gmm, output_dict=True, zero_division=0)

        return class_report
