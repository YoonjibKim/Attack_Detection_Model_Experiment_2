from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler


class KMeansModel:
    @classmethod
    def run(cls, X, y) -> dict:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=2, init='k-means++', random_state=0)
        kmeans.fit(X_scaled)
        labels = kmeans.labels_

        report = classification_report(y, labels, output_dict=True, zero_division=0)

        return report
