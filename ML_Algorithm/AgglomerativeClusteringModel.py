from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler


class AgglomerativeClusteringModel:
    @classmethod
    def run(cls, X, y) -> dict:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        agglomerative_clustering = AgglomerativeClustering(n_clusters=2, linkage='complete')
        labels = agglomerative_clustering.fit_predict(X_scaled)

        report = classification_report(y, labels, output_dict=True, zero_division=0)

        return report

    def grid_run(self):
        pass
