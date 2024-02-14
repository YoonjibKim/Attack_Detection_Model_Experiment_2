from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

class AgglomerativeClusteringModel:
    @classmethod
    def run(cls, X, y):
        # 데이터 표준화
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 계층적 군집화 수행
        agglomerative_clustering = AgglomerativeClustering(n_clusters=2, linkage='complete')
        labels = agglomerative_clustering.fit_predict(X_scaled)

        # 성능 평가
        report = classification_report(y, labels, output_dict=True, zero_division=0)
        return report
