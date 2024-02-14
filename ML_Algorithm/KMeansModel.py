from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

class KMeansModel:
    @classmethod
    def run(cls, X, y):
        # 데이터 표준화
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # KMeans 클러스터링 적용
        kmeans = KMeans(n_clusters=2, init='k-means++', random_state=0)
        kmeans.fit(X_scaled)
        labels = kmeans.labels_

        # 분류 보고서 생성
        report = classification_report(y, labels, output_dict=True, zero_division=0)
        return report
