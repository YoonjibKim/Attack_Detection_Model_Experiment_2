from sklearn.cluster import DBSCAN
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np


class DBSCANModel:
    @classmethod
    def run(cls, X, y) -> dict:
        # 데이터 표준화
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # DBSCAN 알고리즘 적용, eps와 min_samples는 사용자가 지정
        eps = 0.5  # 예시 값
        min_samples = 5  # 예시 값
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(X_scaled)
        labels = dbscan.labels_

        # 노이즈 포인트 제외하고 유니크 레이블 추출
        unique_labels = np.unique(labels[labels >= 0])

        # 클러스터링 결과에 대한 분류 보고서 생성
        class_report = classification_report(y, labels, output_dict=True, zero_division=0, labels=unique_labels)

        return class_report
