from sklearn.cluster import DBSCAN
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np


class DBSCANModel:
    @classmethod
    def run(cls, X, y) -> dict:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        eps = 0.5
        min_samples = 5
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(X_scaled)
        labels = dbscan.labels_

        unique_labels = np.unique(labels[labels >= 0])

        class_report = classification_report(y, labels, output_dict=True, zero_division=0, labels=unique_labels)

        return class_report

    def grid_run(self):
        pass
