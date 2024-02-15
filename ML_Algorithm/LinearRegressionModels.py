import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler


class LinearRegressionModels:
    @classmethod
    def run(cls, X_tn, y_tn, X_te, y_te):
        std_scale = StandardScaler()
        std_scale.fit(X_tn)
        X_tn_std = std_scale.transform(X_tn)
        X_te_std = std_scale.transform(X_te)

        clf_lr = LinearRegression()
        clf_lr.fit(X_tn_std, y_tn)
        pred_lr = clf_lr.predict(X_te_std)
        class_report_lr = classification_report(y_te, np.round(pred_lr), zero_division=0, output_dict=True)

        clf_ridge = Ridge(random_state=0)
        clf_ridge.fit(X_tn_std, y_tn)
        pred_ridge = clf_ridge.predict(X_te_std)
        class_report_ridge = classification_report(y_te, np.round(pred_ridge), zero_division=0, output_dict=True)

        clf_lasso = Lasso(random_state=0)
        clf_lasso.fit(X_tn_std, y_tn)
        pred_lasso = clf_lasso.predict(X_te_std)
        class_report_lasso = classification_report(y_te, np.round(pred_lasso), zero_division=0, output_dict=True)

        clf_elastic = ElasticNet(random_state=0)
        clf_elastic.fit(X_tn_std, y_tn)
        pred_elastic = clf_elastic.predict(X_te_std)
        class_report_elastic = classification_report(y_te, np.round(pred_elastic), zero_division=0, output_dict=True)

        return class_report_lr, class_report_ridge, class_report_lasso, class_report_elastic

    def grid_run(self):
        pass