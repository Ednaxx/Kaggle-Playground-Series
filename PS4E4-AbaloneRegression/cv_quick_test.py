import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


def cv_quick_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    regressor = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    regressor.fit(X_train, y_train)
    score = cross_val_score(regressor, X, y, scoring="neg_mean_squared_log_error", cv=5).mean()
    score = np.sqrt(-score)
    return score
