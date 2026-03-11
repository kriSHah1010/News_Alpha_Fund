import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import joblib


class XGBModel:

    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss"
        )

    def train(self, X, y):

        tscv = TimeSeriesSplit(n_splits=5)

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            self.model.fit(X_train, y_train)

            preds = self.model.predict(X_test)
            acc = accuracy_score(y_test, preds)

            print(f"Fold {fold+1} Accuracy: {acc:.4f}")

        return self.model

    def save(self, path):
        joblib.dump(self.model, path)