from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class DimensionalityReducer:
    def __init__(self, n_components=10):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)

    def fit_transform(self, X_train):
        return self.pca.fit_transform(X_train)

    def transform(self, X_test):
        return self.pca.transform(X_test)

    def explained_variance_ratio(self):
        return self.pca.explained_variance_ratio_


class ModelService:
    def __init__(self, cv_folds=5, n_jobs=-1):
        self.cv_folds = cv_folds
        self.n_jobs = n_jobs
        self.param_grids = {
            "Logistic Regression": {
                "estimator": LogisticRegression(max_iter=2000),
                "params": {
                    "C": [0.01, 0.1, 1, 10, 100],
                    "solver": ["lbfgs"],
                },
            },
            "KNN": {
                "estimator": KNeighborsClassifier(),
                "params": {
                    "n_neighbors": [3, 5, 7, 9, 11, 15],
                    "weights": ["uniform", "distance"],
                    "p": [1, 2],
                },
            },
            "Naive Bayes": {
                "estimator": GaussianNB(),
                "params": {
                    "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6],
                },
            },
            "SVM": {
                "estimator": SVC(),
                "params": {
                    "C": [0.1, 1, 10, 100],
                    "kernel": ["rbf", "linear"],
                    "gamma": ["scale", "auto"],
                },
            },
            "Random Forest": {
                "estimator": RandomForestClassifier(random_state=42),
                "params": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                },
            },
            "Gradient Boosting": {
                "estimator": GradientBoostingClassifier(random_state=42),
                "params": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [2, 3, 4],
                },
            },
        }
        self.best_estimators = {}
        self.results = {}

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        for name, config in self.param_grids.items():
            search = GridSearchCV(
                estimator=config["estimator"],
                param_grid=config["params"],
                cv=self.cv_folds,
                scoring="accuracy",
                n_jobs=self.n_jobs,
            )
            search.fit(X_train, y_train)

            best_model = search.best_estimator_
            self.best_estimators[name] = best_model

            cv_scores = cross_val_score(
                best_model, X_train, y_train, cv=self.cv_folds, scoring="accuracy"
            )

            predictions = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            report = classification_report(y_test, predictions, zero_division=0)
            cm = confusion_matrix(y_test, predictions)

            self.results[name] = {
                "accuracy": accuracy,
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "best_params": search.best_params_,
                "report": report,
                "confusion_matrix": cm,
                "predictions": predictions,
            }
        return self.results

    def best_model(self):
        return max(self.results.items(), key=lambda item: item[1]["accuracy"])
