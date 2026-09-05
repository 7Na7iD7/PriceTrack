import numpy as np
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.stats import ttest_rel, wilcoxon


class WeightedVotingBuilder:
    def __init__(self, cv_folds=5):
        self.cv_folds = cv_folds

    def build(self, best_estimators, cv_results, X_train, y_train):
        names = list(best_estimators.keys())
        weights = np.array([cv_results[name]["cv_mean"] for name in names])
        weights = weights / weights.sum()

        estimators = [(name, best_estimators[name]) for name in names]
        voting = VotingClassifier(
            estimators=estimators,
            voting="soft",
            weights=weights.tolist(),
        )
        voting.fit(X_train, y_train)
        return voting, dict(zip(names, weights))


class StackingBuilder:
    def __init__(self, cv_folds=5):
        self.cv_folds = cv_folds

    def build(self, best_estimators, X_train, y_train):
        estimators = [(name, model) for name, model in best_estimators.items()]
        meta_learner = LogisticRegression(max_iter=2000)
        stacking = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=self.cv_folds,
            stack_method="predict_proba",
            n_jobs=-1,
        )
        stacking.fit(X_train, y_train)
        return stacking


class EnsembleEvaluator:
    def __init__(self, cv_folds=5):
        self.cv_folds = cv_folds

    def evaluate(self, model, X_train, y_train, X_test, y_test):
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="accuracy")

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, zero_division=0)
        cm = confusion_matrix(y_test, predictions)

        return {
            "accuracy": accuracy,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "cv_scores": cv_scores,
            "report": report,
            "confusion_matrix": cm,
            "predictions": predictions,
        }


class SignificanceTester:
    def __init__(self, cv_folds=5, random_state=42):
        self.cv_folds = cv_folds
        self.random_state = random_state

    def compare(self, model_a, model_b, X_train, y_train):
        skf = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )
        scores_a = cross_val_score(model_a, X_train, y_train, cv=skf, scoring="accuracy")
        scores_b = cross_val_score(model_b, X_train, y_train, cv=skf, scoring="accuracy")

        t_stat, t_p = ttest_rel(scores_a, scores_b)
        try:
            w_stat, w_p = wilcoxon(scores_a, scores_b)
        except ValueError:
            w_stat, w_p = None, None

        return {
            "scores_a": scores_a,
            "scores_b": scores_b,
            "t_stat": t_stat,
            "t_p_value": t_p,
            "wilcoxon_stat": w_stat,
            "wilcoxon_p_value": w_p,
            "significant_at_0.05": t_p < 0.05,
        }
