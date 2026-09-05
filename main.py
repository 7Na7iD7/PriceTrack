import logging
import sys
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import DataLoader, DataLoadError
from model_service import DimensionalityReducer, ModelService
from ensemble_service import (
    WeightedVotingBuilder,
    StackingBuilder,
    EnsembleEvaluator,
    SignificanceTester,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mobile_price_project")

CSV_PATH = "mobile_price_data.csv"
PCA_COMPONENTS = 10


def plot_all_results(results, filename, title):
    names = list(results.keys())
    accuracies = [results[name]["accuracy"] * 100 for name in names]
    cv_means = [results[name]["cv_mean"] * 100 for name in names]
    cv_stds = [results[name]["cv_std"] * 100 for name in names]

    x = range(len(names))
    plt.figure(figsize=(11, 6))
    plt.bar(x, accuracies, width=0.4, label="Test Accuracy", color="#4C72B0")
    plt.errorbar(
        x, cv_means, yerr=cv_stds, fmt="o", color="#DD8452", label="CV Accuracy (mean ± std)"
    )
    plt.xticks(list(x), names, rotation=20)
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.ylim(0, 100)
    plt.legend()

    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 1, f"{acc:.2f}%", ha="center")

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    logger.info(f"نمودار مقایسه در {filename} ذخیره شد.")


def plot_significance(sig_result, filename, label_a, label_b):
    plt.figure(figsize=(7, 5))
    plt.boxplot(
        [sig_result["scores_a"], sig_result["scores_b"]],
        labels=[label_a, label_b],
    )
    plt.ylabel("CV Accuracy")
    plt.title(
        f"paired t-test p={sig_result['t_p_value']:.4f} | "
        f"wilcoxon p={sig_result['wilcoxon_p_value']}"
    )
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    logger.info(f"نمودار آزمون آماری در {filename} ذخیره شد.")


def run():
    logger.info("شروع اجرای پروژه دسته‌بندی قیمت موبایل")

    try:
        loader = DataLoader(csv_path=CSV_PATH, target_column="price_range")
        X_train, X_test, y_train, y_test, feature_names = loader.split_and_scale()
        logger.info(f"داده‌ها بارگذاری شدند. تعداد ویژگی‌های اولیه: {len(feature_names)}")
    except DataLoadError as exc:
        logger.error(str(exc))
        sys.exit(1)

    logger.info("مرحله ۱: کاهش ابعاد با PCA")
    reducer = DimensionalityReducer(n_components=PCA_COMPONENTS)
    X_train_pca = reducer.fit_transform(X_train)
    X_test_pca = reducer.transform(X_test)
    variance_kept = reducer.explained_variance_ratio().sum() * 100
    logger.info(f"واریانس حفظ‌شده پس از PCA: {variance_kept:.2f}%")

    logger.info("مرحله ۲: آموزش و تیونینگ مدل‌های پایه")
    service = ModelService()
    base_results = service.train_and_evaluate(X_train_pca, y_train, X_test_pca, y_test)
    for name, result in base_results.items():
        logger.info(
            f"{name}: تست = {result['accuracy'] * 100:.2f}% | "
            f"CV = {result['cv_mean'] * 100:.2f}% ± {result['cv_std'] * 100:.2f}%"
        )
    plot_all_results(base_results, "base_models_accuracy.png", "Base Models Accuracy")

    logger.info("مرحله ۳: ساخت مدل Weighted Soft Voting")
    voting_builder = WeightedVotingBuilder()
    voting_model, weights = voting_builder.build(
        service.best_estimators, base_results, X_train_pca, y_train
    )
    logger.info(f"وزن‌های اختصاص‌یافته به مدل‌ها: {weights}")

    logger.info("مرحله ۴: ساخت مدل Stacking")
    stacking_builder = StackingBuilder()
    stacking_model = stacking_builder.build(service.best_estimators, X_train_pca, y_train)

    evaluator = EnsembleEvaluator()
    voting_result = evaluator.evaluate(voting_model, X_train_pca, y_train, X_test_pca, y_test)
    stacking_result = evaluator.evaluate(stacking_model, X_train_pca, y_train, X_test_pca, y_test)

    logger.info(
        f"Weighted Voting: تست = {voting_result['accuracy'] * 100:.2f}% | "
        f"CV = {voting_result['cv_mean'] * 100:.2f}% ± {voting_result['cv_std'] * 100:.2f}%"
    )
    logger.info(
        f"Stacking: تست = {stacking_result['accuracy'] * 100:.2f}% | "
        f"CV = {stacking_result['cv_mean'] * 100:.2f}% ± {stacking_result['cv_std'] * 100:.2f}%"
    )

    all_results = dict(base_results)
    all_results["Weighted Voting"] = voting_result
    all_results["Stacking"] = stacking_result
    plot_all_results(all_results, "all_models_accuracy.png", "All Models vs Ensembles")

    best_single_name, best_single_result = service.best_model()
    best_ensemble_name = "Stacking" if stacking_result["accuracy"] >= voting_result["accuracy"] else "Weighted Voting"
    best_ensemble_model = stacking_model if best_ensemble_name == "Stacking" else voting_model

    logger.info("مرحله ۵: آزمون معنی‌داری آماری بین بهترین مدل تکی و مدل ترکیبی")
    tester = SignificanceTester()
    sig_result = tester.compare(
        service.best_estimators[best_single_name], best_ensemble_model, X_train_pca, y_train
    )
    logger.info(f"t-test p-value = {sig_result['t_p_value']:.4f}")
    logger.info(f"wilcoxon p-value = {sig_result['wilcoxon_p_value']}")
    logger.info(f"معنی‌دار در سطح ۰.۰۵: {sig_result['significant_at_0.05']}")
    plot_significance(
        sig_result, "significance_test.png", best_single_name, best_ensemble_name
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(voting_result["confusion_matrix"], annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_title("Weighted Voting")
    sns.heatmap(stacking_result["confusion_matrix"], annot=True, fmt="d", cmap="Greens", ax=axes[1])
    axes[1].set_title("Stacking")
    plt.tight_layout()
    plt.savefig("ensemble_confusion_matrices.png", dpi=150)
    plt.close()

    logger.info(f"بهترین مدل تکی: {best_single_name} = {best_single_result['accuracy'] * 100:.2f}%")
    logger.info(f"بهترین مدل ترکیبی: {best_ensemble_name} = {max(voting_result['accuracy'], stacking_result['accuracy']) * 100:.2f}%")
    logger.info("اجرای پروژه با موفقیت به پایان رسید.")


if __name__ == "__main__":
    run()
