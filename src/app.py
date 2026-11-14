"""Reproducible California census-block clustering and cluster classification."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import joblib
import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, silhouette_score

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["Latitude", "Longitude", "MedInc"]
SOURCE = "https://breathecode.herokuapp.com/asset/internal-link?id=439&path=housing.csv"
SEED = 42

def load_data():
    path = ROOT / "data/raw/housing.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        response = requests.get(SOURCE, timeout=60)
        response.raise_for_status()
        path.write_bytes(response.content)
    raw = pd.read_csv(path)
    missing = sorted(set(FEATURES) - set(raw.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    data = raw[FEATURES].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(data.to_numpy()).all():
        raise ValueError("Selected features contain missing or infinite values; inspect before training.")
    audit = {"source": SOURCE, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
             "rows": len(data), "features": FEATURES, "missing_values": data.isna().sum().to_dict(),
             "duplicate_feature_rows": int(data.duplicated().sum())}
    return data, audit

def split_data(data):
    # Keep identical feature triples together, without discarding distinct census blocks.
    groups = pd.util.hash_pandas_object(data, index=False)
    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED).split(data, groups=groups))
    return data.iloc[train_idx].copy(), data.iloc[test_idx].copy()

def fit_clusters(train):
    model = Pipeline([("scale", StandardScaler()),
                      ("kmeans", KMeans(n_clusters=6, n_init=20, random_state=SEED))])
    model.fit(train[FEATURES])
    return model

def label_data(model, data):
    result = data.copy()
    result["cluster"] = pd.Categorical(model.predict(data[FEATURES]), categories=range(6))
    return result

def plot_clusters(train, test=None):
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.get_cmap("tab10")
    for cluster in range(6):
        part = train[train.cluster == cluster]
        ax.scatter(part.Longitude, part.Latitude, s=8, alpha=.25, color=colors(cluster), label=f"Cluster {cluster}")
        if test is not None:
            part = test[test.cluster == cluster]
            ax.scatter(part.Longitude, part.Latitude, s=13, marker="x", alpha=.65, color=colors(cluster))
    ax.set(xlabel="Longitude", ylabel="Latitude", title="Six housing clusters" + (" — test points shown as crosses" if test is not None else " — training data"))
    ax.legend(markerscale=2, loc="best")
    ax.grid(alpha=.15)
    fig.tight_layout()
    return fig

def fit_classifier(train):
    X, y = train[FEATURES], train.cluster.astype(int)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    # This reference compares classifiers' agreement with fixed training-only K-means labels.
    candidates = {
        "decision_tree": (Pipeline([("classifier", DecisionTreeClassifier(random_state=SEED))]),
                          {"classifier__max_depth": [5, 10, 15, None], "classifier__min_samples_leaf": [1, 5, 15]}),
        "logistic_regression": (Pipeline([("scale", StandardScaler()),
                                          ("classifier", LogisticRegression(max_iter=2000))]),
                                {"classifier__C": [.1, 1, 10]})}
    searches, records = {}, []
    for name, (estimator, params) in candidates.items():
        search = GridSearchCV(estimator, params, cv=cv, scoring="accuracy", n_jobs=2, refit=True)
        search.fit(X, y, groups=pd.util.hash_pandas_object(X, index=False))
        searches[name] = search
        records.append({"model": name, "cv_accuracy": float(search.best_score_), "best_params": search.best_params_})
    winner = max(records, key=lambda row: row["cv_accuracy"])["model"]
    return searches[winner].best_estimator_, records, winner

def evaluate(classifier, test):
    expected = test.cluster.astype(int)
    predicted = classifier.predict(test[FEATURES])
    return {"accuracy": float(accuracy_score(expected, predicted)),
            "classification_report": classification_report(expected, predicted, labels=list(range(6)), output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(expected, predicted, labels=list(range(6))).tolist()}, predicted

def save_results(cluster_model, classifier, train, test, audit, comparison, winner, metrics, predicted):
    reports, models = ROOT / "reports", ROOT / "models"
    figures = reports / "figures"
    for folder in [reports, figures, models, ROOT / "data/processed"]:
        folder.mkdir(parents=True, exist_ok=True)
    for name, frame in [("train", train), ("test", test)]:
        frame.to_csv(ROOT / f"data/processed/{name}_clusters.csv", index_label="row_id")
    for name, model in [("kmeans_pipeline", cluster_model), ("classifier_pipeline", classifier)]:
        path = models / f"{name}.joblib"
        joblib.dump(model, path)
        np.testing.assert_array_equal(model.predict(test[FEATURES]), joblib.load(path).predict(test[FEATURES]))
    train_features = set(map(tuple, train[FEATURES].to_numpy()))
    test_features = set(map(tuple, test[FEATURES].to_numpy()))
    assert not train_features.intersection(test_features), "Identical features overlap splits"
    assert set(train.index).isdisjoint(test.index)
    assert set(train.cluster.astype(int)) == set(range(6))
    assert cluster_model.named_steps["scale"].n_samples_seen_ == len(train)
    centers = pd.DataFrame(cluster_model.named_steps["scale"].inverse_transform(cluster_model.named_steps["kmeans"].cluster_centers_), columns=FEATURES)
    centers.index.name = "cluster"
    centers.to_csv(reports / "cluster_centers.csv")
    summary = train.groupby("cluster", observed=False).agg(count=("MedInc", "size"), latitude_mean=("Latitude", "mean"), longitude_mean=("Longitude", "mean"), income_median=("MedInc", "median"))
    summary.to_csv(reports / "cluster_summary.csv")
    for name, figure in [("training_clusters", plot_clusters(train)), ("train_test_clusters", plot_clusters(train, test))]:
        figure.savefig(figures / f"{name}.png", dpi=150)
        plt.close(figure)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(np.array(metrics["confusion_matrix"]), display_labels=range(6)).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Classifier agreement with K-means labels")
    fig.tight_layout();fig.savefig(figures / "confusion_matrix.png", dpi=150);plt.close(fig)
    labeled = test.copy();labeled["classifier_cluster"] = predicted
    labeled.to_csv(reports / "test_predictions.csv", index_label="row_id")
    scaled_train = cluster_model.named_steps["scale"].transform(train[FEATURES])
    results = {"data_audit": audit, "seed": SEED, "train_rows": len(train), "test_rows": len(test),
               "n_clusters": 6, "selected_classifier": winner, "model_comparison": comparison,
               "train_silhouette_sample": float(silhouette_score(scaled_train, train.cluster.astype(int), sample_size=min(3000, len(train)), random_state=SEED)),
               "test_metrics": metrics, "verification": {"disjoint_rows": True, "disjoint_feature_triples": True,
               "preprocessing_fit_on_training_only": True, "six_training_clusters": True, "saved_models_reload_identically": True},
               "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__}}
    (reports / "evaluation.json").write_text(json.dumps(results, indent=2) + "\n")
    return results

def run():
    data, audit = load_data()
    train, test = split_data(data)
    clustering = fit_clusters(train)
    train, test = label_data(clustering, train), label_data(clustering, test)
    classifier, comparison, winner = fit_classifier(train)
    metrics, predictions = evaluate(classifier, test)
    return save_results(clustering, classifier, train, test, audit, comparison, winner, metrics, predictions)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predict", nargs=3, type=float, metavar=("LATITUDE", "LONGITUDE", "MEDINC"))
    args = parser.parse_args()
    if args.predict:
        frame = pd.DataFrame([args.predict], columns=FEATURES)
        if not np.isfinite(frame.to_numpy()).all():
            parser.error("Prediction inputs must be finite numbers")
        print(json.dumps({name: int(joblib.load(ROOT / f"models/{file}.joblib").predict(frame)[0]) for name, file in [("kmeans_cluster", "kmeans_pipeline"), ("classifier_cluster", "classifier_pipeline")]}))
    else:
        results = run()
        print(json.dumps({"train_rows": results["train_rows"], "test_rows": results["test_rows"], "selected_classifier": results["selected_classifier"], "test_accuracy": results["test_metrics"]["accuracy"], "verification": results["verification"]}, indent=2))

if __name__ == "__main__":
    main()
