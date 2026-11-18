"""Сравнить модели на train и записать оценку победителя на отложенном test."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from ensemble_models import MODEL_NAME, fit_and_evaluate, make_forest, tree_pipeline
from grader_contracts.ensembles import EnsembleInput


ROOT = Path(__file__).resolve().parent
SEED = 42
TEST_SIZE = 0.2
N_SPLITS, N_REPEATS = 5, 2
METRICS = ("roc_auc", "accuracy", "precision", "recall", "f1")


def german_split():
    x = pd.read_csv(ROOT / "german.csv", sep=";")
    y = x.pop("Creditability")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    return EnsembleInput(x_train, y_train, x_test, y_test, random_seed=SEED)


def candidates(seed):
    forest = make_forest(seed)
    boost = tree_pipeline(GradientBoostingClassifier(
        n_estimators=240, learning_rate=0.035, max_depth=2, min_samples_leaf=12,
        random_state=seed,
    ))
    extra = tree_pipeline(ExtraTreesClassifier(
        n_estimators=300, min_samples_leaf=3, max_features="sqrt", n_jobs=1, random_state=seed,
    ))
    return {
        "Random Forest (codes)": make_forest(seed, one_hot=False, add_ratios=False),
        "Random Forest (one-hot)": make_forest(seed, add_ratios=False),
        "Random Forest (codes + ratios)": make_forest(seed, one_hot=False),
        MODEL_NAME: forest,
        "Random Forest (one-hot + ratios, leaf=1)": make_forest(seed, min_samples_leaf=1),
        "Random Forest (one-hot + ratios, leaf=8)": make_forest(seed, min_samples_leaf=8),
        "Random Forest (one-hot + ratios, half of features)": make_forest(seed, max_features=0.5),
        "Extra Trees (one-hot + ratios)": extra,
        "Gradient Boosting (one-hot + ratios)": boost,
        "Soft Voting (Random Forest + Gradient Boosting)": VotingClassifier(
            [("forest", forest), ("boost", boost)], voting="soft", n_jobs=1
        ),
    }


def compare_on_train(models, x, y, seed):
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=seed)
    folds = [
        EnsembleInput(x.iloc[t], y.iloc[t], x.iloc[v], y.iloc[v], random_seed=seed)
        for t, v in cv.split(x, y)
    ]
    experiments = []
    for name, model in models.items():
        scores = [fit_and_evaluate(clone(model), name, fold) for fold in folds]
        experiments.append({
            "model_name": name,
            "cv_mean": {m: float(np.mean([getattr(s, m) for s in scores])) for m in METRICS},
            "cv_roc_auc_std": float(np.std([s.roc_auc for s in scores])),
        })
    return experiments


def main():
    data = german_split()
    models = candidates(data.random_seed)
    experiments = compare_on_train(models, data.train_features, data.train_target, data.random_seed)
    winner = max(experiments, key=lambda row: row["cv_mean"]["roc_auc"])["model_name"]
    final = fit_and_evaluate(models[winner], winner, data)
    test_metrics = {m: getattr(final, m) for m in METRICS}
    results = {
        "dataset": "german.csv", "target": "Creditability", "positive_class": 1,
        "random_seed": data.random_seed,
        "split": {"stratified": True, "test_size": TEST_SIZE,
                  "n_train": len(data.train_target), "n_test": len(data.test_target)},
        "cv": {"strategy": "RepeatedStratifiedKFold", "n_splits": N_SPLITS, "n_repeats": N_REPEATS},
        "selection_criterion": "highest mean validation ROC-AUC",
        "experiments": experiments,
        "model_name": final.model_name,
        "test_metrics": test_metrics,
        "test_confusion_matrix_labels_0_1": confusion_matrix(
            data.test_target, final.predictions, labels=[0, 1]
        ).tolist(),
    }
    (ROOT / "experiment_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(winner)
    print("Test:", test_metrics)


if __name__ == "__main__":
    main()
