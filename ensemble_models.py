"""Подготовка признаков и итоговый эксперимент на переданном train/test split."""
from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from grader_contracts.ensembles import EnsembleInput, EnsembleResult


# Порядок столбцов german.csv без Creditability — для матриц без имён.
_GERMAN_FEATURES = (
    "Account_Balance", "Duration_of_Credit_monthly", "Payment_Status_of_Previous_Credit",
    "Purpose", "Credit_Amount", "Value_Savings_Stocks", "Length_of_current_employment",
    "Instalment_per_cent", "Sex_Marital_Status", "Guarantors", "Duration_in_Current_address",
    "Most_valuable_available_asset", "Age_years", "Concurrent_Credits", "Type_of_apartment",
    "No_of_Credits_at_this_Bank", "Occupation", "No_of_dependents", "Telephone", "Foreign_Worker",
)
_NOMINAL = {
    "Purpose", "Sex_Marital_Status", "Guarantors", "Concurrent_Credits",
    "Type_of_apartment", "Occupation",
}
MODEL_NAME = "Random Forest (one-hot + ratios, leaf=3, sqrt)"


def _feature_frame(features, *, add_ratios):
    frame = pd.DataFrame(features)
    if frame.columns.equals(pd.RangeIndex(len(_GERMAN_FEATURES))):
        frame = frame.set_axis(_GERMAN_FEATURES, axis="columns")
    else:
        frame = frame.rename(columns=str)

    if add_ratios and {"Credit_Amount", "Duration_of_Credit_monthly"} <= set(frame):
        term = pd.to_numeric(frame["Duration_of_Credit_monthly"], errors="coerce")
        amount = pd.to_numeric(frame["Credit_Amount"], errors="coerce")
        monthly = amount / term.where(term > 0)
        frame = frame.assign(Amount_per_month=monthly)
        if "Instalment_per_cent" in frame:
            band = pd.to_numeric(frame["Instalment_per_cent"], errors="coerce")
            frame = frame.assign(Monthly_per_instalment_band=monthly / band.where(band > 0))
    return frame


def _categorical_columns(frame, *, one_hot):
    return [
        name for name, column in frame.items()
        if (one_hot and name in _NOMINAL) or not is_numeric_dtype(column)
    ]


def tree_pipeline(estimator, *, one_hot=True, add_ratios=True):
    """Все обучаемые преобразования входят в pipeline и fit-ятся внутри фолда."""
    categories = make_pipeline(
        SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")
    )
    preprocess = ColumnTransformer(
        [("categories", categories, partial(_categorical_columns, one_hot=one_hot))],
        remainder=SimpleImputer(strategy="median"),
        sparse_threshold=0,
    )
    return make_pipeline(
        FunctionTransformer(_feature_frame, kw_args={"add_ratios": add_ratios}),
        preprocess,
        estimator,
    )


def make_forest(seed, *, min_samples_leaf=3, max_features="sqrt", one_hot=True, add_ratios=True):
    forest = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=min_samples_leaf, max_features=max_features,
        random_state=seed, n_jobs=1,
    )
    return tree_pipeline(forest, one_hot=one_hot, add_ratios=add_ratios)


def fit_and_evaluate(model, model_name: str, data: EnsembleInput) -> EnsembleResult:
    """Одинаковый расчёт вероятностей и метрик для валидации и финального теста."""
    model.fit(data.train_features, np.asarray(data.train_target).ravel())
    positive = list(model.classes_).index(1)
    probabilities = model.predict_proba(data.test_features)[:, positive]
    predictions = (probabilities >= 0.5).astype(int)

    y = np.asarray(data.test_target).ravel()
    return EnsembleResult(
        model_name=model_name,
        probabilities=probabilities,
        predictions=predictions,
        roc_auc=float(roc_auc_score(y, probabilities)),
        accuracy=float(accuracy_score(y, predictions)),
        precision=float(precision_score(y, predictions, zero_division=0)),
        recall=float(recall_score(y, predictions, zero_division=0)),
        f1=float(f1_score(y, predictions, zero_division=0)),
    )


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    """Обучить выбранный лес только на train и оценить класс Creditability=1 на test."""
    return fit_and_evaluate(make_forest(data.random_seed), MODEL_NAME, data)
