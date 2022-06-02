"""Helper functions for model performance evaluation."""
from collections import defaultdict
from copy import deepcopy
from typing import Callable

import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, mean_absolute_error, ndcg_score,
                             precision_recall_curve, roc_auc_score)
from tqdm.auto import tqdm


def get_precision_at_k(y_true, y_preds, k=0.9):
    """
    Calculate the precision @ a minimum recall

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        True binary labels {0, 1}
    y_preds : ndarray of shape (n_samples,)
        Predicted score within [0,1]
    k : float
        The given minimum recall

    Returns
    -------
    typing.Dict[str, float]
        Dictionary with keys precision, recall, and threshold
        @ the given minimum recall.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_preds)
    for index, recall in enumerate(recalls):
        if recall < k:
            final_index = index - 1
            break

    return {
        "precision": precisions[final_index],
        "recalls": recalls[final_index],
        "thresholds": thresholds[final_index],
    }


def evaluate_features(
    features: list[str],
    model,
    full_df: pd.DataFrame,
    base_features: list[str] = None,
    label_col: str = "label",
    sample_weight_col: str = None,
    fold_col: str = "fold",
    metric: str = "pairwise_ranking_score",
    k: float = 0.9,
):
    """
    Evaluate a feature subset with the precision @ a minimum recall

    Parameters
    ----------
    features: list[str]
        List of features (column names) to be evaluated.
    model: sklearn model
        Sklearn model object.
    full_df: pd.DataFrame
        Dataframe containing all the training features and a binary label column.
    base_features: list[str]
        List of features (column names) served as the baseline features.
    label_col: str
        Column name of the binary label.
    sample_weight_col: str
        Column name specifying the sample weight.
    fold_col: str
        Column name specifying the validation fold.
    metric: str
        String specifying the metric, the default is pairwise_ranking_score.
    k: float
        Minimum recall

    Returns
    -------
        typing.Tuple[typing.List[float], typing.List[str]]
        Precision scores of each fold and the used features
    """
    if not base_features:
        base_features = []
    feas = base_features + features
    scores = []
    for fold in range(int(full_df[fold_col].max()) + 1):
        curr_model = deepcopy(model)
        if sample_weight_col:
            curr_model.fit(
                full_df.loc[full_df[fold_col] != fold, feas],
                full_df.loc[full_df[fold_col] != fold, label_col],
                sample_weight=full_df.loc[full_df[fold_col] != fold, sample_weight_col],
            )
        else:
            curr_model.fit(
                full_df.loc[full_df[fold_col] != fold, feas],
                full_df.loc[full_df[fold_col] != fold, label_col],
            )
        preds = curr_model.predict_proba(full_df.loc[full_df[fold_col] == fold, feas])[
            :, 1
        ]
        if metric == "pairwise_ranking_score":
            full_df.loc[full_df[fold_col] == fold, "preds"] = preds
        elif metric == "auc":
            scores.append(
                roc_auc_score(full_df.loc[full_df[fold_col] == fold, label_col], preds)
            )
        elif metric == "precision_at_k":
            scores.append(
                get_precision_at_k(
                    full_df.loc[full_df[fold_col] == fold, label_col], preds, k=k
                )["precision"]
            )
        else:
            raise Exception("Metric is not implemented")

    if metric == "pairwise_ranking_score":
        df_tmp = full_df.loc[
            full_df.groupby("job_id")[label_col].transform("nunique") > 1
        ].reset_index(drop=True)
        avg_scores = pairwise_ranking_score(df_tmp, "preds", label_col=label_col)
        return avg_scores, feas

    return np.mean(scores), feas, scores


def evaluate_best_f1(y_true, y_preds):
    """
    Evaluate the best possible f1 score.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        Binary labels {0, 1}
    y_preds : ndarray of shape (n_samples,)
        Predicted score within [0,1]

    Returns
    -------
    typing.Dict[str, float]
        Best f1 score, precision, recall, and threshold @ the best f1
    typing.Tuple[
            typing.List[float], typing.List[str]
        ]
        Precision scores of each fold and the used features
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_preds)
    f1_scores = 2 * recalls * precisions / (recalls + precisions + 1e-6)
    best_f1_idx = np.argmax(f1_scores)
    return {
        "f1": f1_scores[best_f1_idx],
        "precision": precisions[best_f1_idx],
        "recall": recalls[best_f1_idx],
        "threshold": thresholds[best_f1_idx],
    }


def evaluation_binary_classification(full_df, pred_col, gt_col="label", fold_col=None):
    """
    Evaluate the following metrics of a binary classification model:
        1. precision @ 90/75/50/25/10 recall
        2. Area Under the ROC Curve (AUC)
        3. Best F1 score

    Parameters
    ----------

    df : pd.DataFrame
        Dataframe with a binary label column and a predicted score column
    pred_col : str
        Column name of the predicted score within [0,1]
    gt_col : str
        Column name of the binary label
    fold_col : str
        Column name specifying the validation fold

    Returns
    -------
    typing.Dict[str, float]
        Metrics specified in 1, 2, 3.
    """
    if fold_col is not None:
        metrics_df = {
            "precision_at_90recall": 0,
            "precision_at_75recall": 0,
            "precision_at_50recall": 0,
            "precision_at_25recall": 0,
            "precision_at_10recall": 0,
            "auc": 0,
            "accuracy": 0,
        }
        for fold in range(int(full_df[fold_col].max()) + 1):
            y_true = full_df.loc[full_df[fold_col] == fold, gt_col]
            preds = full_df.loc[full_df[fold_col] == fold, pred_col]
            metrics_df["precision_at_90recall"] += get_precision_at_k(y_true, preds)[
                "precision"
            ]
            metrics_df["precision_at_75recall"] += get_precision_at_k(
                y_true, preds, 0.75
            )["precision"]
            metrics_df["precision_at_50recall"] += get_precision_at_k(
                y_true, preds, 0.5
            )["precision"]
            metrics_df["precision_at_25recall"] += get_precision_at_k(
                y_true, preds, 0.25
            )["precision"]
            metrics_df["precision_at_10recall"] += get_precision_at_k(
                y_true, preds, 0.1
            )["precision"]
            metrics_df["auc"] += roc_auc_score(y_true, preds)
            metrics_df["accuracy"] += accuracy_score(y_true, np.round(preds))
        metrics_df = {k: v / full_df[fold_col].nunique() for k, v in metrics_df.items()}
    else:
        metrics_df = {}
        y_true = full_df[gt_col]
        preds = full_df[pred_col]
        metrics_df["precision_at_90recall"] = get_precision_at_k(y_true, preds)[
            "precision"
        ]
        metrics_df["precision_at_75recall"] = get_precision_at_k(y_true, preds, 0.75)[
            "precision"
        ]
        metrics_df["precision_at_50recall"] = get_precision_at_k(y_true, preds, 0.5)[
            "precision"
        ]
        metrics_df["precision_at_25recall"] = get_precision_at_k(y_true, preds, 0.25)[
            "precision"
        ]
        metrics_df["precision_at_10recall"] = get_precision_at_k(y_true, preds, 0.1)[
            "precision"
        ]
        metrics_df["auc"] = roc_auc_score(y_true, preds)
        metrics_df["accuracy"] = accuracy_score(y_true, np.round(preds))

    best_f1_dict = evaluate_best_f1(full_df[gt_col], full_df[pred_col])
    metrics_df["best_f1"] = best_f1_dict["f1"]
    metrics_df["recall_at_best_f1"] = best_f1_dict["precision"]
    metrics_df["precision_at_best_f1"] = best_f1_dict["recall"]
    metrics_df["thr_at_best_f1"] = best_f1_dict["threshold"]
    return metrics_df


def pairwise_ranking_score(
    full_df, pred_col, group_col="job_id", label_col="is_start", level="group"
):
    """Compute pairwise ranking score."""
    scores_all = []
    for _, df_tmp in full_df.groupby(group_col):
        labels = df_tmp[label_col].values
        preds = df_tmp[pred_col].values
        idx1 = np.where(labels == 1)[0]
        idx0 = np.where(labels == 0)[0]

        # create a list of pred_pair i.e., pred_pairs
        # pred_pairs[:, 0] and pred_pairs[:, 1] has positive and negative labels respectively
        pred_pairs = np.stack(
            [
                np.repeat(preds[idx1], len(preds[idx0])),
                np.tile(preds[idx0], len(preds[idx1])),
            ],
            axis=1,
        )
        scores = np.zeros(len(pred_pairs))
        scores[pred_pairs[:, 0] > pred_pairs[:, 1]] = 1
        scores[pred_pairs[:, 0] == pred_pairs[:, 1]] = 0.5
        scores[pred_pairs[:, 0] < pred_pairs[:, 1]] = 0
        if len(pred_pairs) > 0:
            if level == "group":
                scores_all.append(np.mean(scores))
            elif level == "row":
                scores_all.extend(scores)
            else:
                raise Exception("Level is not implemented")
    return np.mean(scores_all)


def raw_pairwise_ranking_score(
    full_df,
    pred_col,
    group_col="job_id",
    label_col="is_start",
):
    """Compute raw pairwise ranking score."""

    return pairwise_ranking_score(full_df, pred_col, group_col, label_col, level="row")


def auc(full_df, pred_col, label_col="is_start"):
    """Compute AUC score."""
    if len(full_df[label_col].unique()) < 2:
        return None
    return roc_auc_score(full_df[label_col], full_df[pred_col])


def ndcg(full_df, pred_col, group_col="job_id", label_col="is_start"):
    """Compute normalized discounted cumulative gain."""
    scores = []
    for query in full_df[group_col].unique():
        query_df = full_df.loc[full_df[group_col] == query, :].sort_values(
            pred_col, ascending=False
        )
        scores.append(ndcg_score(query_df[label_col], query_df[pred_col]))
    return np.mean(scores)


def mae(full_df, pred_col, label_col="is_start"):
    """Compute mean absolute error."""
    return mean_absolute_error(full_df[label_col], full_df[pred_col])


def compute_lr_feature_weights(
    full_df: pd.DataFrame,
    features_selected: list,
    label_col: str,
    sample_weight_col: str = None,
    ranking_model=LogisticRegression(max_iter=10000),
    scale: bool = False,
    sort: bool = True,
):
    """
    Return a dataframe to show the LR feature weights

    Parameters
    ----------
    full_df : pd.DataFrame
        Dataframe with all the features and a predicted score column
    features_selected : list
        List of features (column names) to be evaluated
    label_col : str
        Column name of the binary label
    sample_weight_col : str
        Column name specifying the sample weights
    ranking_model:
        Sklearn logistic regression model class with its parameters
    scale : bool
        If True, the features are scaled to 0 mean and unit variance
    sort : boolean
        If True, the returned table is sorted by the absolute value of the weights

    Returns
    -------
        sklearn LR model, pd.DataFrame: Model and feature weightss
    """
    df_tmp = full_df.copy()
    if scale:
        for fea in features_selected:
            df_tmp[fea] = (df_tmp[fea] - df_tmp[fea].mean()) / df_tmp[fea].std(ddof=0)

    model = deepcopy(ranking_model)
    if sample_weight_col:
        model.fit(
            df_tmp[features_selected],
            df_tmp[label_col],
            sample_weight=df_tmp[sample_weight_col],
        )
    else:
        model.fit(df_tmp[features_selected], df_tmp[label_col])
    results_df = dict(
        zip(
            features_selected + ["intercept"],
            np.concatenate([model.coef_[0], model.intercept_]),
        )
    )
    if sort:
        return model, pd.DataFrame(
            sorted(results_df.items(), key=lambda x: np.abs(x[1]), reverse=True),
            columns=["features", "weights"],
        )

    return model, pd.DataFrame(results_df.items(), columns=["features", "weights"])


def logistic_train_func(train_set, features, train_label_col, sample_weight_col=None):
    model = LogisticRegression(max_iter=10000)
    if sample_weight_col:
        model.fit(
            train_set[features],
            train_set[train_label_col],
            sample_weight=train_set[sample_weight_col],
        )
    else:
        model.fit(
            train_set[features],
            train_set[train_label_col],
        )
    return model


def logistic_predict_func(model, test_features):
    return np.array(model.predict_proba(test_features)[:, 1])


def generate_oof_preds(
    df_train: pd.DataFrame,
    dfs_eval: list[pd.DataFrame],
    dfs_eval_names: list[str],
    features: list[str],
    train_label_col: str = "is_start",
    pred_col: str = "preds",
    test_jobs_folds: list[list[int]] = None,
    sample_weight_col: str = None,
    fix_dev_leak_cols: list[str] = ["developer_id"],
    train_func: Callable = logistic_train_func,
    pred_func: Callable = logistic_predict_func,
    features_to_zero: list[str] = None,
):
    """
    Generate the out of fold predictions (in-place) as a
    column in each of the test set listed in dfs_eval

    Parameters
    ----------

    df_train : pd.DataFrame
        Training dataframe
    dfs_eval : list
        List of testing dataframe (e.g. packet2start, search2start, etc.)
    dfs_eval_names : list
        List of strings specifying the names of the testing dataframes
    features: list
        List of feature column names
    train_label_col: str
        Label column name for training
    pred_col: str
        Column name to store the out of fold predictions
    test_jobs_folds: list
        List of testing folds, each fold contains a list of testing job_ids
    sample_weight_col: str
        Column name stating the sample weight of each row
    fix_dev_leak_cols: list[str]
        List of dev cols, training rows with these cols included in test sets are removed.
    train_func: Callable
        A function for training with same arguments as logistic_train_func
    pred_func: Callable
        A function for training with same arguments as logistic_predict_func
    features_to_zero: list[str]
        The features where a dummy value (0) is used when inference (usually the job features)
    """

    if test_jobs_folds is None:  # leave one job out CV
        test_jobs = set()
        for df_eval in dfs_eval:
            test_jobs = test_jobs.union(df_eval["job_id"].unique())
        # group all jobs which are not intersected with the training set together
        # no point doing LOOCV for them
        test_jobs_hold_out = test_jobs - set(df_train["job_id"].unique())
        test_jobs_intersect = test_jobs - test_jobs_hold_out
        test_jobs_folds = [[job_id] for job_id in test_jobs_intersect]
        if test_jobs_hold_out:
            test_jobs_folds.append(test_jobs_hold_out)

    for test_jobs in tqdm(test_jobs_folds, total=len(test_jobs_folds)):
        if fix_dev_leak_cols:
            if "developer_id" not in fix_dev_leak_cols:
                fix_dev_leak_cols.append("developer_id")
            test_sets = pd.concat(
                [
                    df_eval.loc[df_eval["job_id"].isin(test_jobs), fix_dev_leak_cols]
                    for df_eval in dfs_eval
                ]
            ).reset_index(drop=True)

            train_set = (
                df_train.loc[(~df_train["job_id"].isin(test_jobs))]
                .merge(test_sets, on=fix_dev_leak_cols, how="left", indicator=True)
                .query('_merge == "left_only"')
                .drop("_merge", axis=1)
                .reset_index(drop=True)
            )
        else:
            train_set = df_train.loc[(~df_train["job_id"].isin(test_jobs))].reset_index(
                drop=True
            )
        model = train_func(train_set, features, train_label_col, sample_weight_col)
        for df_eval, _name in zip(dfs_eval, dfs_eval_names):
            test_cond = df_eval["job_id"].isin(test_jobs)
            test_x = df_eval.loc[test_cond, features].reset_index(drop=True)
            if features_to_zero:
                for col in features_to_zero:
                    test_x[col] = 0
            if test_cond.sum():
                df_eval.loc[test_cond, pred_col] = pred_func(model, test_x)


def full_ranking_evaluation(
    df_train: pd.DataFrame,
    features: list[str],
    dfs_eval: list[pd.DataFrame],
    dfs_eval_names: list[str],
    train_label_col: str = "is_start",
    eval_labels_cols: list[str] = None,
    pred_col: str = "preds",
    test_jobs_folds: list[list[int]] = None,
    sample_weight_col: str = None,
    metrics: list[str] = ["pairwise_ranking_score", "raw_pairwise_ranking_score"],
    exclude_uni_label_jobs: bool = True,
    win_loss_ns: list[int] = [1, 3, 5, 10],
    include_positive_rank_summary: bool = True,
    fix_dev_leak_cols: list[str] = ["developer_id"],
    features_to_zero: list[str] = None,
    train_func: Callable = logistic_train_func,
    pred_func: Callable = logistic_predict_func,
    verbose: bool = True,
):
    """
    Evaluate a model on a list of test sets using a list of metrics

    Parameters
    ----------
    df_train: pd.DataFrame
        Training dataframe
    features: list[str]
        list of feature column names
    dfs_eval: list[pd.DataFrame]
        List of testing dataframe (e.g. packet2start, search2start, etc.)
    dfs_eval_names: list
        List of strings stating the name of the test sets (following order of dfs_eval)
    train_label_col : str
        Label column name for training
    eval_labels_cols: list
        List of strings stating the name of the test sets (following order of dfs_eval)
    pred_col: str
        Column name to store the out of fold predictions
    test_jobs_folds: list
        List of testing folds, each fold contains a list of testing job_ids
    sample_weight_col: str
        Column name stating the sample weight of each row
    metrics: list
        List of function names for computing various evaluation metrics
    exclude_uni_label_jobs: bool
        If true, all jobs in the test sets with uniform labels are excluded
    win_loss_ns: list
        List of ints for calculating top N win loss
    win_loss_interview_weights: list
        List of weightages for calculating interview_winloss
    include_positive_rank_summary: bool
        If True, include the positive rank summaries as metrics
    fix_dev_leak_cols: list
        List of dev cols, training rows with these cols included in test sets are removed.
    features_to_zero: list
        The features where a dummy value (0) is used when inference (usually the job features)
    train_func: Callable
        A function for training with same arguments as logistic_train_func
    pred_func: Callable
        A function for training with same arguments as logistic_predict_func
    verbose: bool
        If True, display the results as a table
    """
    if exclude_uni_label_jobs:
        if eval_labels_cols is None:
            eval_labels_cols = ["is_start"] * len(dfs_eval)
        dfs_eval = [
            df_eval.loc[
                df_eval.groupby("job_id")[label_col].transform("nunique") > 1
            ].reset_index(drop=True)
            for df_eval, label_col in zip(dfs_eval, eval_labels_cols)
        ]

    generate_oof_preds(
        df_train=df_train,
        dfs_eval=dfs_eval,
        dfs_eval_names=dfs_eval_names,
        features=features,
        train_label_col=train_label_col,
        pred_col=pred_col,
        test_jobs_folds=test_jobs_folds,
        sample_weight_col=sample_weight_col,
        fix_dev_leak_cols=fix_dev_leak_cols,
        train_func=train_func,
        pred_func=pred_func,
        features_to_zero=features_to_zero,
    )
    metric_dicts = []
    for df_eval, name, label_col in zip(dfs_eval, dfs_eval_names, eval_labels_cols):
        metric_dict = {
            metric: globals()[metric](df_eval, pred_col, label_col=label_col)
            for metric in metrics
        }

        # win loss
        df_eval["rank"] = df_eval.groupby("job_id")[pred_col].rank(ascending=False)
        for win_loss_n in win_loss_ns:
            metric_dict[f"win_loss_top_{win_loss_n}"] = (
                df_eval.loc[df_eval["rank"] <= win_loss_n]
                .groupby("job_id")[label_col]
                .mean()
                .mean()
            )

        # interview win loss
        if label_col == "is_start":
            df_eval_interview = df_eval.loc[
                df_eval.groupby("job_id")["is_si"].transform("sum")
                > df_eval.groupby("job_id")["is_start"].transform("sum")
            ].reset_index(drop=True)
            job_max_start_rank = df_eval_interview["job_id"].map(
                df_eval_interview.loc[df_eval_interview["is_start"] == 1]
                .groupby("job_id")["rank"]
                .max()
            )
            df_eval_interview["loss_interview"] = (
                (df_eval_interview["is_start"] == 0)
                & (df_eval_interview["is_si"] == 1)
                & (df_eval_interview["rank"] < job_max_start_rank)
            )
            wins = df_eval_interview.groupby("job_id")["is_start"].sum().sort_index()
            losses = (
                df_eval_interview.groupby("job_id")["loss_interview"].sum().sort_index()
            )
            metric_dict["interview_win_portion"] = (wins / (wins + losses)).mean()
        else:
            metric_dict["interview_win_portion"] = np.nan
        # positive rank summary
        if include_positive_rank_summary:
            positive_cond = df_eval[label_col] == 1

            metric_dict.update(
                {
                    "avg_started_ranks": df_eval.loc[positive_cond, "rank"].mean(),
                    "10pct_started_ranks": df_eval.loc[positive_cond, "rank"].quantile(
                        0.1
                    ),
                    "25pct_started_ranks": df_eval.loc[positive_cond, "rank"].quantile(
                        0.25
                    ),
                    "50pct_started_ranks": df_eval.loc[positive_cond, "rank"].quantile(
                        0.5
                    ),
                    "75pct_started_ranks": df_eval.loc[positive_cond, "rank"].quantile(
                        0.75
                    ),
                    "90pct_started_ranks": df_eval.loc[positive_cond, "rank"].quantile(
                        0.9
                    ),
                    "95pct_started_ranks": df_eval.loc[positive_cond, "rank"].quantile(
                        0.95
                    ),
                    "99pct_started_ranks": df_eval.loc[positive_cond, "rank"].quantile(
                        0.99
                    ),
                    "max_started_ranks": df_eval.loc[positive_cond, "rank"].max(),
                }
            )
        metric_dicts.append(metric_dict)
    if verbose:
        display(
            pd.concat(
                [
                    pd.DataFrame({"test_sets": dfs_eval_names}),
                    pd.DataFrame(metric_dicts),
                ],
                axis=1,
            )
        )
    return metric_dicts, dfs_eval
