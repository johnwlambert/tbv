"""Precision/recall computation utilities."""

import os
from enum import Enum, auto
from pathlib import Path
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn.metrics


_PathLike = Union[str, "os.PathLike[str]"]


# Number of recall points to sample uniformly in [0, 1]. Default to 101 recall samples.
N_REC_SAMPLES = 101


class InterpType(Enum):
    ALL = auto()


class PrecisionRecallMeter:
    """Compute moving averages of precision/recall as streaming data is provided from batches."""

    def __init__(self) -> None:
        """ """
        self.all_y_true = np.zeros((0, 1))
        self.all_y_hat = np.zeros((0, 1))
        self.all_y_hat_probs = np.zeros((0, 1))

    def update(self, y_true: np.ndarray, y_hat: np.ndarray, y_hat_probs: np.ndarray) -> None:
        """ """
        y_true = y_true.reshape(-1, 1)
        y_hat = y_hat.reshape(-1, 1)
        y_hat_probs = y_hat_probs.reshape(-1, 1)

        self.all_y_true = np.vstack([self.all_y_true, y_true])
        self.all_y_hat = np.vstack([self.all_y_hat, y_hat])
        self.all_y_hat_probs = np.vstack([self.all_y_hat_probs, y_hat_probs])

    def get_metrics(self) -> Tuple[float, float, float]:
        """Compute average precision from currently accumulated data."""
        ap = calc_ap(y_hat=self.all_y_hat, y_true=self.all_y_true, y_hat_probs=self.all_y_hat_probs)

        plot_precision_recall_curve_sklearn(
            y_hat=self.all_y_hat, y_true=self.all_y_true, y_hat_probs=self.all_y_hat_probs, save_plot=True
        )

        return ap

    def save_pr_curve(self, save_fpath: str) -> None:
        """Compute the P/R curve and save it to disk."""

        ap = calc_ap(y_hat=self.all_y_hat, y_true=self.all_y_true, y_hat_probs=self.all_y_hat_probs)

        plot_precision_recall_curve_sklearn(
            y_hat=self.all_y_hat,
            y_true=self.all_y_true,
            y_hat_probs=self.all_y_hat_probs,
            save_plot=True,
            save_fpath=save_fpath,
        )


def assign_tp_fp_fn_tn(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    """ """
    is_TP = np.logical_and(y_true == y_pred, y_pred == 1)
    is_FP = np.logical_and(y_true != y_pred, y_pred == 1)
    is_FN = np.logical_and(y_true != y_pred, y_pred == 0)
    is_TN = np.logical_and(y_true == y_pred, y_pred == 0)

    return is_TP, is_FP, is_FN, is_TN


def compute_tp_fp_fn_tn_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    """ """

    TP = np.logical_and(y_true == y_pred, y_pred == 1).sum()
    FP = np.logical_and(y_true != y_pred, y_pred == 1).sum()

    FN = np.logical_and(y_true != y_pred, y_pred == 0).sum()
    TN = np.logical_and(y_true == y_pred, y_pred == 0).sum()

    return TP, FP, FN, TN


def compute_precision_recall(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    """Compute precision and recall at a fixed operating point.

    In confusion matrix, `actual` are along rows, `predicted` are columns:

              Predicted
          \\ P  N
    Actual P TP FN
           N FP TN

    Args:
        y_true: array of shape (K,) representing ground truth classes. We define 1 as the target class (positive)
        y_pred: array of shape (K,) representing predicted classes.

    Returns:
        prec: precision.
        rec: recall.
        mAcc: mean accuracy.
    """
    EPS = 1e-7

    TP, FP, FN, TN = compute_tp_fp_fn_tn_counts(y_true, y_pred)

    # form a confusion matrix
    C = np.zeros((2, 2))
    C[0, 0] = TP
    C[0, 1] = FN

    C[1, 0] = FP
    C[1, 1] = TN

    # Normalize the confusion matrix
    C[0] /= C[0].sum() + EPS
    C[1] /= C[1].sum() + EPS

    mAcc = np.mean(np.diag(C))

    prec = TP / (TP + FP + EPS)
    rec = TP / (TP + FN + EPS)

    if (TP + FN) == 0:
        # there were no positive GT elements
        print("Recall undefined...")
        # raise Warning("Recall undefined...")

    # import sklearn.metrics
    # prec, rec, _, support = sklearn.metrics.precision_recall_fscore_support(y_true, y_pred)
    return prec, rec, mAcc


def calc_ap(y_hat: np.ndarray, y_true: np.ndarray, y_hat_probs: np.ndarray) -> float:
    """Compute precision and recall, interpolated over n fixed recall points.

    Args:
        y_hat: array of length (K,) representing model predictions.
        y_true: array of length (K,) representing ground truth classes.
        y_hat_probs: array of length (K,) representing confidence/probability associated with each prediction in y_hat.

    Returns:
        avg_precision: Average precision.
    """
    # compute probability estimates of the positive class
    pos_prob = y_hat_probs.copy()
    pos_prob[y_hat != 1] = 1 - pos_prob[y_hat != 1]

    ap = sklearn.metrics.average_precision_score(y_true, y_score=pos_prob)
    return ap


def plot(rec_interp: np.ndarray, prec_interp: np.ndarray, figs_fpath: Path) -> Path:
    """Plot and save the precision recall curve.

    Args:
        rec_interp: Interpolated recall data of shape (N,).
        prec_interp: Interpolated precision data of shape (N,).
        cls_name: Class name.
        figs_fpath: Path to the folder which will contain the output figures.

    Returns:
        dst_fpath: Plot file path.
    """
    plt.plot(rec_interp, prec_interp)
    plt.title("PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    dst_fpath = Path(f"{figs_fpath}/pr_curve.png")
    plt.savefig(dst_fpath)
    plt.close()
    return dst_fpath


def interp_prec(prec: np.ndarray, method: InterpType = InterpType.ALL) -> np.ndarray:
    """Interpolate the precision over all recall levels. See equation 2 in
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.167.6629&rep=rep1&type=pdf
    for more information.

    Args:
        prec: Increasing precision at all recall levels (N,).
        method: Accumulation method.

    Returns:
        prec_interp: Monotonically increasing precision at all recall levels (N,).
            Could be considered "interpolated"
    """
    if method == InterpType.ALL:
        prec_interp = np.maximum.accumulate(prec)
    else:
        raise NotImplementedError("This interpolation method is not implemented!")
    return prec_interp


def test_interp_prec() -> None:
    """Make sure we can force the precision to be monotonically increasing."""

    prec = np.array([0.67, 0.5, 1, 1])
    prec_interp = pr_utils.interp_prec(prec)

    # should be monotonically increasing now.
    expected_prec_interp = np.array([0.67, 0.67, 1.0, 1.0])

    assert np.allclose(prec_interp, expected_prec_interp)


def plot_precision_recall_curve_sklearn(
    y_hat: np.ndarray,
    y_true: np.ndarray,
    y_hat_probs: np.ndarray,
    save_plot: bool = True,
    save_fpath: str = "2022_02_02_precision_recall.pdf",
) -> None:
    """Compute a P/R curve using the sklearn library.

    See sklearn implementation here:
        https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/metrics/_ranking.py#L786

    Args:
        y_hat: array of length (K,) representing model predictions.
        y_true: array of length (K,) representing ground truth classes.
        y_hat_probs: array of length (K,) representing confidence/probability associated with each prediction in y_hat.
        save_plot: whether to save P/R curve plot to disk.
        save_fpath: file path to save figure to.

    Returns:
        prec: array of shape (K,) representing monotonically increasting precision values such that element i is the
            precision of predictions with score >= thresholds[i] and the last element is 1.
            (TODO: issue bug report on the documentation for this)
            We force monotonicity.
        recall: array of shape (K,) representing decreasing recall values such that element i is the recall of predictions
            with score >= thresholds[i] and the last element is 0.
        thresholds: array of shape (K-1,) representing confidence thresholds for each precision and recall value.
            see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html
    """
    y_true_list = []
    probas_pred = []
    for (y_hat_, y_true_, y_hat_prob_) in zip(y_hat, y_true, y_hat_probs):

        y_true_list.append(y_true_)

        if y_hat_ == 1:
            pos_prob = y_hat_prob_
        else:
            pos_prob = 1 - y_hat_prob_

        probas_pred.append(pos_prob)

    prec, recall, thresholds = sklearn.metrics.precision_recall_curve(
        y_true=y_true_list, probas_pred=probas_pred, pos_label=1
    )
    prec = interp_prec(prec)

    if save_plot:
        plt.style.use("ggplot")
        # sns.set_style({"font.family": "Times New Roman"})

        palette = np.array(sns.color_palette("hls", 5))
        color = palette[0]

        plt.plot(recall, prec, color=color)

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.tight_layout()
        plt.savefig(save_fpath, dpi=500)
        plt.close("all")
        # plt.show()

    return prec, recall, thresholds
