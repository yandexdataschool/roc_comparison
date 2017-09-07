import argparse
import pandas as pd
import numpy as np
import scipy.stats
import json
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

# AUC comparison adapted from
# https://github.com/Netflix/vmaf/
def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float)
    ty = np.empty([k, n], dtype=np.float)
    tz = np.empty([k, m + n], dtype=np.float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    """Computes log(10) of p-values.
    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:
       log10(pvalue)
    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)


def compute_ground_truth_statistics(ground_truth):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    return order, label_1_count


def delong_roc_variance(ground_truth, predictions):
    """
    Computes ROC AUC variance for a single set of predictions
    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov


def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """
    Computes log(p-value) for hypothesis that two ROC AUCs are different
    Args:
       ground_truth: np.array of 0 and 1
       predictions_one: predictions of the first model,
          np.array of floats of the probability of being class 1
       predictions_two: predictions of the second model,
          np.array of floats of the probability of being class 1
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    print(aucs, delongcov)
    return calc_pvalue(aucs, delongcov)


def stratified_sampling_mask(array, sample_fraction):
    # WARNING(kazeevn)
    # selection size might be off from sample_fraction due to
    # rounding error accumulation
    # also doesn't shuffle
    # doen't scale well with the number of classes
    selected = np.zeros(shape=array.shape, dtype=np.bool)
    for value, count in zip(*np.unique(array, return_counts=True)):
        indices = (array == value).nonzero()[0]
        this_selection = np.random.choice(count, size=int(count*sample_fraction), replace=False)
        selected[indices[this_selection]] = True
    return selected


# Given the main body is a one-time script, lots of things are hardcoded
# If you need flexibility, please use the functions above
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--one-class-results", type=str,
                        help="tsv with first column being the 1 vs all"
                        " predictions", required=True)
    parser.add_argument("--multiclass-baselines", type=str,
                        help="csv with header", required=True)
    parser.add_argument('--ttest-method', choices=['one-tailed',
                                                   'two-tailed'], default='one-tailed')
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--tabulate-by-sample-size", action="store_true")
    parser.add_argument("--by-sample-size-plot", type=str)
    parser.add_argument("--classes-count", type=int, required=True)
    parser.add_argument("--baselines-count", type=int, required=True)
    parser.add_argument("--target-class", type=int, required=True)
    args = parser.parse_args()
    np.set_printoptions(precision=20)
    one_class_predictions = pd.read_csv(args.one_class_results,
                                        sep='\t', header=None,
                                        names=["predictions", "true_label"])
    baseline_columns = np.arange(args.baselines_count)*args.classes_count + args.target_class
    baselines = pd.read_csv(args.multiclass_baselines, usecols=baseline_columns)

    if args.tabulate_by_sample_size:
        trials = 15
        results = {}
        for baseline_name, baseline_predictions in baselines.iteritems():
            sample_sizes = []
            pvalues = []
            pvalue_stds = []
            for sample_fraction in np.logspace(start=-4, stop=0, endpoint=True, num=10):
                sample_pvalues = []
                for trial in range(trials if sample_fraction < 1 else 1):
                    sample_mask = stratified_sampling_mask(
                        one_class_predictions["true_label"].values, sample_fraction)
                    sample_pvalues.append(delong_roc_test(
                        one_class_predictions["true_label"].values[sample_mask],
                        one_class_predictions["predictions"].values[sample_mask],
                        baseline_predictions.values[sample_mask]))
                sample_sizes.append(int(sample_mask.sum()))
                pvalues.append(float(np.mean(sample_pvalues)))
                pvalue_stds.append(float(np.std(sample_pvalues)))
            results[baseline_name] = {
                "sample_sizes": sample_sizes,
                "pvalues": pvalues,
                "std(pvalues)": pvalue_stds}
        if args.by_sample_size_plot:
            fig, ax = plt.subplots()
            for baseline_name, result in results.items():
                ax.plot(result["sample_sizes"], result["pvalues"], label=baseline_name)
            ax.legend(loc=3)
            ax.set_xscale("log")
            ax.set_xlabel("Sample size")
            ax.set_ylabel("log10(p-value)")
            ax.set_ylim((-20, 1))
            loc = plticker.MultipleLocator(base=1)
            ax.yaxis.set_minor_locator(loc)
            major_loc = plticker.MultipleLocator(base=5)
            ax.yaxis.set_major_locator(major_loc)
            ax.grid()
            with open(args.by_sample_size_plot, "wb") as plot_io:
                fig.savefig(plot_io, bbox="tight", filetype="pdf")
    else:
        result = {}
        for baseline_name, baseline_predictions in baselines.iteritems():
            result[baseline_name] = delong_roc_test(
                one_class_predictions["true_label"].values,
                one_class_predictions["predictions"].values,
                baseline_predictions.values)
    with open(args.output, "w") as out_io:
        json.dump(results, out_io)

if __name__ == "__main__":
    main()
