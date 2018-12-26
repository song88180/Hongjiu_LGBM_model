__all__ = ('minimization_metric', 'evaluation_metrics')


import numpy
import pandas
import scipy.stats
import sklearn.metrics


def minimization_metric(truth, prediction):
    return -scipy.stats.kendalltau(truth, prediction)[0]


def evaluation_metrics(truth, prediction):
    prediction = prediction.clip(min=numpy.finfo('f4').eps)
    pearson = scipy.stats.pearsonr(truth, prediction)[0]
    logpearson = scipy.stats.pearsonr(numpy.log(truth),
                                      numpy.log(prediction))[0]
    spearman = scipy.stats.spearmanr(truth, prediction)[0]
    kendall = scipy.stats.kendalltau(truth, prediction)[0]
    mse = sklearn.metrics.mean_squared_error(truth, prediction)
    logmse = sklearn.metrics.mean_squared_error(numpy.log(truth),
                                                numpy.log(prediction))
    auc_0_5 = sklearn.metrics.roc_auc_score(truth > 0.5, prediction)
    auc_1 = sklearn.metrics.roc_auc_score(truth > 1, prediction)
    return pandas.Series(
        [pearson, logpearson, spearman, kendall, mse, logmse, auc_0_5, auc_1],
        index=['Pearson', 'Log-pearson', 'Spearman', 'Kendall', 'MSE',
               'Log-MSE', 'AUC 0.5', 'AUC 1'],
    )
