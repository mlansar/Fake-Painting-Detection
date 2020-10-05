'''Examples of organizer-provided metrics.
You can just replace this code by your own.
Make sure to indicate the name of the function that you chose as metric function
in the file metric.txt. E.g. mse_metric, because this file may contain more 
than one function, hence you must specify the name of the function that is your metric.'''

import numpy as np
import scipy as sp
from sklearn import metrics


def mse_metric(solution, prediction):
    '''Mean-square error.
    Works even if the target matrix has more than one column'''
    mse = np.mean((solution-prediction)**2)
    return np.mean(mse)

def AUC_metric(solution, prediction):
	fpr, tpr, thresholds = metrics.roc_curve(solution, prediction, pos_label=1)
	return metrics.auc(fpr, tpr)

