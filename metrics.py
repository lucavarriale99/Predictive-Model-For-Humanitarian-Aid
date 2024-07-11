import numpy as np
from typing import Union

def spike_precision(y_pred, y_true, alpha, beta):

    """
    Returns the percentage of spikes that were predicted correctly under beta tolerance
    """    
    spikes_true = np.array(y_true[y_true>alpha]).reshape((-1,))
    spikes_pred = np.array(y_pred[y_true>alpha]).reshape((-1,))
    if len(spikes_pred)==0:
        return 1
    return sum(np.abs(spikes_pred-spikes_true)<beta)/len(spikes_true)

def spike_recall(y_pred, y_true, alpha, beta):

    """
    Returns the recall in spike prediction
    """

    y_true = np.array(y_true).reshape((-1,))
    y_pred = np.array(y_pred).reshape((-1,))

    spikes_true = y_true[y_pred>alpha]
    spikes_pred = y_pred[y_pred>alpha]
    if len(spikes_true)==0:
        return 1
    return sum(np.abs(spikes_pred-spikes_true)<beta)/len(spikes_true)

