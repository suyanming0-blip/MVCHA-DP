from mvcha_dp.protocol.checks import require
from mvcha_dp.protocol.selection import Validation


def observe(scope, epoch, *, label_role, read_labels, probability, prediction):
    scope.validate()
    require(label_role == scope.metric_source, 'Invalid validation label role')
    import numpy as np
    from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef
    y = np.asarray(read_labels())
    proba = np.asarray(probability)
    pred = np.asarray(prediction)
    require(
        y.ndim == 1 and len(y) > 0 and (proba.shape == y.shape) and (pred.shape == y.shape),
        'Validation vector shapes differ'
    )
    require(
        np.isin(y, [0, 1]).all() and len(np.unique(y)) == 2,
        'Validation AUC requires two binary classes'
    )
    require(
        np.isin(pred, [0, 1]).all() and np.isfinite(proba).all() and ((proba >= 0) & (proba <= 1)).all(),
        'Invalid validation predictions'
    )
    return Validation.create(
        scope,
        epoch,
        auc=roc_auc_score(y, proba),
        f1=f1_score(y, pred, zero_division=0),
        mcc=matthews_corrcoef(y, pred)
    )
