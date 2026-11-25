import numpy as np
from mvcha_dp.protocol.checks import require


def positive_probability(estimator, latent):
    classes = np.asarray(estimator.classes_)
    require(
        classes.ndim == 1 and len(classes) == 2 and (set(classes.tolist()) == {0, 1}),
        'Expected estimator classes {0, 1}'
    )
    positive = np.flatnonzero(classes == 1)
    require(len(positive) == 1, 'Missing or duplicate positive class')
    probability = np.asarray(estimator.predict_proba(latent))
    require(probability.ndim == 2 and probability.shape == (len(latent), len(classes)), 'Invalid probability shape')
    require(
        np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all(),
        'Invalid probability values'
    )
    return probability[:, int(positive[0])]


def evaluate(estimator, latent, labels):
    from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef
    y = np.asarray(labels)
    require(y.shape == (len(latent),) and len(y) > 0 and np.isin(y, [0, 1]).all(), 'Invalid evaluation labels')
    probability = positive_probability(estimator, latent)
    prediction = np.asarray(estimator.predict(latent))
    require(prediction.shape == y.shape and np.isin(prediction, [0, 1]).all(), 'Invalid estimator predictions')
    return {
        'auc': float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else None,
        'f1': float(f1_score(y, prediction, zero_division=0)),
        'mcc': float(matthews_corrcoef(y, prediction)),
        'evaluation_count': len(y),
        'positive_count': int(y.sum()),
        'predicted_positive_count': int(prediction.sum())
    }
