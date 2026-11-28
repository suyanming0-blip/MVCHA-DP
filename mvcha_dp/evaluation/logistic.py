import numpy as np
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression


class SplitClassInvalid(ValueError):
    pass


def fit_final_lr(latent, labels, train_ids):
    x = np.asarray(latent, dtype=np.float32).copy()
    y = np.asarray(labels, dtype=np.int64).reshape(-1).copy()
    if len(x) != len(y) or len(train_ids) != len(y) or (not np.isfinite(x).all()) or (not np.isin(y, [0, 1]).all()):
        raise ValueError('Invalid aligned finite train-only latent input')
    (lab, counts) = np.unique(y, return_counts=True)
    if len(lab) != 2:
        raise SplitClassInvalid('SPLIT_CLASS_INVALID: final LR needs two real training classes')
    info = dict(
        train_ids=list(map(int, train_ids)),
        input_shape=list(x.shape),
        before=np.bincount(y, minlength=2).tolist(),
        gate_apply=int(y.sum()) <= int(len(y) * 0.4),
        smotetomek_calls=0,
        lr_fit_calls=0,
        k_neighbors=None,
        tomek_executed=False,
        reason=None
    )
    (rx, ry) = (x, y)
    if info['gate_apply']:
        minority = int(counts.min())
        if minority == 1:
            info['reason'] = 'INSUFFICIENT_MINORITY_FOR_INTERPOLATION'
        else:
            k = min(5, minority - 1)
            sampler = SMOTETomek() if minority >= 6 else SMOTETomek(smote=SMOTE(k_neighbors=k))
            (rx, ry) = sampler.fit_resample(x, y)
            info.update(smotetomek_calls=1, k_neighbors=k, tomek_executed=True)
            if len(ry) == 0 or len(np.unique(ry)) != 2:
                (rx, ry) = (x, y)
                info['reason'] = 'INVALID_RESAMPLED_CLASS_STRUCTURE_FALLBACK_REAL'
            else:
                state = np.random.get_state()
                np.random.shuffle(rx)
                np.random.set_state(state)
                np.random.shuffle(ry)
    else:
        info['reason'] = 'BASELINE_POSITIVE_RATIO_GATE_SKIP'
    clf = LogisticRegression(class_weight={0: 1, 1: 2}, solver='liblinear')
    clf.fit(rx, ry)
    info.update(
        lr_fit_calls=1,
        output_shape=list(rx.shape),
        after=np.bincount(ry, minlength=2).tolist(),
        parameters=clf.get_params()
    )
    return (clf, info)
