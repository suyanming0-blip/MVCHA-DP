# Validation readout.

from contextlib import contextmanager
import random
import numpy as np
import torch
from mvcha_dp.protocol.checks import require
from .metrics import positive_probability
from .validation import observe


@contextmanager
def isolated_readout_rng(seed=321):
    require(type(seed) is int and 0 <= seed < 2**32, 'Invalid readout seed')
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.get_rng_state().clone()
    cuda_initialized = torch.cuda.is_initialized()
    cuda_states = torch.cuda.get_rng_state_all() if cuda_initialized else None
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.random.default_generator.manual_seed(seed)
        if cuda_initialized:
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_initialized:
            torch.cuda.set_rng_state_all(cuda_states)


def extract_latent(encoder, forward):
    modes = [(module, module.training) for module in encoder.modules()]
    try:
        encoder.eval()
        with torch.no_grad():
            return forward().detach().cpu().numpy()
    finally:
        for (module, mode) in modes:
            module.training = mode


def lr_validation(scope, epoch, *, train_role, validation_role, train_ids, validation_ids, read_train_labels, read_validation_labels, latent_provider, h_sync_epoch):
    scope.validate()
    expected_train = 'inner_train' if scope.scenario == 'WPDP' else 'source_inner_train'
    require(
        train_role == expected_train and validation_role == scope.metric_source,
        'Invalid readout label roles'
    )
    for ids in (train_ids, validation_ids):
        require(
            len(ids) > 0 and all((type(x) is int and x >= 0 for x in ids)) and (len(ids) == len(set(ids))),
            'Invalid or duplicate split IDs'
        )
    require(not set(train_ids).intersection(validation_ids), 'Training/validation IDs overlap')
    require(type(h_sync_epoch) is int and 0 <= h_sync_epoch <= epoch, 'Invalid active H sync epoch')
    with isolated_readout_rng(scope.seed):
        (train_x, val_x) = latent_provider()
        (train_x, val_x) = (np.asarray(train_x), np.asarray(val_x))
        require(
            train_x.ndim == val_x.ndim == 2 and train_x.shape[1] == val_x.shape[1] and (len(train_x) == len(train_ids)) and (len(val_x) == len(validation_ids)) and np.isfinite(train_x).all() and np.isfinite(val_x).all(),
            'Invalid split latent'
        )
        y = np.asarray(read_train_labels())
        from mvcha_dp.evaluation.logistic import fit_final_lr
        (clf, info) = fit_final_lr(train_x, y, train_ids)
        proba = positive_probability(clf, val_x)
        pred = clf.predict(val_x)
        row = observe(
            scope,
            epoch,
            label_role=validation_role,
            read_labels=read_validation_labels,
            probability=proba,
            prediction=pred
        )
        metadata = dict(
            resampling_applied=bool(info['smotetomek_calls'] and info['reason'] != 'INVALID_RESAMPLED_CLASS_STRUCTURE_FALLBACK_REAL'),
            resampling_attempted=bool(info['smotetomek_calls']),
            original_train_count=info['input_shape'][0],
            original_class_counts=info['before'],
            resampled_train_count=info['output_shape'][0],
            resampled_class_counts=info['after'],
            SMOTE_k=info['k_neighbors'],
            LR_classes=clf.classes_.tolist(),
            H_sync_epoch=h_sync_epoch,
            current_epoch=epoch,
            readout_seed=scope.seed,
            resampling_reason=info['reason'],
            frozen_readout_info=info
        )
    return (row, metadata, clf)


def refit_selected(expected, **kwargs):
    (actual, metadata, clf) = lr_validation(expected.scope, expected.epoch, **kwargs)
    return (actual, metadata, clf)
