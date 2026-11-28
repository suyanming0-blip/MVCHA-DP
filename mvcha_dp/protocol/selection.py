# Checkpoint selection and early stopping.

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from .checks import require
AUC_TOLERANCE = Decimal('0.002')
MAX_EPOCHS = 200
MIN_COMPLETED_EPOCHS = 50
PATIENCE = 30
FAMILIES = ('Promise', 'SOUND', 'Line-level-defect-prediction')


def metric(value, low, high):
    require(not isinstance(value, bool), 'Boolean is not a metric')
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError('Invalid metric value')
    require(
        result.is_finite() and Decimal(str(low)) <= result <= Decimal(str(high)),
        'Invalid validation metric'
    )
    return result


@dataclass(frozen=True)
class Scope:
    scenario: str
    family: str
    task_id: str
    metric_source: str
    seed: int

    def validate(self):
        require(self.scenario in ('WPDP', 'CVDP', 'CPDP') and self.family in FAMILIES, 'Invalid scenario/family')
        require(
            self.metric_source == ('inner_validation' if self.scenario == 'WPDP' else 'source_validation'),
            'Invalid selector label role'
        )
        require(type(self.seed) is int and self.seed in range(321, 341), 'Seed must be in 321..340')


@dataclass(frozen=True)
class Validation:
    scope: Scope
    epoch: int
    auc: Decimal
    f1: Decimal
    mcc: Decimal

    @classmethod
    def create(cls, scope, epoch, *, auc, f1, mcc):
        scope.validate()
        require(type(epoch) is int and 0 <= epoch < MAX_EPOCHS, 'Epoch must be zero-based 0..199')
        return cls(scope, epoch, metric(auc, 0, 1), metric(f1, 0, 1), metric(mcc, -1, 1))

    @property
    def secondary_score(self):
        return (self.f1 + self.mcc) / 2

    def as_dict(self):
        return dict(
            epoch=self.epoch,
            completed_epochs=self.epoch + 1,
            LR_AUC=str(self.auc),
            LR_F1=str(self.f1),
            LR_MCC=str(self.mcc),
            LR_SECONDARY=str(self.secondary_score)
        )


def validate_history(history):
    require(bool(history), 'Empty validation history')
    scope = history[0].scope
    scope.validate()
    for (epoch, row) in enumerate(history):
        require(
            isinstance(row, Validation) and row.scope == scope and (row.epoch == epoch),
            'Invalid validation history sequence or scope'
        )
        metric(row.auc, 0, 1)
        metric(row.f1, 0, 1)
        metric(row.mcc, -1, 1)
        require(
            isinstance(row.auc, Decimal) and isinstance(row.f1, Decimal) and isinstance(row.mcc, Decimal),
            'Expected Decimal validation metrics'
        )
    require(len(history) <= MAX_EPOCHS, 'History exceeds max_epochs=200')


def selected_checkpoint(history):
    validate_history(history)
    peak = max((row.auc for row in history))
    eligible = [row for row in history if row.auc >= peak - AUC_TOLERANCE]
    return max(eligible, key=lambda row: (row.secondary_score, row.auc, row.mcc, row.f1, -row.epoch))


class EarlyStopping:

    def __init__(self, scope):
        scope.validate()
        self.scope = scope
        self.history = []
        self.selected_epoch = None
        self.stale_epochs = 0
        self.stopped = False
        self.last_selected_checkpoint_change_epoch = None

    def append(self, row):
        require(not self.stopped, 'Training has already stopped')
        require(row.scope == self.scope and row.epoch == len(self.history), 'Wrong trial or missing/repeated epoch')
        candidate = self.history + [row]
        winner = selected_checkpoint(candidate)
        changed = winner.epoch != self.selected_epoch
        self.stale_epochs = 0 if changed else self.stale_epochs + 1
        previous = self.selected_epoch
        if changed:
            self.last_selected_checkpoint_change_epoch = row.epoch
        self.selected_epoch = winner.epoch
        self.history = candidate
        early = row.epoch + 1 >= MIN_COMPLETED_EPOCHS and self.stale_epochs >= PATIENCE
        exhausted = row.epoch + 1 == MAX_EPOCHS
        self.stopped = early or exhausted
        return {
            'epoch': row.epoch,
            'completed_epochs': row.epoch + 1,
            'auc_peak': str(max((x.auc for x in candidate))),
            'selected_epoch': winner.epoch,
            'previous_selected_epoch': previous,
            'selected_changed': changed,
            'historical_reselection': changed and winner.epoch < row.epoch,
            'stale_epochs': self.stale_epochs,
            'stop': self.stopped,
            'last_selected_checkpoint_change_epoch': self.last_selected_checkpoint_change_epoch,
            'stop_epoch': row.epoch if self.stopped else None,
            'stop_reason': 'patience' if early else 'max_epochs' if exhausted else None
        }
