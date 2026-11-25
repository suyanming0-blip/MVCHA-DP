import unittest

from mvcha_dp.protocol.selection import Scope, Validation, EarlyStopping, selected_checkpoint
from mvcha_dp.data.package import NativeLabelAccess


class ProtocolTests(unittest.TestCase):
    def scope(self):
        return Scope('MVCHA_DP_V2_2_REVIEW', 'WPDP', 'Promise', 'test', 'inner_validation')

    def test_auc_window_retains_secondary_winner(self):
        s = self.scope()
        history = [Validation.create(s, 0, auc=.8, f1=.8, mcc=.7),
                   Validation.create(s, 1, auc=.802, f1=.5, mcc=.4)]
        self.assertEqual(selected_checkpoint(history).epoch, 0)
        history.append(Validation.create(s, 2, auc=.8021, f1=.5, mcc=.4))
        self.assertEqual(selected_checkpoint(history).epoch, 2)

    def test_minimum_completed_epochs(self):
        s = self.scope()
        controller = EarlyStopping(s)
        for i in range(50):
            result = controller.append(Validation.create(s, i, auc=.8, f1=.6, mcc=.4))
            self.assertEqual(result['stop'], i == 49)
        self.assertEqual(result['selected_epoch'], 0)
        with self.assertRaises(ValueError):
            controller.append(Validation.create(s, 50, auc=.8, f1=.6, mcc=.4))

    def test_invalid_metric_and_labels(self):
        with self.assertRaises(ValueError):
            Validation.create(self.scope(), 0, auc=float('nan'), f1=.6, mcc=.4)
        labels = NativeLabelAccess(None, 'target')
        labels.begin_training()
        with self.assertRaises(ValueError):
            labels.mark_lr_complete(None)


if __name__ == '__main__':
    unittest.main()
