VERSION = 'MVCHA_DP_MVHR_PARITY_V2_2'
SCHEMA = 2
PROFILES = {'WPDP': 64, 'CVDP': 128, 'CPDP': 64}


def is_parity(cfg):
    return cfg.get('model_version') == VERSION


def validate(cfg):
    if not is_parity(cfg):
        raise ValueError('Unsupported model version')
    scenario = cfg.get('scenario')
    if scenario not in PROFILES:
        raise ValueError('Unknown scenario')
    width = PROFILES[scenario]
    if cfg.get('importance_math_version') != 'V2.1' or cfg.get('feature_order') != 'TNC':
        raise ValueError('Importance math/feature schema mismatch')
    if cfg.get('n_hid') != width or cfg.get('drop_out') != 0.5:
        raise ValueError('Invalid scenario encoder profile')
    if type(cfg.get('max_epoch')) is not int or cfg['max_epoch'] != 200:
        raise ValueError('Expected max_epoch=200')
    if cfg.get('cls_hidden_dim') != 32 or cfg.get('cls_dropout') != 0:
        raise ValueError('Expected cls_hidden_dim=32 and cls_dropout=0')
    if scenario != 'WPDP' and (not cfg.get('enable_domain_adaptation') or not cfg.get('enable_mcd')):
        raise ValueError('Domain adaptation or MCD is disabled')
    return scenario
