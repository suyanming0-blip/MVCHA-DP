VERSION = 'MVCHA_DP_V2_2'
PROFILES = {'WPDP': 64, 'CVDP': 128, 'CPDP': 64}


def validate(cfg):
    if cfg.get('model_version') != VERSION:
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
    if not cfg.get('use_node_importance') or not cfg.get('use_dynamic_k'):
        raise ValueError('Node importance and dynamic K are required')
    if scenario != 'WPDP' and (not cfg.get('enable_domain_adaptation') or not cfg.get('enable_mcd')):
        raise ValueError('Domain adaptation or MCD is disabled')
    return scenario
