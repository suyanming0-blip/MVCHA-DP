import torch
import torch.nn as nn
from mvcha_dp.training.state import updating_only
from mvcha_dp.models.hgnn import HGNN
from mvcha_dp.models.domain_adaptation import DomainDiscriminator, discrepancy_loss
from mvcha_dp.models.classifier import DualClassifier


class MVCHADP(nn.Module):

    def __init__(self, in_dim, n_hid, dropout, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_hid = n_hid
        from mvcha_dp.protocol.contract import validate
        profile = validate(cfg)
        if profile == 'WPDP':
            raise ValueError('Invalid WPDP model class')
        self.hgnn = HGNN(in_dim, n_hid, dropout, profile=profile)
        self.enable_domain_adaptation = cfg.get('enable_domain_adaptation', False)
        if self.enable_domain_adaptation:
            dd_hidden = cfg.get('domain_hidden_dim', cfg.get('cls_hidden_dim', 64))
            dd_dropout = cfg.get('domain_dropout', cfg.get('cls_dropout', 0.5))
            self.domain_disc = DomainDiscriminator(n_hid, dd_hidden, dd_dropout)
        clf_hidden = cfg.get('cls_hidden_dim', 64)
        clf_dropout = cfg.get('cls_dropout', 0.5)
        self.classifier = DualClassifier(n_hid, clf_hidden, clf_dropout)

    def forward(self, x, H, W, lambd=1.0):
        embeddings = self.hgnn.encode(x, H, W)
        (cls_out1, cls_out2) = self.classifier(embeddings)
        domain_out = None
        if self.enable_domain_adaptation:
            domain_out = self.domain_disc(embeddings, lambd)
        return (embeddings, cls_out1, cls_out2, domain_out)


def adaptation_step(model, optimizer, step, X_s, y_s, train_idx_s, H_s, W_s, X_t, H_t, W_t, cfg):
    if cfg.get('grl_lambda', 1.0) != 1.0:
        raise ValueError('Expected grl_lambda=1.0')
    if step == 'classifier':
        parameters = list(model.classifier.parameters())
    elif step == 'encoder':
        parameters = model.hgnn.encoder_parameters()
    elif step == 'domain':
        parameters = list(model.domain_disc.parameters())
    else:
        raise ValueError(step)
    model.train()
    with updating_only(model, parameters):
        optimizer.zero_grad(set_to_none=True)
        if step == 'domain':
            with torch.no_grad():
                zs = model.hgnn.encode(X_s, H_s, W_s)
                zt = model.hgnn.encode(X_t, H_t, W_t)
            ds = model.domain_disc(zs, cfg.get('grl_lambda', 1.0))
            dt = model.domain_disc(zt, cfg.get('grl_lambda', 1.0))
            domain = nn.functional.cross_entropy(
                ds,
                torch.zeros(len(X_s), dtype=torch.long, device=X_s.device)
            ) + nn.functional.cross_entropy(dt, torch.ones(len(X_t), dtype=torch.long, device=X_t.device))
            loss = domain
            cls = dis = domain.new_zeros(())
        else:
            (_, s1, s2, ds) = model(X_s, H_s, W_s, cfg.get('grl_lambda', 1.0))
            (_, t1, t2, dt) = model(X_t, H_t, W_t, cfg.get('grl_lambda', 1.0))
            cls = nn.functional.cross_entropy(s1[train_idx_s], y_s[train_idx_s]) + nn.functional.cross_entropy(s2[train_idx_s], y_s[train_idx_s])
            dis = discrepancy_loss(t1, t2) if cfg.get('enable_mcd', False) else cls.new_zeros(())
            domain = cls.new_zeros(())
            if step == 'classifier':
                loss = cls - cfg.get('lambda_discrepancy', 0.4) * dis
            else:
                if cfg.get('enable_domain_adaptation', False):
                    domain = nn.functional.cross_entropy(
                        ds,
                        torch.zeros(len(X_s), dtype=torch.long, device=X_s.device)
                    ) + nn.functional.cross_entropy(dt, torch.ones(len(X_t), dtype=torch.long, device=X_t.device))
                loss = cls + cfg.get('lambda_domain', 0.6) * domain + cfg.get('lambda_discrepancy', 0.4) * dis
        loss.backward()
        optimizer.step()
    return dict(
        loss=float(loss.detach()),
        cls=float(cls.detach()),
        domain=float(domain.detach()),
        discrepancy=float(dis.detach())
    )
