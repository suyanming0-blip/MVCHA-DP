from contextlib import contextmanager


def capture_checkpoint(model, epoch, **graphs):
    return {
        'epoch': int(epoch),
        'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        'graphs': {name: tuple(t.detach().cpu().clone() for t in pair) for name, pair in graphs.items()},
        'importance': {role: {'active_sync': {'epoch': runtime.sync_history[-1]['epoch']}}
                       for role, runtime in model.importance_runtimes.items()}
    }


@contextmanager
def updating_only(model, parameters):
    active = {id(p) for p in parameters}
    previous = [p.requires_grad for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    try:
        for p in model.parameters():
            p.requires_grad_(id(p) in active)
        yield
    finally:
        for (p, flag) in zip(model.parameters(), previous):
            p.requires_grad_(flag)
