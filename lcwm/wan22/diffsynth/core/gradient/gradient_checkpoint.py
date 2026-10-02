from torch.utils.checkpoint import checkpoint


def gradient_checkpoint_forward(model, enabled: bool, *args, **kwargs):
    if not enabled:
        return model(*args, **kwargs)
    return checkpoint(
        model,
        *args,
        **kwargs,
        use_reentrant=False,
    )
