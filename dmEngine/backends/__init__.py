BACKEND_REGISTRY = {
    'torch_csr': {
        'module': 'dmEngine.backends.torch_csr',
        'class': 'TorchCSRBackend',
    },
    'semi_structured': {
        'module': 'dmEngine.backends.semi_structured',
        'class': 'SemiStructuredBackend',
    },
    'triton_wsparse': {
        'module': 'dmEngine.backends.triton_wsparse',
        'class': 'TritonWSparseBackend',
    },
    'sputnik_wsparse': {
        'module': 'dmEngine.backends.sputnik_wsparse',
        'class': 'SputnikWSparseBackend',
    },
}


def load_backend(name, config=None):
    """Dynamically load a weight-sparse backend by registry name."""
    import importlib

    info = BACKEND_REGISTRY[name]
    mod = importlib.import_module(info['module'])
    cls = getattr(mod, info['class'])
    return cls(config)
