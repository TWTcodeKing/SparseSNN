"""Standard ANN ResNet wrappers using torchvision for weight-sparse benchmarking."""

import torchvision.models as tv_models

RESNET_REGISTRY = {
    'resnet18': 'resnet18',
    'resnet34': 'resnet34',
    'resnet50': 'resnet50',
    'resnet101': 'resnet101',
}


def build_resnet(arch='resnet50', num_classes=1000):
    """Build standard ANN ResNet from torchvision. No spiking neurons."""
    if arch not in RESNET_REGISTRY:
        raise ValueError(f"Unknown arch '{arch}'. Available: {list(RESNET_REGISTRY.keys())}")
    factory = getattr(tv_models, RESNET_REGISTRY[arch])
    return factory(num_classes=num_classes, weights=None)
