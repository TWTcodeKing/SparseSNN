"""Dense input generators (100% non-zero) for weight-sparse benchmarking."""

import torch


def make_dense_input(batch_size=32, channels=3, height=224, width=224, device='cuda'):
    """Generate random input tensor guaranteed to have no zeros.

    Args:
        batch_size: Batch size.
        channels: Number of input channels.
        height: Image height.
        width: Image width.
        device: Target device.

    Returns:
        Tensor of shape (batch_size, channels, height, width) with all non-zero values.
    """
    return torch.rand(batch_size, channels, height, width, device=device) + 0.01


class DummyDataloader:
    """Iterable that yields (dense_images, random_labels) tuples.

    Each batch contains fresh random dense data with no zero values.
    """

    def __init__(self, batch_size=32, num_classes=1000, channels=3,
                 height=224, width=224, num_batches=100, device='cuda'):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.channels = channels
        self.height = height
        self.width = width
        self.num_batches = num_batches
        self.device = device

    def __iter__(self):
        for _ in range(self.num_batches):
            images = make_dense_input(
                self.batch_size, self.channels, self.height, self.width, self.device
            )
            labels = torch.randint(0, self.num_classes, (self.batch_size,), device=self.device)
            yield images, labels

    def __len__(self):
        return self.num_batches
