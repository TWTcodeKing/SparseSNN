import torch

if __name__ == '__main__':
    test_t = torch.randn(4, 1, 3, 224, 224)
    t_ = test_t.flatten(0, 1)
    print(t_.shape)