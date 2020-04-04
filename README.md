# A PyTorch implementation of "An Empirical study of Conv-TasNet"

## Conv-Tasnet(Deep w / dilation)
```python
Encoder:
    Conv1D(1, N, kernel_size, stride=stride)
    Conv1D(N, N, kernel_size=3, stride=1, dilation=1,padding=1)
    Conv1D(N, N, kernel_size=3, stride=1, dilation=2, padding=2)
    Conv1D(N, N, kernel_size=3, stride=1, dilation=4, padding=4)
    Conv1D(N, N, kernel_size=3, stride=1, dilation=8, padding=8)

Decoder:
    nn.ConvTranspose1d(N, N, kernel_size=3, stride=1, dilation=8, padding=8)
    nn.ConvTranspose1d(N, N, kernel_size=3, stride=1, dilation=4, padding=4)
    nn.ConvTranspose1d(N, N, kernel_size=3, stride=1, dilation=2, padding=2)
    nn.ConvTranspose1d(N, N, kernel_size=3, stride=1, dilation=1, padding=1)
    nn.ConvTranspose1d(N, 1, kernel_size=kernel_size, stride=stride)
```

## Conv-Tasnet(Deep w / PRelu)
```python
Encoder:
    Conv1D(1, N, kernel_size, stride=stride)
    Conv1D(N, N, kernel_size=3, stride=1,padding=1)
    Conv1D(N, N, kernel_size=3, stride=1, padding=1)
    Conv1D(N, N, kernel_size=3, stride=1, padding=1)
    Conv1D(N, N, kernel_size=3, stride=1, padding=1)

Decoder:
    nn.ConvTranspose1d(N, N, kernel_size=3, stride=1, padding=1)
    nn.ConvTranspose1d(N, N, kernel_size=3, stride=1, padding=1)
    nn.ConvTranspose1d(N, N, kernel_size=3, stride=1, padding=1)
    nn.ConvTranspose1d(N, N, kernel_size=3, stride=1, padding=1)
    nn.ConvTranspose1d(N, 1, kernel_size=kernel_size, stride=stride)
```

## How to use

You can replace the Conv_TasNet.py file in the [Conv-TasNet](https://github.com/JusperLee/Conv-TasNet) repository for training.

## Reference
[1]. Kadioglu B, Horgan M, Liu X, et al. An empirical study of Conv-TasNet[J]. arXiv preprint arXiv:2002.08688, 2020.