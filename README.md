# bpnet-pytorch (wip)

![model](img/banner.png)

Implementation of BPNet, a base-resolution deep neural network for functional genomics tasks. The offical implementation of BPNet can be found [here](https://github.com/kundajelab/bpnet).

## Installation

```shell
$ pip install bpnet-pytorch
```

## Usage
```Python
from bpnet_pytorch import BPNet

model = BPNet()

x = torch.randn(1, 4, 1000)
out = model(x)
# out['x'] contains the output of the convolution layers.
# May not be useful, but left for debugging purpose for now.
### shape: (1, 64, 1000)

# out['profile'] contains the output of profile head.
### shape: (1, 1000, 2), 2 for +/- strands.

# out['total_count'] contains the output of total count head.
### shape: (1, 2), 2 for +/- strands.
```

## TODO

- [x] Confirm that the receptive field is +-1034bp.

![receptive-field-check](img/receptive_field_check.png)

- [ ] Prepare training data.

- [ ] Train the model and reproduce the performance.