# batched-fwdgrad
## Introduction
Computes multiple directional derivatives per forward pass by batching the JVPs.

$y=f(x)$ is computed jointly with $\{\dot{y}_i\} = \{f'(x)\dot{x}_i\}$.

## Installation
```
pip install git+https://github.com/hushon/batched-fwdgrad.git
```

## Examples
```python
from batched_fwdgrad.models import CustomVisionTransformer

# vit_b_16 model
model = CustomVisionTransformer(
    image_size=224,
    patch_size=16,
    num_layers=12,
    num_heads=12,
    hidden_dim=768,
    mlp_dim=3072,
)

batch_size = 2
num_directions = 5
x = torch.randn(batch_size, 3, 224, 224)
dx = torch.randn(batch_size, num_directions, 3, 224, 224)

with torch.no_grad():
    y, dy = model(x, dx)
```

## Estimate the gradient $\partial y / \partial x$ using JVP

We can compute $\mathop{\mathbb{E}}\left[\frac{\partial y}{\partial x} v v^\top\right] = \frac{\partial y}{\partial x} \mathop{\mathbb{E}}\left[v v^\top\right] = \frac{\partial y}{\partial x}$ where $v\sim\mathcal{N}(0,\mathbb{I})$. The JVPs $\frac{\partial y}{\partial x} v$ for the Monte Carlo estimation are efficiently computed using forward-mode differentiation.
```python
batch_size = 1
num_directions = 1000
x = torch.randn(batch_size, 3, 224, 224)
dx = torch.randn(batch_size, num_directions, 3, 224, 224)

with torch.no_grad():
    y, dy = loss_fn(x, dx)

x_grad = torch.mean(dy.view(batch_size, num_directions, 1, 1, 1) * dx, dim=1)
```

## Cite this repo
If you found this work useful, please consider citing it: 
```
@software{batchedfwdgrad2024,
  title = {{batched-fwdgrad}},
  url = {https://github.com/hushon/batched-fwdgrad},
  version = {0.1.0},
  year = {2024}
}
```