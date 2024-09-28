# batched-fwdgrad

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
    primal, dual = model(x, dx)
```