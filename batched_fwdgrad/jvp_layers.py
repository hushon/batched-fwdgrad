import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Tuple
import einops
import math


class CustomSequential(nn.Sequential):
    def forward(self, input: torch.Tensor, jvp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for module in self:
            input, jvp = module(input, jvp)
        return input, jvp


class CustomLinear(nn.Linear):
    def forward(self, input: torch.Tensor, jvp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output = super().forward(input)
        jvp = super().forward(jvp)
        return output, jvp


class CustomReLU(nn.ReLU):
    def forward(self, input, jvp):
        mask = input > 0.
        output = F.relu(input, self.inplace)
        jvp = jvp * mask.unsqueeze_(1)
        return output, jvp


# class CustomLeakyReLU(nn.LeakyReLU):
#     def forward(self, input, jvp):
#         mask = input > 0.
#         output = F.leaky_relu(input, self.negative_slope, self.inplace)
#         jvp = jvp * torch.where(mask, 1., self.negative_slope)
#         # jvp = jvp.where(mask, jvp*self.negative_slope)
#         return output, jvp


# class CustomGeLU(nn.GELU):
#     def forward(self, input: torch.Tensor, jvp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         r"""
#         Jacobian is computed using the sigmoid approximation of GeLU
#         y ~= x * sigmoid(1.702 * x)
#         J ~= sigmoid(1.702 * x) * (1 + 1.702 * x * (1 - sigmoid(1.702 * x)))
#         """
#         output = F.gelu(input)
#         sigmoid = torch.sigmoid(1.702 * input)
#         jacobian = sigmoid * (1 + 1.702 * input * (1 - sigmoid)).unsqueeze_(1)
#         jvp = jvp.mul_(jacobian)
#         return output, jvp

class CustomGeLU(nn.GELU):
    def forward(self, x: torch.Tensor, x_tangent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Efficiently computes the GELU activation and its JVP.

        Args:
            x (torch.Tensor): Input tensor.
            x_tangent (torch.Tensor): Tangent of the input tensor.

        Returns:
            y (torch.Tensor): Output after applying GELU.
            y_tangent (torch.Tensor): Tangent of the output.
        """
        # Constants
        sqrt_2 = math.sqrt(2.0)
        sqrt_2_pi = math.sqrt(2.0 * math.pi)

        # Compute x over sqrt(2)
        x_over_sqrt2 = x / sqrt_2

        # Compute Phi(x) and phi(x)
        Phi_x = 0.5 * (1.0 + torch.erf(x_over_sqrt2))
        phi_x = torch.exp(-0.5 * x ** 2) / sqrt_2_pi

        # Compute y
        y = x * Phi_x

        # Compute gradient
        grad = Phi_x + x * phi_x

        # Compute y_tangent
        y_tangent = grad.unsqueeze(1) * x_tangent

        return y, y_tangent


class CustomTanh(nn.Tanh):
    def forward(self, input: torch.Tensor, jvp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output = F.tanh(input)
        jvp = jvp * (1 - output ** 2).unsqueeze_(1)
        return output, jvp


class CustomDropout(nn.Dropout):
    def forward(self, x: torch.Tensor, x_tangent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output = super().forward(x)
        if self.training:
            mask = (output.abs() > 0.0).float()
            jvp = (mask.unsqueeze(1) * x_tangent).div_(1 - self.p)
        else:
            jvp = x_tangent
        return output, jvp


class CustomConv2d(nn.Conv2d):
    def forward(self, input: torch.Tensor, jvp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_directions, channel_dims, h, w = jvp.shape
        jvp = jvp.view(batch_size * num_directions, channel_dims, h, w)
        output = super().forward(torch.cat([input, jvp], dim=0))
        output, jvp = output[:batch_size, ...], output[batch_size:, ...]
        jvp = jvp.view(batch_size, num_directions, -1)
        return output, jvp


# class CustomBatchNorm2d(nn.BatchNorm2d):
#     def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
#                  track_running_stats=True):
#         super().__init__(num_features=num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
#         for k in ('weight', 'bias'):
#             x = self._parameters.pop(k)
#             if x is not None:
#                 self.register_buffer(k, x.data)
#             else:
#                 self.register_buffer(k, None)
#         self.weight_tangent = nn.Parameter(torch.zeros_like(self.weight))
#         self.bias_tangent = nn.Parameter(torch.zeros_like(self.bias))

#     def forward(self, input: torch.Tensor, jvp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         # assert not self.training
#         self._check_input_dim(input)

#         # exponential_average_factor is set to self.momentum
#         # (when it is available) only so that it gets updated
#         # in ONNX graph when this node is exported to ONNX.
#         if self.momentum is None:
#             exponential_average_factor = 0.0
#         else:
#             exponential_average_factor = self.momentum

#         if self.training and self.track_running_stats:
#             # TODO: if statement only here to tell the jit to skip emitting this when it is None
#             if self.num_batches_tracked is not None:  # type: ignore
#                 self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore
#                 if self.momentum is None:  # use cumulative moving average
#                     exponential_average_factor = 1.0 / float(self.num_batches_tracked)
#                 else:  # use exponential moving average
#                     exponential_average_factor = self.momentum

#         r"""
#         Decide whether the mini-batch stats should be used for normalization rather than the buffers.
#         Mini-batch stats are used in training mode, and in eval mode when buffers are None.
#         """
#         if self.training:
#             bn_training = True
#         else:
#             bn_training = (self.running_mean is None) and (self.running_var is None)

#         r"""
#         Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
#         passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
#         used for normalization (i.e. in eval mode when buffers are not None).
#         """
#         assert self.running_mean is None or isinstance(self.running_mean, torch.Tensor)
#         assert self.running_var is None or isinstance(self.running_var, torch.Tensor)

#         output = F.batch_norm(
#             input,
#             # If buffers are not to be tracked, ensure that they won't be updated
#             self.running_mean if not self.training or self.track_running_stats else None,
#             self.running_var if not self.training or self.track_running_stats else None,
#             self.weight, self.bias, bn_training, exponential_average_factor, self.eps)
#         jvp = F.batch_norm(
#             input,
#             # If buffers are not to be tracked, ensure that they won't be updated
#             self.running_mean if not self.training or self.track_running_stats else None,
#             self.running_var if not self.training or self.track_running_stats else None,
#             self.weight_tangent, self.bias_tangent, False, exponential_average_factor, self.eps) \
#             + F.batch_norm(
#             jvp,
#             # If buffers are not to be tracked, ensure that they won't be updated
#             torch.zeros_like(self.running_mean) if not self.training or self.track_running_stats else None,
#             self.running_var if not self.training or self.track_running_stats else None,
#             self.weight, None, False, exponential_average_factor, self.eps)
#         return output, jvp


class CustomLayerNorm(nn.LayerNorm):
    r"""
    Compute the LayerNorm function and its JVP.

    Inputs:
    - x: Input tensor of shape [..., normalized_shape], requires_grad=False
    - x_tangent: Tangent of x, same shape as x
    - gamma: Optional scale parameter of shape normalized_shape
    - gamma_tangent: Tangent of gamma, same shape as gamma
    - self.bias: Optional shift parameter of shape normalized_shape
    - beta_tangent: Tangent of beta, same shape as beta
    - eps: Small epsilon for numerical stability

    Returns:
    - y: Output tensor, same shape as x
    - y_tangent: Tangent of y, same shape as y
    """
    def forward(self, x: torch.Tensor, jvp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_tangent = jvp
        x = x.unsqueeze(1)

        # Compute mean and its tangent
        mu = x.mean(dim=-1, keepdim=True)
        mu_tangent = x_tangent.mean(dim=-1, keepdim=True)

        # Centered input and its tangent
        x_minus_mu = x - mu
        x_minus_mu_tangent = x_tangent - mu_tangent

        # Compute variance and its tangent
        sigma2 = (x_minus_mu ** 2).mean(dim=-1, keepdim=True)

        # Compute standard deviation and its tangent
        sigma = torch.sqrt(sigma2 + self.eps)
        sigma_tangent = (x_minus_mu * x_minus_mu_tangent).mean(dim=-1, keepdim=True) / sigma

        # Compute normalized input and its tangent
        hat_x = x_minus_mu / sigma
        hat_x_tangent = (x_minus_mu_tangent * sigma - x_minus_mu * sigma_tangent) / (sigma ** 2)

        # Apply gamma and beta, compute their tangents if provided
        if self.weight is not None:
            y = self.weight * hat_x
            y_tangent = self.weight * hat_x_tangent
        else:
            y = hat_x
            y_tangent = hat_x_tangent

        if self.bias is not None:
            y = y + self.bias

        output = y.squeeze(1)
        jvp = y_tangent
        return output, jvp


class CustomAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def forward(self, input: torch.Tensor, jvp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_directions, channel_dims, h, w = jvp.shape
        jvp = jvp.view(batch_size * num_directions, channel_dims, h, w)
        output = super().forward(torch.cat([input, jvp], dim=0))
        output, jvp = output[:batch_size, ...], output[batch_size:, ...]
        jvp = jvp.view(batch_size, num_directions, -1)
        return output, jvp


# # TODO: Implement batched jvp
# class CustomMaxPool2d(nn.MaxPool2d):
#     def forward(self, input: torch.Tensor, jvp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         output, indices = F.max_pool2d(input, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode, True)
#         b, c, out_h, out_w = output.shape
#         jvp = torch.gather(jvp.view(b, c, -1), 2, indices.view(b, c, -1)).reshape(b, c, out_h, out_w)
#         return output, jvp


class CustomSoftmax(nn.Softmax):
    def forward(self, input: torch.Tensor, jvp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        J = diag(softmax(x)) - softmax(x) @ softmax(x)^T
        dy = J @ dx = softmax(x)*dx - softmax(x) @ softmax(x)^T @ dx
        """
        assert self.dim == -1
        softmax = super().forward(input)
        # jvp = softmax.unsqueeze(1) * jvp - softmax.unsqueeze(1) * torch.einsum("bj,bij->bi", softmax, jvp).unsqueeze(2)
        # jvp = softmax.unsqueeze(1) * (jvp - torch.einsum("bj,bij->bi", softmax, jvp).unsqueeze(2))
        jvp = softmax.unsqueeze(1) * (jvp - (softmax.unsqueeze(1)*jvp).sum(dim=self.dim, keepdim=True))
        return softmax, jvp


class CustomCrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(self, input: torch.Tensor, jvp: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        J = softmax(x) - one_hot(y)
        """
        assert self.reduction == 'none'
        output = super().forward(input, target)
        softmax = F.softmax(input, dim=-1)
        onehot = F.one_hot(target, num_classes=input.size(-1)).float()
        jvp = torch.sum((softmax - onehot).unsqueeze(1) * jvp, dim=-1)
        return output, jvp


# class CustomMultiheadSelfAttention(nn.MultiheadAttention):
#     def forward(self, x, jvp) -> Tuple[torch.Tensor, torch.Tensor]:
#         assert self.batch_first

#         batch_size, seq_len, embed_dim = x.size()
#         num_heads = self.num_heads
#         head_dim = self.head_dim

#         # Linear projections
#         Q, K, V = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
#         Q_tangent, K_tangent, V_tangent = F.linear(jvp, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)

#         # Reshape and transpose for multiple heads
#         Q = Q.view(batch_size, seq_len, num_heads, head_dim)
#         K = K.view(batch_size, seq_len, num_heads, head_dim)
#         V = V.view(batch_size, seq_len, num_heads, head_dim)

#         Q_tangent = Q_tangent.view(batch_size, seq_len, num_heads, head_dim)
#         K_tangent = K_tangent.view(batch_size, seq_len, num_heads, head_dim)
#         V_tangent = V_tangent.view(batch_size, seq_len, num_heads, head_dim)

#         # Scaled dot-product attention
#         scaling = head_dim ** -0.5
#         scores = torch.einsum('bqnd,bknd->bqkn', Q, K) * scaling  # Shape: (batch_size, seq_len, seq_len, num_heads)
#         scores_tangent = (torch.einsum('bqnd,bknd->bqkn', Q_tangent, K) + torch.einsum('bqnd,bknd->bqkn', Q, K_tangent)) * scaling

#         attn_weights = F.softmax(scores, dim=-2)
#         attn_weights_tangent = _softmax_jvp(attn_weights, scores_tangent, dim=-2)

#         assert not self.dropout > 0.0

#         # Compute attention outputs
#         attention_output = torch.einsum('bqkn,bknd->bnqd', attn_weights, V) # Shape: (batch_size, num_heads, seq_len, head_dim)
#         attention_output_tangent = torch.einsum('bqkn,bknd->bnqd', attn_weights_tangent, V) + torch.einsum('bqkn,bknd->bnqd', attn_weights, V_tangent)

#         # Step 6: Concatenate heads
#         attention_output = torch.einsum('bnqd->bqnd', attention_output).reshape(batch_size, seq_len, embed_dim)
#         attention_output_tangent = torch.einsum('bnqd->bqnd', attention_output_tangent).reshape(batch_size, seq_len, embed_dim)

#         # Step 7: Final linear projection
#         output = self.out_proj(attention_output)  # Shape: (batch_size, seq_len, embed_dim)
#         output_tangent = self.out_proj(attention_output_tangent)  # Shape: (batch_size, seq_len, embed_dim)

#         return output, output_tangent
class CustomMultiheadSelfAttention(nn.MultiheadAttention):
    def forward(self, x, jvp) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.batch_first

        batch_size, seq_len, embed_dim = x.size()
        num_directions = jvp.size(1)
        num_heads = self.num_heads
        head_dim = self.head_dim

        # Linear projections
        Q, K, V = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        Q_tangent, K_tangent, V_tangent = F.linear(jvp, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)

        # Reshape and transpose for multiple heads
        Q = Q.view(batch_size, seq_len, num_heads, head_dim)
        K = K.view(batch_size, seq_len, num_heads, head_dim)
        V = V.view(batch_size, seq_len, num_heads, head_dim)

        Q_tangent = Q_tangent.view(batch_size, num_directions, seq_len, num_heads, head_dim)
        K_tangent = K_tangent.view(batch_size, num_directions, seq_len, num_heads, head_dim)
        V_tangent = V_tangent.view(batch_size, num_directions, seq_len, num_heads, head_dim)

        # Scaled dot-product attention
        scaling = head_dim ** -0.5
        scores = torch.einsum('bqnd,bknd->bnqk', Q, K) * scaling  # Shape: (batch_size, seq_len, seq_len, num_heads)
        scores_tangent = (torch.einsum('bmqnd,bknd->bmnqk', Q_tangent, K) + torch.einsum('bqnd,bmknd->bmnqk', Q, K_tangent)) * scaling

        attn_weights, attn_weights_tangent = CustomSoftmax(dim=-1)(scores, scores_tangent)

        assert not self.dropout > 0.0

        # Compute attention outputs
        attention_output = torch.einsum('bnqk,bknd->bqnd', attn_weights, V) # Shape: (batch_size, num_heads, seq_len, head_dim)
        attention_output_tangent = torch.einsum('bmnqk,bknd->bmqnd', attn_weights_tangent, V) + torch.einsum('bnqk,bmknd->bmqnd', attn_weights, V_tangent)

        # Step 6: Concatenate heads
        attention_output = attention_output.reshape(batch_size, seq_len, embed_dim)
        attention_output_tangent = attention_output_tangent.reshape(batch_size, num_directions, seq_len, embed_dim)

        # Step 7: Final linear projection
        output = self.out_proj(attention_output)
        output_tangent = self.out_proj(attention_output_tangent)

        return output, output_tangent

