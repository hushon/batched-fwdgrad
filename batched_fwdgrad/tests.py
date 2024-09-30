import torch
from torch import nn
import torch.nn.functional as F
import einops
from batched_fwdgrad import jvp_layers, models

torch.backends.cudnn.benchmark = True
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True


def rademacher_vector(*size, **kwargs):
    # Sample from Bernoulli(0.5) and convert {0, 1} to {-1, 1}
    return 2 * torch.randint(0, 2, size=size, **kwargs).float() - 1


def jvp_sanity_check(model, jvp_model, x, dx):
    num_directions = dx.size(1)

    dual_list = []
    for i in range(num_directions):
        primal, dual = torch.autograd.functional.jvp(model, x, dx[:, i])
        dual_list.append(dual)
    dual = torch.stack(dual_list, dim=1)

    jvp_model.load_state_dict(state_dict)
    with torch.no_grad():
        primal2 = jvp_model(x, dx)
        dual2 = primal2.tangent
        # primal2, dual2 = model2(x, dx, target)

    assert torch.allclose(primal.view(-1), primal2.view(-1), atol=1e-6)
    assert torch.allclose(dual.view(-1), dual2.view(-1), atol=1e-6)


# def grad_sanity_check(model, jvp_model, x, dx):
#     state_dict = model.state_dict()
#     loss = model(x).view(batch_size, -1).sum(-1)
#     # loss = model(x, x, x)[0].view(batch_size, -1).sum(-1)
#     # grad = torch.autograd.grad(loss, x)
#     loss.backward()
#     jvp_model.load_state_dict(state_dict)

#     num_directions = 10000
#     with torch.no_grad():
#         dx = torch.randn(batch_size, num_directions, 5, 64)
#         # dx = rademacher_vector((batch_size, num_directions, 5, 64))
#         primal2, dual2 = jvp_model(x, dx)
#         primal2 = primal2.view(batch_size, -1).sum(-1)
#         dual2 = dual2.view(batch_size, num_directions, -1).sum(-1)
#         grad_fwd = (dx * dual2.view(batch_size, num_directions, 1, 1)).mean(1)

#     # print cosine similarity
#     print(F.cosine_similarity(x.grad.view(batch_size, -1), grad_fwd.view(batch_size, -1)))


@torch.no_grad()
def measure_time(model, inputs):
    import time, tqdm
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in tqdm.trange(100):
        primal, dual = model(*inputs)
        # output = model(*inputs)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"Inference time: {end_time - start_time} seconds")


if __name__ == "__main__":

    import torchvision
    # model = nn.Linear(5, 5)
    # model = nn.Softmax()
    # model = nn.GELU()
    # model = nn.Tanh()
    # model = nn.LayerNorm((5,))
    # model = torchvision.models.vit_b_16()
    # model = nn.MultiheadAttention(64, 8, batch_first=True)
    # model = nn.CrossEntropyLoss(reduction='none')
    # forward_fn = lambda x: model(x, target)
    # forward_fn = lambda x: model(x, x, x)[0]
    import torchvision
    # model = torchvision.models.vit_b_16()
    import timm
    model = timm.create_model('vit_base_patch16_224')


    # model2 = jvp_layers.CustomLinear(5, 5)
    # model2 = jvp_layers.CustomSoftmax()
    # model2 = jvp_layers.CustomGeLU()
    # model2 = jvp_layers.CustomTanh()
    # model2 = jvp_layers.CustomLayerNorm((5,))
    # model2 = jvp_layers.CustomMultiheadSelfAttention(64, 8, batch_first=True)
    # model2 = jvp_layers.CustomCrossEntropyLoss(reduction='none')
    # model2 = models.CustomVisionTransformer(
    #     image_size=224,
    #     patch_size=16,
    #     num_layers=12,
    #     num_heads=12,
    #     hidden_dim=768,
    #     mlp_dim=3072,
    # )
    from batched_fwdgrad import timm_models
    kwargs = {'patch_size': 16, 'embed_dim': 768, 'depth': 12, 'num_heads': 12, 'num_classes': 1000, 'in_chans': 3, 'img_size': (224, 224)}
    net = timm_models.VisionTransformer(**kwargs)
    model2 = timm_models.CustomPromptViT(net, 1)

    # model2.load_state_dict(model.state_dict())

    batch_size = 2
    num_directions = 10
    sample_shape = (3, 224, 224)
    x = torch.randn(batch_size, *sample_shape, requires_grad=True)
    dx = rademacher_vector(batch_size, num_directions, *sample_shape)
    target = torch.randint(0, 5, (batch_size,))


    forward_fn = lambda x: model2(x, prompts)
    jvp_sanity_check(forward_fn, model2, x, dx)
    breakpoint()


    # batch_size = 64*(12+1)
    batch_size = 64
    num_directions = 12
    sample_shape = (3, 224, 224)
    device = torch.device("cuda")
    x = torch.randn(batch_size, *sample_shape, device=device)
    dx = rademacher_vector(batch_size, num_directions, *sample_shape, device=device)

    model = model.to(device)
    model2 = model2.to(device)
    measure_time(model2, (x, dx))