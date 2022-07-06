import torch


def visualize_field(
    model, 
    loss_func=lambda x: x.sum(),
    bboxs=[0, 0, 1, 1], 
    size=(500, 500),
):
    x = torch.stack(
        torch.meshgrid(
            torch.linspace(bboxs[0], bboxs[2], size[0]),
            torch.linspace(bboxs[1], bboxs[3], size[1]),
            indexing="ij"
        ),
        dim=-1
    )
    x = x.clone().detach().requires_grad_(True)
    y = model(x)
    loss_func(y).backward()
    grad = x.grad
    return x, y, grad
