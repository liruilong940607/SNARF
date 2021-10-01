import torch
from torch.linalg import Tensor
import torch.nn as nn
import numpy as np


class SkinNet(nn.Module):
    """Canonical coordinates to skinning weights"""
    def __init__(self, in_dim=3, out_dim=24):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.net = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        n_dim = x.shape[-1]
        assert n_dim == self.in_dim
        out = self.net(x)
        return out


class OpaNet(nn.Module):
    """Canonical coordinates to opacities"""
    def __init__(self, in_dim=3, out_dim=1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.net = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        n_dim = x.shape[-1]
        assert n_dim == self.in_dim
        out = self.net(x)
        out = torch.softmax(out, dim=-1)
        return out


def deform(
    x: torch.Tensor,
    joint_deforms: torch.Tensor, 
    skin_net: nn.Module, 
) -> torch.Tensor:
    """
    :param x: x in the canonical space. [..., 3]
    :param joint_deforms: joint deformations. [n_joints, 3 or 4, 4]
    :param skin_net: a callable function that takes in x, and outputs
        the skinning weights at x. [..., 3] -> [..., n_joints]
    :return x in the world space: [..., 3]
    """
    # [..., n_joints]
    skin_weights = skin_net(x)
    # [..., 3 or 4, 4]
    transforms = torch.einsum("...j,jpq->...pq", skin_weights, joint_deforms)
    x_world = (
        torch.einsum("...ij,...j->...i", transforms[..., 0:3, 0:3], x) 
        + transforms[..., 0:3, 3]
    )
    return x_world


def correpondance_search(
    x_world: torch.Tensor,
    joint_deforms: torch.Tensor, 
    skin_net: nn.Module, 
    steps: int = 50,
    tol: float = 1e-5,
) -> torch.Tensor:
    """
    Search the canonical correspondence x of x_world implicitly 
    as the root of the following equation:
        d(x, B) − x_world = 0
    where B is the joint_deforms, and the deformation field d() is:
        d(x, B) = [skin_net(x) dot B] dot x
    :param x_world: x in the world space. [batch_size, n_points, 3]
    :param joint_deforms: joint deformations. [n_joints, 3, 4]
    :param skin_net: a callable function that takes in x, and outputs
        the skinning weights at x. [batch_size, n_points, 3] -> 
        [batch_size, n_points, n_joints]
    :return 
        x in the canonical space. [batch_size, n_points, n_joints, 3]
        x_mask: binary mask about if x is valid. [batch_size, n_points, n_joints]
    """
    batch_size, n_points, _3 = x_world.shape
    n_joints, _3, _4 = joint_deforms.shape
    # [n_joints, 3, 4] -> [n_joints, 4, 4]
    joint_deforms = torch.cat([
        joint_deforms, 
        torch.zeros_like(joint_deforms[:, 0:1, :]),
    ], dim=1)
    # [n_joints, 4, 4]
    joint_deforms_inv = torch.pinverse(joint_deforms)
    # The initial states {x_init} are obtained by transforming 
    # the deformed point x_world rigidly to the canonical space 
    # for each of the n_joints bones:
    # x_init: [batch_size, n_points, n_joints, 3]
    x = (
        torch.einsum("jpq,bnq->bnjp", joint_deforms_inv[..., 0:3, 0:3], x_world) 
        + joint_deforms_inv[..., 0:3, 3]
    )
    x_world = x_world.unsqueeze(2)
    # J_init: [3 (output), batch_size, n_points, n_joints, 3 (input)]
    func = lambda data: deform(data, joint_deforms, skin_net).view(-1, 3).sum(dim=0)
    J = torch.autograd.functional.jacobian(func, inputs=x)
    # [batch_size, n_points, n_joints, 3 (output), 3 (input)]
    J_inv = torch.pinverse(J.permute(1, 2, 3, 0, 4))
    for i in range(steps):
        x -= torch.einsum(
            "...ij,...j->...i", 
            J_inv, 
            deform(x, joint_deforms, skin_net) - x_world
        )
        J = torch.autograd.functional.jacobian(func, inputs=x)
        J_inv = torch.pinverse(J.permute(1, 2, 3, 0, 4))
    errs = deform(x, joint_deforms, skin_net) - x_world
    errs = torch.linalg.norm(errs, ord=2, dim=-1)
    x_mask = errs < tol
    return x, x_mask


def calc_grad_skin_net(
    x: torch.Tensor,
    joint_deforms: torch.Tensor, 
    skin_net: nn.Module, 
) -> torch.Tensor:
    # grad_d__x: gradients of deformation w.r.t. x
    # [3 (output), batch_size, n_points, n_joints, 3 (input)]
    func = lambda x: deform(x, joint_deforms, skin_net).view(-1, 3).sum(dim=0)
    grad_d__x = torch.autograd.functional.jacobian(func, inputs=x)
    # grad_d__param: gradients of deformation w.r.t. skin_net parameters
    # [3 (output), batch_size, n_points, n_joints, 3 (input)]
    func = lambda skin_net: deform(x, joint_deforms, skin_net).view(-1, 3).sum(dim=0)
    grad_d__param = torch.autograd.functional.jacobian(func, inputs=skin_net)
    print (grad_d__param.shape)
    
    # print (grad_d__x.shape)
    # # [batch_size, n_points, n_joints, 3 (output), 3 (input)]
    # J_inv = torch.pinverse(J.permute(1, 2, 3, 0, 4))
    


if __name__ == "__main__":
    skin_net = SkinNet()
    x = torch.randn(2, 5, 3)
    x_world = torch.randn(2, 5, 3)
    joint_deforms = torch.randn(24, 3, 4)

    # with torch.no_grad():
    func = lambda data: deform(data, joint_deforms, skin_net)
    J = torch.autograd.functional.jacobian(func, inputs=x)

    # print ("J", J.shape)
    # print (J[0, 1, :, 0, 2, :])
    
    x_world = deform(x, joint_deforms, skin_net)
    # print (x_world.shape, x_world)

    x, x_mask = correpondance_search(x_world, joint_deforms, skin_net)
    # print (x.shape, x_mask.shape)

    calc_grad_skin_net(x, joint_deforms, skin_net)