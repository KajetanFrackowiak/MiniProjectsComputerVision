import torch
import torch.nn as nn

class PlanarFlow(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.u = nn.Parameter(torch.randn(1, dim))
        self.w = nn.Parameter(torch.randn(1, dim))
        self.b = nn.Parameter(torch.randn(1))
    
    def forward(self, z):
        # z: (batch, dim)
        linear = torch.matmul(z, self.w.t()) + self.b
        f_z = z + self.u * torch.tanh(linear)
        
        # Compute log-det-Jacobian
        psi = (1 - torch.tanh(linear) ** 2) * self.w
        log_det_jacobian = torch.log(torch.abs(1 + torch.matmul(psi, self.u.t())) + 1e-8)
        return f_z, log_det_jacobian 


class RadialFlow(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.z0 = nn.Parameter(torch.randn(1, dim))
        self.alpha = nn.Parameter(torch.randn(1))
        self.beta = nn.Parameter(torch.randn(1))

    def forward(self, z):
        # Radial flow transformation
        diff = z - self.z0
        r = torch.norm(diff, dim=1, keepdim=True)
        h = 1 / (self.alpha + r)
        f_z = z + self.beta * h * diff

        # Log determinant of Jacobian
        d = z.size(1)
        h_prime = -1 / (self.alpha + r) ** 2
        log_det_jacobian = (d - 1) * torch.log(1 + self.beta * h) + torch.log(1 + self.beta * h + self.beta * h_prime * r)
        return f_z, log_det_jacobian