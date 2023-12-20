import torch
import torch.nn as nn

from nerfacc.losses import distortion

class NeRFLoss(nn.Module):
    def __init__(self, lambda_opacity=0.0, lambda_distortion=0.01):
        super().__init__()

        self.lambda_opacity = lambda_opacity
        self.lambda_distortion = lambda_distortion

    def forward(self, rgb, target, opp=None, distkwargs=None):
        d = {}
        d['rgb'] = (rgb-target)**2

        if self.lambda_opacity > 0:
            o = opp+torch.finfo(torch.float16).eps
            # encourage opacity to be either 0 or 1 to avoid floater
            d['opacity'] = self.lambda_opacity*(-o*torch.log(o))

        if self.lambda_distortion > 0 and distkwargs is not None:
            # import pdb; pdb.set_trace()
            # num_rays = distkwargs['rays_a'].shape[0]
            # num_samples_per_ray = distkwargs['ws'].shape[0] // num_rays
            # ray_indices = torch.arange(num_rays, device = distkwargs['ws'].device)
            # ray_indices = torch.repeat_interleave(ray_indices, num_samples_per_ray)
            
            d['distortion'] = self.lambda_distortion * \
                distortion(distkwargs['ws'], distkwargs['deltas'],
                            distkwargs['ts'],  distkwargs['rays_a'][:, 1:])

        return d
