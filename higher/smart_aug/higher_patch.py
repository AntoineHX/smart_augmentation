""" Patch for Higher package
    
    Recommended use ::
    
    import higher
    import higher_patch

    Might become unnecessary with future update of the Higher package.
"""
import higher
import torch as _torch

def detach_(self):
    """Removes all params from their compute graph in place.

    """
    # detach param groups
    for group in self.param_groups:
        for k, v in group.items():
            if isinstance(v,_torch.Tensor):
                v.detach_().requires_grad_()

    # detach state
    for state_dict in self.state:
        for k,v_dict in state_dict.items():
            if isinstance(k,_torch.Tensor): k.detach_().requires_grad_()
            for k2,v2 in v_dict.items():
                if isinstance(v2,_torch.Tensor):
                    v2.detach_().requires_grad_()

higher.optim.DifferentiableOptimizer.detach_ = detach_