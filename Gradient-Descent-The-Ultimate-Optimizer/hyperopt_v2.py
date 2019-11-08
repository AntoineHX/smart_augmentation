import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer

class Optimizable():
    """
    This is the interface for anything that has parameters that need to be
    optimized, somewhat like torch.nn.Model but with the right plumbing for
    hyperoptimizability. (Specifically, torch.nn.Model uses the Parameter
    interface which does not give us enough control about the detachments.)
    Nominal operation of an Optimizable at the lowest level is as follows:
        o = MyOptimizable(…)
        o.initialize()
        loop {
            o.begin()
            o.zero_grad()
            loss = –compute loss function from parameters–
            loss.backward()
            o.adjust()
        }
    Optimizables recursively handle updates to their optimiz*ers*.
    """
    #def __init__(self):
    #    super(Optimizable, self).__init__()
    #    self.parameters = nn.Parameter(torch.zeros(()))

    def __init__(self, parameters, optimizer):
        self.params = parameters  # a dict mapping names to tensors
        self.optimizer = optimizer  # which must itself be Optimizable!
        self.all_params_with_gradients = []
        #self.device = device

    def initialize(self):
        """Initialize parameters, e.g. with a Kaiming initializer."""
        pass

    def begin(self):
        """Enable gradient tracking on current parameters."""
        self.all_params_with_gradients = nn.ParameterList() #Reintialisation pour eviter surcharge de la memoire
        print("Opti param :", type(self.params))
        #for name, param in self.params:
        if isinstance(self.params,dict): #Dict
            for name, param in self.params:
                param.requires_grad_()  # keep gradient information…
                param.retain_grad()  # even if not a leaf…
                self.all_params_with_gradients.append(param)
        if isinstance(self.params,list): #List
            for param in self.params:
                param.requires_grad_()  # keep gradient information…
                param.retain_grad()  # even if not a leaf…
                self.all_params_with_gradients.append(param)
        self.optimizer.begin()

    def zero_grad(self):
        """ Set all gradients to zero. """
        for param in self.all_params_with_gradients:
            param.grad = torch.zeros(param.shape, device=param.device)
        self.optimizer.zero_grad()

    """ Note: at this point you would probably call .backwards() on the loss
    function. """

    def adjust(self):
        """ Update parameters """
        pass


class NoOpOptimizer(Optimizable):#, nn.Module):
    """
    NoOpOptimizer sits on top of a stack, and does not affect what lies below.
    """

    def __init__(self):
        #super(Optimizable, self).__init__()
        pass

    def initialize(self):
        pass

    def begin(self):
        #print("NoOpt begin")
        pass

    def zero_grad(self):
        pass

    def adjust(self, params):
        pass

    def step(self):
        pass

    def print_grad_fn(self):
        pass
        
    def __str__(self):
        return "static"


class SGD(Optimizer, nn.Module): #Eviter Optimizer
    """
    A hyperoptimizable SGD
    """

    def __init__(self, params, lr=0.01, height=0):
        self.height=height
        #params : a optimiser
        #reste (defaults) param de l'opti
        print('SGD - H', height)
        nn.Module.__init__(self)

        optim_keys = ('lr','') #A mettre dans Optimizable ? #'' pour eviter iteration dans la chaine de charactere...
        '''
        self_params = {"lr": torch.tensor(lr),
                        "momentum": 0,
                        "dampening":0,
                        "weight_decay":0,
                        "nesterov": False}
        '''
        #self_params = dict(lr=torch.tensor(lr), 
        #                    momentum=0, dampening=0, weight_decay=0, nesterov=False)

        self_params = nn.ParameterDict({
            "lr": nn.Parameter(torch.tensor(lr)),
            "momentum": nn.Parameter(torch.tensor(0.0)),
            "dampening": nn.Parameter(torch.tensor(0.0)),
            "weight_decay": nn.Parameter(torch.tensor(0.0)),
        })

        for k in self_params.keys() & optim_keys:
            self_params[k].requires_grad_()  # keep gradient information…
            self_params[k].retain_grad()  # even if not a leaf…
            #self_params[k].register_hook(print)

        if height==0:
            optimizer = NoOpOptimizer()
        else:
            #def dict_generator(): yield {k: self_params[k] for k in self_params.keys() & optim_keys}
            #(dict for dict in {k: self_params[k] for k in self_params.keys() & optim_keys}) #Devrait mar
            optimizer = SGD(params=(self_params[k]for k in self_params.keys() & optim_keys), lr=lr, height=height-1)
            #optimizer.register_backward_hook(print)

        self.optimizer = optimizer
        #if(height==0):
        #    for n,p in params.items():
        #        print(n,p)

        #Optimizable.__init__(self, self_params, optimizer)

        #print(type(params))
        #for p in params:
        #    print(type(p))
        Optimizer.__init__(self, params, self_params)

        for group in self.param_groups:
            for p in group['params']:
                print(type(p.data), p.size())
        print('End SGD-H', height)  

    def begin(self):
        for group in self.param_groups:
            for p in group['params']:
                #print(type(p.data), p.size())
                p.requires_grad_()  # keep gradient information…
                p.retain_grad()  # even if not a leaf…
                #p.register_hook(lambda x: print(self.height, x.grad_fn))

        self.optimizer.begin()

    def print_grad_fn(self):
        self.optimizer.print_grad_fn()
        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                print(self.height," - ", i, p.grad_fn)

    #def adjust(self, params):
    #    self.optimizer.adjust(self.params)
    #    for name, param in params.items():
    #        g = param.grad.detach()
    #        params[name] = param.detach() - g * self.params["lr"]

    def step(self):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        print('SGD start')
        self.optimizer.step()

        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                #d_p = p.grad.data
                d_p = p.grad.detach()

                #print(group['lr'])
                p.data.add_(-group['lr'].item(), d_p)
                #group['params'][i] = p.detach() - d_p * group['lr']
                p.data-= group['lr']*d_p #Data ne pas utiliser perte info

            for p in group['params']:
                if p.grad is None:
                    print(p, p.grad)
                    continue

        print("SGD end")
        #return loss

    def __str__(self):
        return "sgd(%f) / " % self.params["lr"] + str(self.optimizer)


class Adam(Optimizable, nn.Module):
    """
    A fully hyperoptimizable Adam optimizer
    """

    def clamp(x):
        return (x.tanh() + 1.0) / 2.0

    def unclamp(y):
        z = y * 2.0 - 1.0
        return ((1.0 + z) / (1.0 - z)).log() / 2.0

    def __init__(
        self,
        alpha=0.001,
        beta1=0.9,
        beta2=0.999,
        log_eps=-8.0,
        optimizer=NoOpOptimizer(),
        device = torch.device('cuda')
    ):
        #super(Adam, self).__init__()
        nn.Module.__init__(self)
        self.device = device
        params = nn.ParameterDict({
            "alpha": nn.Parameter(torch.tensor(alpha, device=self.device)),
            "beta1": nn.Parameter(Adam.unclamp(torch.tensor(beta1, device=self.device))),
            "beta2": nn.Parameter(Adam.unclamp(torch.tensor(beta2, device=self.device))),
            "log_eps": nn.Parameter(torch.tensor(log_eps, device=self.device)),
        })
        Optimizable.__init__(self, params, optimizer)
        self.num_adjustments = 0
        self.cache = {}

        for name, param in params.items():
            param.requires_grad_()  # keep gradient information…
            param.retain_grad()  # even if not a leaf…

    def adjust(self, params, pytorch_mod=False):
        self.num_adjustments += 1
        self.optimizer.adjust(self.params)
        t = self.num_adjustments
        beta1 = Adam.clamp(self.params["beta1"])
        beta2 = Adam.clamp(self.params["beta2"])

        updated_param = []
        if pytorch_mod:
            params = params.named_parameters(prefix='') #Changer nom d'input...

        for name, param in params:
            if name not in self.cache:
                self.cache[name] = {
                    "m": torch.zeros(param.shape, device=self.device),
                    "v": torch.zeros(param.shape, device=self.device)
                    + 10.0 ** self.params["log_eps"].data
                    # NOTE that we add a little ‘fudge factor' here because sqrt is not
                    # differentiable at exactly zero
                }
            #print(name, param.device)
            g = param.grad.detach()
            self.cache[name]["m"] = m = (
                beta1 * self.cache[name]["m"].detach() + (1.0 - beta1) * g
            )
            self.cache[name]["v"] = v = (
                beta2 * self.cache[name]["v"].detach() + (1.0 - beta2) * g * g
            )
            self.all_params_with_gradients.append(nn.Parameter(m)) #Risque de surcharger la memoire => Dict mieux ?
            self.all_params_with_gradients.append(nn.Parameter(v))
            m_hat = m / (1.0 - beta1 ** float(t))
            v_hat = v / (1.0 - beta2 ** float(t))
            dparam = m_hat / (v_hat ** 0.5 + 10.0 ** self.params["log_eps"])
            updated_param[name] = param.detach() - self.params["alpha"] * dparam

        if pytorch_mod: params.update(updated_param) #Changer nom d'input...
        else: params = updated_param

    def __str__(self):
        return "adam(" + str(self.params) + ") / " + str(self.optimizer)
