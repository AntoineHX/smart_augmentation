import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import higher
import time

data_train = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=torchvision.transforms.ToTensor())
dl_train = torch.utils.data.DataLoader(data_train, batch_size=300, shuffle=True, num_workers=0, pin_memory=False)


class Aug_model(nn.Module):
    def __init__(self, model, hyper_param=True):
        super(Aug_model, self).__init__()

        #### Origin of the issue ? ####
        if hyper_param:
            self._params = nn.ParameterDict({
                    "hyper_param": nn.Parameter(torch.Tensor([0.5])),
                })
        ###############################

        self._mods = nn.ModuleDict({
            'model': model,
            })

    def forward(self, x):
        return self._mods['model'](x) #* self._params['hyper_param']

    def __getitem__(self, key):
        return self._mods[key]

class Aug_model2(nn.Module): #Slow increase like no hyper_param
    def __init__(self, model, hyper_param=True):
        super(Aug_model2, self).__init__()

        #### Origin of the issue ? ####
        if hyper_param:
            self._params = nn.ParameterDict({
                    "hyper_param": nn.Parameter(torch.Tensor([0.5])),
                })
        ###############################

        self._mods = nn.ModuleDict({
            'model': model,
            'fmodel': higher.patch.monkeypatch(model, device=None, copy_initial_weights=True)
            })

    def forward(self, x):
        return self._mods['fmodel'](x) * self._params['hyper_param']

    def get_diffopt(self, opt, track_higher_grads=True):
        return higher.optim.get_diff_optim(opt, 
            self._mods['model'].parameters(),
            fmodel=self._mods['fmodel'],
            track_higher_grads=track_higher_grads)

    def __getitem__(self, key):
        return self._mods[key]

if __name__ == "__main__":

    device = torch.device('cuda:1')
    aug_model = Aug_model2(
                    model=torch.hub.load('pytorch/vision:v0.4.2', 'resnet18', pretrained=False),
                    hyper_param=True #False will not extend step time
                    ).to(device)

    inner_opt = torch.optim.SGD(aug_model['model'].parameters(), lr=1e-2, momentum=0.9)

    #fmodel = higher.patch.monkeypatch(aug_model, device=None, copy_initial_weights=True)
    #diffopt = higher.optim.get_diff_optim(inner_opt, aug_model.parameters(),fmodel=fmodel,track_higher_grads=True)
    diffopt = aug_model.get_diffopt(inner_opt)

    for i, (xs, ys) in enumerate(dl_train):
        xs, ys = xs.to(device), ys.to(device)

        #logits = fmodel(xs)
        logits = aug_model(xs)
        loss = F.cross_entropy(F.log_softmax(logits, dim=1), ys, reduction='mean')

        t = time.process_time()
        diffopt.step(loss) #(opt.zero_grad, loss.backward, opt.step)
        #print(len(fmodel._fast_params),"step", time.process_time()-t)
        print(len(aug_model['fmodel']._fast_params),"step", time.process_time()-t)