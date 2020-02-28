import torch
import numpy as np
import torchvision
from PIL import Image
from torch import topk
from torch import nn
import torch.nn.functional as F
from torch import topk
import cv2
from torchvision import transforms
import os

class Lambda(nn.Module):
    "Create a layer that simply calls `func` with `x`"
    def __init__(self, func): 
        super().__init__()
        self.func=func
    def forward(self, x): return self.func(x)

class SaveFeatures():
    activations, gradients = None, None
    def __init__(self, m): 
        self.forward = m.register_forward_hook(self.forward_hook_fn)
        self.backward = m.register_backward_hook(self.backward_hook_fn)

    def forward_hook_fn(self, module, input, output): 
        self.activations = output.cpu().detach()

    def backward_hook_fn(self, module, grad_input, grad_output): 
        self.gradients = grad_output[0].cpu().detach()

    def remove(self): 
        self.forward.remove()
        self.backward.remove()

def main(cam):
    device = 'cuda:0'
    model_name = 'resnet50'
    root = '/mnt/md0/data/cifar10/tmp/cifar/train'
    _root = 'cifar'
    
    os.makedirs(os.path.join(_root + '_CAM'), exist_ok=True)
    os.makedirs(os.path.join(_root + '_CAM'), exist_ok=True)

    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = torchvision.datasets.ImageFolder(
        root=root, transform=train_transform,
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    model = torchvision.models.__dict__[model_name](pretrained=True)
    flat = list(model.children())
    body, head = nn.Sequential(*flat[:-2]), nn.Sequential(flat[-2], Lambda(func=lambda x: torch.flatten(x, 1)), nn.Linear(flat[-1].in_features, len(loader.dataset.classes))) 
    model = nn.Sequential(body, head)

    model.load_state_dict(torch.load('checkpoint.pt', map_location=lambda storage, loc: storage))
    model = model.to(device)
    model.eval()

    activated_features = SaveFeatures(model[0])

    for i, (img, target ) in enumerate(loader):
        img = img.to(device)
        pred = model(img)
        import ipdb; ipdb.set_trace()
        # get the gradient of the output with respect to the parameters of the model
        pred[:, target.item()].backward()

        # import ipdb; ipdb.set_trace()
        # pull the gradients out of the model
        gradients = activated_features.gradients[0]

        pooled_gradients = gradients.mean(1).mean(1)

        # get the activations of the last convolutional layer
        activations = activated_features.activations[0]

        heatmap = F.relu(((activations*pooled_gradients[...,None,None])).sum(0))
        heatmap /= torch.max(heatmap)

        heatmap = heatmap.numpy()

        
        image = cv2.imread(dataset.imgs[i][0])
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        # superimposed_img = heatmap * 0.3 + image * 0.5
        superimposed_img = heatmap 

        clss = dataset.imgs[i][0].split(os.sep)[1]
        name = dataset.imgs[i][0].split(os.sep)[2].split('.')[0]
        cv2.imwrite(os.path.join(_root+"_CAM", name + '.jpg'), superimposed_img)
        print(f'{os.path.join(_root+"_CAM", name + ".jpg")} saved')
    
    activated_features.remove()

if __name__ == "__main__":
    main(cam=True)
