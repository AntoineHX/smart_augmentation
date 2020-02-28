import torch
import numpy as np
import torchvision
from PIL import Image
from torch import topk
import torch.nn.functional as F
from torch import topk
import cv2
from torchvision import transforms
import os

class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = ((output.cpu()).data).numpy()
    def remove(self): self.hook.remove()

def getCAM(feature_conv, weight_fc, class_idx):
    _, nc, h, w = feature_conv.shape
    cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    # cam_img = np.uint8(255 * cam_img)
    return cam_img

def main(cam):
    device = 'cuda:0'
    model_name = 'resnet50'
    root = 'NEW_SS'
    
    os.makedirs(os.path.join(root + '_CAM', 'OK'), exist_ok=True)
    os.makedirs(os.path.join(root + '_CAM', 'NOK'), exist_ok=True)

    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = torchvision.datasets.ImageFolder(
        root=root, transform=train_transform,
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    model = torchvision.models.__dict__[model_name](pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)

    model.load_state_dict(torch.load('checkpoint.pt', map_location=lambda storage, loc: storage))
    model = model.to(device)
    model.eval()

    weight_softmax_params = list(model._modules.get('fc').parameters())
    weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())

    final_layer = model._modules.get('layer4')

    activated_features = SaveFeatures(final_layer)

    for i, (img, target ) in enumerate(loader):
        img = img.to(device)
        prediction = model(img)
        pred_probabilities = F.softmax(prediction, dim=1).data.squeeze()
        class_idx = topk(pred_probabilities,1)[1].int()
        # if target.item() != class_idx:
        #     print(dataset.imgs[i][0])

        if cam:
            overlay = getCAM(activated_features.features, weight_softmax, class_idx )

            import ipdb; ipdb.set_trace()
            import PIL 
            from torchvision.transforms import ToPILImage

            img = ToPILImage()(overlay).resize(size=(1280, 1024), resample=PIL.Image.BILINEAR)
            img.save('heat-pil.jpg')


            img = cv2.imread(dataset.imgs[i][0])
            height, width, _ = img.shape
            overlay = cv2.resize(overlay, (width, height))
            heatmap = cv2.applyColorMap(overlay, cv2.COLORMAP_JET)
            cv2.imwrite('heat-cv2.jpg', heatmap)

            img = cv2.imread(dataset.imgs[i][0])
            height, width, _ = img.shape
            overlay = cv2.resize(overlay, (width, height))
            heatmap = cv2.applyColorMap(overlay, cv2.COLORMAP_JET)
            result = heatmap * 0.3 + img * 0.5

            clss = dataset.imgs[i][0].split(os.sep)[1]
            name = dataset.imgs[i][0].split(os.sep)[2].split('.')[0]
            cv2.imwrite(os.path.join(root+"_CAM", clss, name + '.jpg'), result)
            print(f'{os.path.join(root+"_CAM", clss, name + ".jpg")} saved')
    
    activated_features.remove()

if __name__ == "__main__":
    main(cam=True)
