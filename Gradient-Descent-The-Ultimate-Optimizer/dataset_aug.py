import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

class MNIST_aug(Dataset):

    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']
               
    def __init__(self):
        self.images = [TF.to_pil_image(x) for x in torch.ByteTensor(10, 3, 48, 48)]
        self.set_stage(0) # initial stage
        
    def __getitem__(self, index):
        image = self.images[index]
        
        # Just apply your transformations here
        image = self.crop(image)
        x = TF.to_tensor(image)
        return x
        
    def set_stage(self, stage):
        if stage == 0:
            print('Using (32, 32) crops')
            self.crop = transforms.RandomCrop((32, 32))
        elif stage == 1:
            print('Using (28, 28) crops')
            self.crop = transforms.RandomCrop((28, 28))
        
    def __len__(self):
        return len(self.images)


dataset = MyData()
loader = DataLoader(dataset,
                    batch_size=2,
                    num_workers=2,
                    shuffle=True)

for batch_idx, data in enumerate(loader):
    print('Batch idx {}, data shape {}'.format(
        batch_idx, data.shape))
    
loader.dataset.set_stage(1)

for batch_idx, data in enumerate(loader):
    print('Batch idx {}, data shape {}'.format(
        batch_idx, data.shape))

