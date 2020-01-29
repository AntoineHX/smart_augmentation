""" Example use of smart augmentation.

"""

from model import *
from dataug import *
from train_utils import *

# Use available TF (see transformations.py)
tf_names = [
    ## Geometric TF ##
    'Identity',
    'FlipUD',
    'FlipLR',
    'Rotate',
    'TranslateX',
    'TranslateY',
    'ShearX',
    'ShearY',

    ## Color TF (Expect image in the range of [0, 1]) ##
    'Contrast',
    'Color',
    'Brightness',
    'Sharpness',
    'Posterize',
    'Solarize', #=>Image entre [0,1] #Pas opti pour des batch
]


device = torch.device('cuda') #Select device to use

if device == torch.device('cpu'):
    device_name = 'CPU'
else:
    device_name = torch.cuda.get_device_name(device)

##########################################
if __name__ == "__main__":

    #Parameters
    n_inner_iter = 1
    epochs = 150
    optim_param={
        'Meta':{
            'optim':'Adam',
            'lr':1e-2, #1e-2
        },
        'Inner':{
            'optim': 'SGD',
            'lr':1e-2, #1e-2
            'momentum':0.9, #0.9
        }
    }

    #Models
    model = LeNet(3,10)
    #model = ResNet(num_classes=10)
    #model = MobileNetV2(num_classes=10)
    #model = WideResNet(num_classes=10, wrn_size=32)

    #Smart_aug initialisation
    tf_dict = {k: TF.TF_dict[k] for k in tf_names}
    model = Higher_model(model) #run_dist_dataugV3
    aug_model = Augmented_model(
        Data_augV5(TF_dict=tf_dict, 
            N_TF=3, 
            mix_dist=0.8, 
            fixed_prob=False, 
            fixed_mag=False, 
            shared_mag=False), 
        model).to(device)

    print("{} on {} for {} epochs - {} inner_it".format(str(aug_model), device_name, epochs, n_inner_iter))

    # Training
    trained_model = run_simple_smartaug(model=aug_model, epochs=epochs, inner_it=n_inner_iter, opt_param=optim_param)
