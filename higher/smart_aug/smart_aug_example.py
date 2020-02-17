""" Example use of smart augmentation.

"""

from LeNet import *
from dataug import *
from train_utils import *

tf_config='../config/base_tf_config.json'
TF_loader=TF_loader()

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
            'lr':1e-2, #1e-2/1e-1 (ResNet)
            'momentum':0.9, #0.9
            'decay':0.0005, #0.0005
            'nesterov':False, #False (True: Bad behavior w/ Data_aug)
            'scheduler':'cosine', #None, 'cosine', 'multiStep', 'exponential'
        }
    }

    #Models
    model = LeNet(3,10)

    #Smart_aug initialisation
    tf_dict, tf_ignore_mag =TF_loader.load_TF_dict(tf_config)
    model = Higher_model(model) #run_dist_dataugV3
    aug_model = Augmented_model(
        Data_augV5(TF_dict=tf_dict, 
            N_TF=3, 
            mix_dist=0.8, 
            fixed_prob=False, 
            fixed_mag=False, 
            shared_mag=False,
            TF_ignore_mag=tf_ignore_mag), 
        model).to(device)

    print("{} on {} for {} epochs - {} inner_it".format(str(aug_model), device_name, epochs, n_inner_iter))

    # Training
    trained_model = run_simple_smartaug(model=aug_model, epochs=epochs, inner_it=n_inner_iter, opt_param=optim_param)
