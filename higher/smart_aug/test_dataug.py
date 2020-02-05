""" Script to run experiment on smart augmentation.

"""
import sys
from LeNet import *
from dataug import *
#from utils import *
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

    #'TranslateXabs',
    #'TranslateYabs',

    ## Color TF (Expect image in the range of [0, 1]) ##
    'Contrast',
    'Color',
    'Brightness',
    'Sharpness',
    'Posterize',
    'Solarize',

    #Color TF (Common mag scale)
    #'+Contrast',
    #'+Color',
    #'+Brightness',
    #'+Sharpness',
    #'-Contrast',
    #'-Color',
    #'-Brightness',
    #'-Sharpness',
    #'=Posterize',
    #'=Solarize',

    ## Bad Tranformations ##
    # Bad Geometric TF #
    #'BShearX',
    #'BShearY',
    #'BTranslateX-', 
    #'BTranslateX-',
    #'BTranslateY',
    #'BTranslateY-',

    #'BadContrast',
    #'BadBrightness', 

    #'Random',
    #'RandBlend'
]


device = torch.device('cuda') #Select device to use

if device == torch.device('cpu'):
    device_name = 'CPU'
else:
    device_name = torch.cuda.get_device_name(device)

torch.backends.cudnn.benchmark = True #Faster if same input size #Not recommended for reproductibility

#Increase reproductibility
torch.manual_seed(0)
np.random.seed(0)

##########################################
if __name__ == "__main__":

    #Task to perform
    tasks={
        #'classic',
        'aug_model'
    }
    #Parameters
    n_inner_iter = 1
    epochs = 150
    dataug_epoch_start=0
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
    #import torchvision.models as models
    #model=models.resnet18()
    model_name = str(model) #'wide_resnet50_2' #'resnet18' #str(model)
    #model = getattr(models.resnet, model_name)(pretrained=False)

    #### Classic ####
    if 'classic' in tasks:
        t0 = time.perf_counter()
        model = model.to(device)


        print("{} on {} for {} epochs".format(model_name, device_name, epochs))
        log= train_classic(model=model, opt_param=optim_param, epochs=epochs, print_freq=1)
        #log= train_classic_higher(model=model, epochs=epochs)

        exec_time=time.perf_counter() - t0
        max_cached = torch.cuda.max_memory_cached()/(1024.0 * 1024.0) #torch.cuda.max_memory_reserved()
        ####
        print('-'*9)
        times = [x["time"] for x in log]
        out = {"Accuracy": max([x["acc"] for x in log]), 
            "Time": (np.mean(times),np.std(times), exec_time), 
            'Optimizer': optim_param['Inner'], 
            "Device": device_name, 
            "Memory": max_cached, 
            "Log": log}
        print(model_name,": acc", out["Accuracy"], "in:", out["Time"][0], "+/-", out["Time"][1])
        filename = "{}-{} epochs".format(model_name,epochs)
        with open("../res/log/%s.json" % filename, "w+") as f:
            try:
                json.dump(out, f, indent=True)
                print('Log :\"',f.name, '\" saved !')
            except:
                print("Failed to save logs :",f.name)
                print(sys.exc_info()[1])

        try:
            plot_resV2(log, fig_name="../res/"+filename)
        except:
            print("Failed to plot res")
            print(sys.exc_info()[1])

        print('Execution Time (s): %.00f '%(exec_time))
        print('-'*9)

    #### Augmented Model ####
    if 'aug_model' in tasks:
        torch.cuda.reset_max_memory_cached() #reset_peak_stats
        t0 = time.perf_counter()

        tf_dict = {k: TF.TF_dict[k] for k in tf_names}
        model = Higher_model(model, model_name) #run_dist_dataugV3
        aug_model = Augmented_model(Data_augV5(TF_dict=tf_dict, N_TF=1, mix_dist=0.5, fixed_prob=False, fixed_mag=False, shared_mag=False), model).to(device)
        #aug_model = Augmented_model(RandAug(TF_dict=tf_dict, N_TF=2), model).to(device)

        print("{} on {} for {} epochs - {} inner_it".format(str(aug_model), device_name, epochs, n_inner_iter))
        log= run_dist_dataugV3(model=aug_model,
             epochs=epochs, 
             inner_it=n_inner_iter, 
             dataug_epoch_start=dataug_epoch_start, 
             opt_param=optim_param,
             print_freq=1, 
             unsup_loss=1, 
             hp_opt=False,
             save_sample_freq=1)

        exec_time=time.perf_counter() - t0
        max_cached = torch.cuda.max_memory_cached()/(1024.0 * 1024.0) #torch.cuda.max_memory_reserved() #MB
        ####
        print('-'*9)
        times = [x["time"] for x in log]
        out = {"Accuracy": max([x["acc"] for x in log]), 
            "Time": (np.mean(times),np.std(times), exec_time), 
            'Optimizer': optim_param, 
            "Device": device_name, 
            "Memory": max_cached, 
            "Param_names": aug_model.TF_names(), 
            "Log": log}
        print(str(aug_model),": acc", out["Accuracy"], "in:", out["Time"][0], "+/-", out["Time"][1])
        filename = "{}-{} epochs (dataug:{})- {} in_it".format(str(aug_model),epochs,dataug_epoch_start,n_inner_iter)+"(CV0.1)"
        with open("../res/log/%s.json" % filename, "w+") as f:
            try:
                json.dump(out, f, indent=True)
                print('Log :\"',f.name, '\" saved !')
            except:
                print("Failed to save logs :",f.name)
                print(sys.exc_info()[1])
        try:
            plot_resV2(log, fig_name="../res/"+filename, param_names=aug_model.TF_names())
        except:
            print("Failed to plot res")
            print(sys.exc_info()[1])

        print('Execution Time (s): %.00f '%(exec_time))
        print('-'*9)