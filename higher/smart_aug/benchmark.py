""" Script to run series of experiments.

"""
from dataug import *
#from utils import *
from train_utils import *

import torchvision.models as models

model_list={models.resnet: ['resnet18', 'resnet50','wide_resnet50_2']} #lr=0.1

optim_param={
    'Meta':{
        'optim':'Adam',
        'lr':1e-2, #1e-2
    },
    'Inner':{
        'optim': 'SGD',
        'lr':1e-1, #1e-2/1e-1 (ResNet)
        'momentum':0.9, #0.9
        'decay':0.0005, #0.0005
        'nesterov':True,
        'scheduler':'cosine', #None, 'cosine', 'multiStep', 'exponential'
    }
}

res_folder="../res/benchmark/CIFAR10/"
#res_folder="../res/HPsearch/"
epochs= 200
dataug_epoch_start=0
nb_run= 3

# Use available TF (see transformations.py)
'''
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
tf_dict = {k: TF.TF_dict[k] for k in tf_names}
'''
tf_config='../config/base_tf_config.json'
TF_loader=TF_loader()
tf_dict, tf_ignore_mag =TF_loader.load_TF_dict(tf_config)

device = torch.device('cuda')

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

    ### Benchmark ###
    #'''
    n_inner_iter = 1
    dist_mix = [0.5]#[0.5, 1.0]
    N_seq_TF= [3, 4]
    mag_setup = [(False, False)] #[(True, True), (False, False)] #(FxSh, Independant)

    for model_type in model_list.keys():
            for model_name in model_list[model_type]:
                for run in range(nb_run):

                     for n_tf in N_seq_TF:
                        for dist in dist_mix:
                            for m_setup in mag_setup:

                                torch.cuda.reset_max_memory_allocated() #reset_peak_stats
                                torch.cuda.reset_max_memory_cached() #reset_peak_stats
                                t0 = time.perf_counter()

                                model = getattr(model_type, model_name)(pretrained=False, num_classes=len(dl_train.dataset.classes))

                                model = Higher_model(model, model_name) #run_dist_dataugV3
                                if n_inner_iter!=0:
                                    aug_model = Augmented_model(
                                        Data_augV5(TF_dict=tf_dict, 
                                            N_TF=n_tf, 
                                            mix_dist=dist, 
                                            fixed_prob=False, 
                                            fixed_mag=m_setup[0], 
                                            shared_mag=m_setup[1]), 
                                        model).to(device)
                                else:
                                    aug_model = Augmented_model(RandAug(TF_dict=tf_dict, N_TF=n_tf), model).to(device)

                                print("{} on {} for {} epochs - {} inner_it".format(str(aug_model), device_name, epochs, n_inner_iter))
                                log= run_dist_dataugV3(model=aug_model,
                                     epochs=epochs, 
                                     inner_it=n_inner_iter, 
                                     dataug_epoch_start=dataug_epoch_start, 
                                     opt_param=optim_param,
                                     print_freq=epochs/4, 
                                     unsup_loss=1, 
                                     hp_opt=False,
                                     save_sample_freq=None)

                                exec_time=time.perf_counter() - t0
                                max_allocated = torch.cuda.max_memory_allocated()/(1024.0 * 1024.0)
                                max_cached = torch.cuda.max_memory_cached()/(1024.0 * 1024.0) #torch.cuda.max_memory_reserved() #MB
                                ####
                                print('-'*9)
                                times = [x["time"] for x in log]
                                out = {"Accuracy": max([x["acc"] for x in log]), 
                                    "Time": (np.mean(times),np.std(times), exec_time), 
                                    'Optimizer': optim_param, 
                                    "Device": device_name, 
                                    "Memory": [max_allocated, max_cached], 
                                    "TF_config": tf_config,
                                    "Param_names": aug_model.TF_names(), 
                                    "Log": log}
                                print(str(aug_model),": acc", out["Accuracy"], "in:", out["Time"][0], "+/-", out["Time"][1])
                                filename = "{}-{} epochs (dataug:{})- {} in_it-{}".format(str(aug_model),epochs,dataug_epoch_start,n_inner_iter, run)
                                with open(res_folder+"log/%s.json" % filename, "w+") as f:
                                    try:
                                        json.dump(out, f, indent=True)
                                        print('Log :\"',f.name, '\" saved !')
                                    except:
                                        print("Failed to save logs :",f.name)

                                print('Execution Time : %.00f '%(exec_time))
                                print('-'*9)
    #'''
    ### Benchmark - RandAugment/Vanilla ###
    '''
    for model_type in model_list.keys():
            for model_name in model_list[model_type]:
                for run in range(nb_run):
                    torch.cuda.reset_max_memory_allocated() #reset_peak_stats
                    torch.cuda.reset_max_memory_cached() #reset_peak_stats
                    t0 = time.perf_counter()

                    model = getattr(model_type, model_name)(pretrained=False, num_classes=len(dl_train.dataset.classes)).to(device)

                    print("{} on {} for {} epochs".format(model_name, device_name, epochs))
                    #print("RandAugment(N{}-M{:.2f})-{} on {} for {} epochs".format(rand_aug['N'],rand_aug['M'],model_name, device_name, epochs))
                    log= train_classic(model=model, opt_param=optim_param, epochs=epochs, print_freq=epochs/4)

                    exec_time=time.perf_counter() - t0
                    max_allocated = torch.cuda.max_memory_allocated()/(1024.0 * 1024.0)
                    max_cached = torch.cuda.max_memory_cached()/(1024.0 * 1024.0) #torch.cuda.max_memory_reserved() #MB
                    ####
                    print('-'*9)
                    times = [x["time"] for x in log]
                    out = {"Accuracy": max([x["acc"] for x in log]), 
                        "Time": (np.mean(times),np.std(times), exec_time), 
                        'Optimizer': optim_param, 
                        "Device": device_name, 
                        "Memory": [max_allocated, max_cached],
                        #"Rand_Aug": rand_aug, 
                        "Log": log}
                    print(model_name,": acc", out["Accuracy"], "in:", out["Time"][0], "+/-", out["Time"][1])
                    filename = "{}-{} epochs -{}".format(model_name,epochs, run)
                    #print("RandAugment-",model_name,": acc", out["Accuracy"], "in:", out["Time"][0], "+/-", out["Time"][1])
                    #filename = "RandAugment(N{}-M{:.2f})-{}-{} epochs -{}".format(rand_aug['N'],rand_aug['M'],model_name,epochs, run)
                    with open(res_folder+"log/%s.json" % filename, "w+") as f:
                        try:
                            json.dump(out, f, indent=True)
                            print('Log :\"',f.name, '\" saved !')
                        except:
                            print("Failed to save logs :",f.name)
                            print(sys.exc_info()[1])

                    #plot_resV2(log, fig_name=res_folder+filename)

                    print('Execution Time : %.00f '%(exec_time))
                    print('-'*9)
    '''
    ### HP Search ###
    '''
    from LeNet import *
    inner_its = [1]
    dist_mix = [1.0]#[0.0, 0.5, 0.8, 1.0]
    N_seq_TF= [5, 6]
    mag_setup = [(True, True), (False, False)] #(FxSh, Independant)
    #prob_setup = [True, False]
    
    try:
        os.mkdir(res_folder)
        os.mkdir(res_folder+"log/")
    except FileExistsError:
        pass

    for n_inner_iter in inner_its:
            for n_tf in N_seq_TF:
                for dist in dist_mix:
                    #for i in TF_nb:
                    for m_setup in mag_setup:
                        #for p_setup in prob_setup:
                        p_setup=False
                        for run in range(nb_run):

                            t0 = time.perf_counter()

                            model = getattr(models.resnet, 'resnet18')(pretrained=False, num_classes=len(dl_train.dataset.classes))
                            #model = LeNet(3,10)
                            model = Higher_model(model) #run_dist_dataugV3
                            aug_model = Augmented_model(Data_augV5(TF_dict=tf_dict, N_TF=n_tf, mix_dist=dist, fixed_prob=p_setup, fixed_mag=m_setup[0], shared_mag=m_setup[1]), model).to(device)
                            #aug_model = Augmented_model(RandAug(TF_dict=tf_dict, N_TF=2), model).to(device)

                            print("{} on {} for {} epochs - {} inner_it".format(str(aug_model), device_name, epochs, n_inner_iter))
                            log= run_dist_dataugV3(model=aug_model,
                                 epochs=epochs, 
                                 inner_it=n_inner_iter, 
                                 dataug_epoch_start=dataug_epoch_start, 
                                 opt_param=optim_param,
                                 print_freq=epochs/4, 
                                 unsup_loss=1, 
                                 hp_opt=False,
                                 save_sample_freq=None)

                            exec_time=time.perf_counter() - t0
                            ####
                            print('-'*9)
                            times = [x["time"] for x in log]
                            out = {"Accuracy": max([x["acc"] for x in log]), "Time": (np.mean(times),np.std(times), exec_time), 'Optimizer': optim_param, "Device": device_name, "Param_names": aug_model.TF_names(), "Log": log}
                            print(str(aug_model),": acc", out["Accuracy"], "in:", out["Time"][0], "+/-", out["Time"][1])
                            filename = "{}-{} epochs (dataug:{})- {} in_it-{}".format(str(aug_model),epochs,dataug_epoch_start,n_inner_iter, run)
                            with open(res_folder+"log/%s.json" % filename, "w+") as f:
                                try:
                                    json.dump(out, f, indent=True)
                                    print('Log :\"',f.name, '\" saved !')
                                except:
                                    print("Failed to save logs :",f.name)

                            print('Execution Time : %.00f '%(exec_time))
                            print('-'*9)
    '''
