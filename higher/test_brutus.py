from model import *
from dataug import *
#from utils import *
from train_utils import *

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

device = torch.device('cuda')

if device == torch.device('cpu'):
    device_name = 'CPU'
else:
    device_name = torch.cuda.get_device_name(device)

##########################################
if __name__ == "__main__":


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
            'lr':1e-1, #1e-2
            'momentum':0.9, #0.9
        }
    }

    #model = LeNet(3,10)
    model = ResNet(num_classes=10)
    #model = MobileNetV2(num_classes=10)
    #model = WideResNet(num_classes=10, wrn_size=32)

    tf_dict = {k: TF.TF_dict[k] for k in tf_names}

    ####
    '''
    t0 = time.process_time()

    aug_model = Augmented_model(RandAug(TF_dict=tf_dict, N_TF=2), model).to(device)

    print("{} on {} for {} epochs - {} inner_it".format(str(aug_model), device_name, epochs, n_inner_iter))
    log= run_dist_dataugV2(model=aug_model, epochs=epochs, inner_it=n_inner_iter, dataug_epoch_start=dataug_epoch_start, print_freq=10, KLdiv=True, loss_patience=None)

    exec_time=time.process_time() - t0
    ####
    times = [x["time"] for x in log]
    out = {"Accuracy": max([x["acc"] for x in log]), "Time": (np.mean(times),np.std(times), exec_time), "Device": device_name, "Param_names": aug_model.TF_names(), "Log": log}
    filename = "{}-{} epochs (dataug:{})- {} in_it".format(str(aug_model),epochs,dataug_epoch_start,n_inner_iter)
    with open("res/log/%s.json" % filename, "w+") as f:
        json.dump(out, f, indent=True)
        print('Log :\"',f.name, '\" saved !')
    '''

    ####
    '''
    t0 = time.process_time()

    aug_model = Augmented_model(Data_augV5(TF_dict=tf_dict, N_TF=3, mix_dist=0.0, fixed_prob=False, fixed_mag=False, shared_mag=False), model).to(device)
    
    print("{} on {} for {} epochs - {} inner_it".format(str(aug_model), device_name, epochs, n_inner_iter))
    log= run_dist_dataugV2(model=aug_model, epochs=epochs, inner_it=n_inner_iter, dataug_epoch_start=dataug_epoch_start, print_freq=10, KLdiv=True, loss_patience=None)

    exec_time=time.process_time() - t0
    ####
    times = [x["time"] for x in log]
    out = {"Accuracy": max([x["acc"] for x in log]), "Time": (np.mean(times),np.std(times), exec_time), "Device": device_name, "Param_names": aug_model.TF_names(), "Log": log}
    filename = "{}-{} epochs (dataug:{})- {} in_it".format(str(aug_model),epochs,dataug_epoch_start,n_inner_iter)
    with open("res/log/%s.json" % filename, "w+") as f:
        json.dump(out, f, indent=True)
        print('Log :\"',f.name, '\" saved !')
    '''
    res_folder="res/brutus-tests2/"
    epochs= 150
    inner_its = [1]
    dist_mix = [0.0, 0.5, 0.8, 1.0]
    dataug_epoch_starts= [0]
    tf_dict = {k: TF.TF_dict[k] for k in tf_names}
    TF_nb = [len(tf_dict)] #range(10,len(TF.TF_dict)+1) #[len(TF.TF_dict)]
    N_seq_TF= [4, 3, 2]
    mag_setup = [(True,True), (False, False)] #(Fixed, Shared)
    #prob_setup = [True, False]
    nb_run= 3
    
    try:
        os.mkdir(res_folder)
        os.mkdir(res_folder+"log/")
    except FileExistsError:
        pass

    for n_inner_iter in inner_its:
        for dataug_epoch_start in dataug_epoch_starts:
            for n_tf in N_seq_TF:
                for dist in dist_mix:
                    #for i in TF_nb:
                    for m_setup in mag_setup:
                        #for p_setup in prob_setup:
                        p_setup=False
                        for run in range(nb_run):
                            if (n_inner_iter == 0 and (m_setup!=(True,True) and p_setup!=True)) or (p_setup and dist!=0.0): continue #Autres setup inutiles sans meta-opti
                            #keys = list(TF.TF_dict.keys())[0:i]
                            #ntf_dict = {k: TF.TF_dict[k] for k in keys}

                            t0 = time.process_time()

                            aug_model = Augmented_model(Data_augV5(TF_dict=tf_dict, N_TF=n_tf, mix_dist=dist, fixed_prob=p_setup, fixed_mag=m_setup[0], shared_mag=m_setup[1]), model).to(device)
                            #aug_model = Augmented_model(RandAug(TF_dict=tf_dict, N_TF=2), model).to(device)

                            print("{} on {} for {} epochs - {} inner_it".format(str(aug_model), device_name, epochs, n_inner_iter))
                            log= run_dist_dataugV3(model=aug_model,
                                 epochs=epochs, 
                                 inner_it=n_inner_iter, 
                                 dataug_epoch_start=dataug_epoch_start, 
                                 opt_param=optim_param,
                                 print_freq=50, 
                                 KLdiv=True, 
                                 loss_patience=None)

                            exec_time=time.process_time() - t0
                            ####
                            print('-'*9)
                            times = [x["time"] for x in log]
                            out = {"Accuracy": max([x["acc"] for x in log]), "Time": (np.mean(times),np.std(times), exec_time), 'Optimizer': optim_param, "Device": device_name, "Param_names": aug_model.TF_names(), "Log": log}
                            print(str(aug_model),": acc", out["Accuracy"], "in:", out["Time"][0], "+/-", out["Time"][1])
                            filename = "{}-{} epochs (dataug:{})- {} in_it-{}".format(str(aug_model),epochs,dataug_epoch_start,n_inner_iter, run)
                            with open("res/log/%s.json" % filename, "w+") as f:
                                try:
                                    json.dump(out, f, indent=True)
                                    print('Log :\"',f.name, '\" saved !')
                                except:
                                    print("Failed to save logs :",f.name)
                            try:
                                plot_resV2(log, fig_name="res/"+filename, param_names=aug_model.TF_names())
                            except:
                                print("Failed to plot res")

                            print('Execution Time : %.00f '%(exec_time))
                            print('-'*9)
                        #'''
