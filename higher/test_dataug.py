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

    #Non fonctionnel
    #'Auto_Contrast', #Pas opti pour des batch (Super lent)
    #'Equalize',
]

device = torch.device('cuda')

if device == torch.device('cpu'):
    device_name = 'CPU'
else:
    device_name = torch.cuda.get_device_name(device)

##########################################
if __name__ == "__main__":

    n_inner_iter = 10
    epochs = 2
    dataug_epoch_start=0

    #### Classic ####
    '''
    #model = LeNet(3,10).to(device)
    model = WideResNet(num_classes=10, wrn_size=16).to(device)
    #model = Augmented_model(Data_augV3(mix_dist=0.0), LeNet(3,10)).to(device)
    #model.augment(mode=False)

    print(str(model), 'on', device_name)
    log= train_classic(model=model, epochs=epochs)
    #log= train_classic_higher(model=model, epochs=epochs)

    ####
    plot_res(log, fig_name="res/{}-{} epochs".format(str(model),epochs))
    print('-'*9)
    times = [x["time"] for x in log]
    out = {"Accuracy": max([x["acc"] for x in log]), "Time": (np.mean(times),np.std(times)), "Device": device_name, "Log": log}
    print(str(model),": acc", out["Accuracy"], "in (ms):", out["Time"][0], "+/-", out["Time"][1])
    with open("res/log/%s.json" % "{}-{} epochs".format(str(model),epochs), "w+") as f:
        json.dump(out, f, indent=True)
        print('Log :\"',f.name, '\" saved !')
    print('-'*9)
    '''
    #### Augmented Model ####
    #'''
    t0 = time.process_time()
    tf_dict = {k: TF.TF_dict[k] for k in tf_names}
    #tf_dict = TF.TF_dict
    aug_model = Augmented_model(Data_augV5(TF_dict=tf_dict, N_TF=2, mix_dist=0.5), LeNet(3,10)).to(device)
    #aug_model = Augmented_model(Data_augV4(TF_dict=tf_dict, N_TF=2, mix_dist=0.0), WideResNet(num_classes=10, wrn_size=160)).to(device)
    print(str(aug_model), 'on', device_name)
    #run_simple_dataug(inner_it=n_inner_iter, epochs=epochs)
    log= run_dist_dataugV2(model=aug_model, epochs=epochs, inner_it=n_inner_iter, dataug_epoch_start=dataug_epoch_start, print_freq=1, loss_patience=10)

    ####
    plot_resV2(log, fig_name="res/{}-{} epochs (dataug:{})- {} in_it".format(str(aug_model),epochs,dataug_epoch_start,n_inner_iter), param_names=tf_names)
    print('-'*9)
    times = [x["time"] for x in log]
    out = {"Accuracy": max([x["acc"] for x in log]), "Time": (np.mean(times),np.std(times)), "Device": device_name, "Param_names": aug_model.TF_names(), "Log": log}
    print(str(aug_model),": acc", out["Accuracy"], "in (s?):", out["Time"][0], "+/-", out["Time"][1])
    with open("res/log/%s.json" % "{}-{} epochs (dataug:{})- {} in_it".format(str(aug_model),epochs,dataug_epoch_start,n_inner_iter), "w+") as f:
        json.dump(out, f, indent=True)
        print('Log :\"',f.name, '\" saved !')

    print('Execution Time : %.00f (s?)'%(time.process_time() - t0))
    print('-'*9)
    #'''
    #### TF number tests ####
    '''
    res_folder="res/TF_nb_tests/"
    epochs= 100
    inner_its = [0, 1, 10]
    dist_mix = [0.0, 0.5]
    dataug_epoch_starts= [0]
    TF_nb = [len(TF.TF_dict)] #range(10,len(TF.TF_dict)+1) #[len(TF.TF_dict)]
    N_seq_TF= [2, 3, 4, 6]
    
    try:
        os.mkdir(res_folder)
        os.mkdir(res_folder+"log/")
    except FileExistsError:
        pass

    for n_inner_iter in inner_its:
        print("---Starting inner_it", n_inner_iter,"---")
        for dataug_epoch_start in dataug_epoch_starts:
            print("---Starting dataug", dataug_epoch_start,"---")
            for n_tf in N_seq_TF:
                for i in TF_nb:
                    keys = list(TF.TF_dict.keys())[0:i]
                    ntf_dict = {k: TF.TF_dict[k] for k in keys}

                    aug_model = Augmented_model(Data_augV4(TF_dict=ntf_dict, N_TF=n_tf, mix_dist=0.0), LeNet(3,10)).to(device)
                    print(str(aug_model), 'on', device_name)
                    #run_simple_dataug(inner_it=n_inner_iter, epochs=epochs)
                    log= run_dist_dataugV2(model=aug_model, epochs=epochs, inner_it=n_inner_iter, dataug_epoch_start=dataug_epoch_start, print_freq=10, loss_patience=None)

                    ####
                    plot_res(log, fig_name=res_folder+"{}-{} epochs (dataug:{})- {} in_it".format(str(aug_model),epochs,dataug_epoch_start,n_inner_iter), param_names=keys)
                    print('-'*9)
                    times = [x["time"] for x in log]
                    out = {"Accuracy": max([x["acc"] for x in log]), "Time": (np.mean(times),np.std(times)), "Device": device_name, "Param_names": aug_model.TF_names(), "Log": log}
                    print(str(aug_model),": acc", out["Accuracy"], "in (s?):", out["Time"][0], "+/-", out["Time"][1])
                    with open(res_folder+"log/%s.json" % "{}-{} epochs (dataug:{})- {} in_it".format(str(aug_model),epochs,dataug_epoch_start,n_inner_iter), "w+") as f:
                        json.dump(out, f, indent=True)
                        print('Log :\"',f.name, '\" saved !')
                    print('-'*9)

    '''    