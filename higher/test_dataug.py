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

    #'BRotate',
    #'BTranslateX',
    #'BTranslateY',
    #'BShearX',
    #'BShearY',
    #'BadTranslateX',
    #'BadTranslateX_neg',
    #'BadTranslateY',
    #'BadTranslateY_neg',

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
    epochs = 100
    dataug_epoch_start=0

    #### Classic ####
    '''
    model = LeNet(3,10).to(device)
    #model = WideResNet(num_classes=10, wrn_size=16).to(device)
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
    print(str(model),": acc", out["Accuracy"], "in:", out["Time"][0], "+/-", out["Time"][1])
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
    aug_model = Augmented_model(Data_augV5(TF_dict=tf_dict, N_TF=1, mix_dist=0.0, fixed_prob=True, fixed_mag=False, shared_mag=False), LeNet(3,10)).to(device)
    #aug_model = Augmented_model(Data_augV5(TF_dict=tf_dict, N_TF=2, mix_dist=0.5, fixed_mag=True, shared_mag=True), WideResNet(num_classes=10, wrn_size=160)).to(device)
    print(str(aug_model), 'on', device_name)
    #run_simple_dataug(inner_it=n_inner_iter, epochs=epochs)
    log= run_dist_dataugV2(model=aug_model, epochs=epochs, inner_it=n_inner_iter, dataug_epoch_start=dataug_epoch_start, print_freq=10, loss_patience=None)

    ####
    print('-'*9)
    times = [x["time"] for x in log]
    out = {"Accuracy": max([x["acc"] for x in log]), "Time": (np.mean(times),np.std(times)), "Device": device_name, "Param_names": aug_model.TF_names(), "Log": log}
    print(str(aug_model),": acc", out["Accuracy"], "in:", out["Time"][0], "+/-", out["Time"][1])
    filename = "{}-{} epochs (dataug:{})- {} in_it".format(str(aug_model),epochs,dataug_epoch_start,n_inner_iter)
    with open("res/log/%s.json" % filename, "w+") as f:
        json.dump(out, f, indent=True)
        print('Log :\"',f.name, '\" saved !')

    plot_resV2(log, fig_name="res/"+filename, param_names=tf_names)

    print('Execution Time : %.00f '%(time.process_time() - t0))
    print('-'*9)
    #'''
    #### TF tests ####
    #'''
    res_folder="res/brutus-tests/"
    epochs= 150
    inner_its = [1, 10]
    dist_mix = [0.0, 0.5, 1]
    dataug_epoch_starts= [0]
    tf_dict = {k: TF.TF_dict[k] for k in tf_names}
    TF_nb = [len(tf_dict)] #range(10,len(TF.TF_dict)+1) #[len(TF.TF_dict)]
    N_seq_TF= [1, 2, 3, 4]
    mag_setup = [(True,True), (False, False)]
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
                        for run in range(nb_run):
                            if n_inner_iter == 0 and (m_setup!=(True,True) or p_setup!=True): continue #Autres setup inutiles sans meta-opti
                            if n_inner_iter ==1 and (n_tf==1 or n_tf==2): continue #Deja resultats
                            #keys = list(TF.TF_dict.keys())[0:i]
                            #ntf_dict = {k: TF.TF_dict[k] for k in keys}

                            aug_model = Augmented_model(Data_augV5(TF_dict=tf_dict, N_TF=n_tf, mix_dist=dist, fixed_prob=False, fixed_mag=m_setup[0], shared_mag=m_setup[1]), LeNet(3,10)).to(device)
                            print(str(aug_model), 'on', device_name)
                            #run_simple_dataug(inner_it=n_inner_iter, epochs=epochs)
                            log= run_dist_dataugV2(model=aug_model, epochs=epochs, inner_it=n_inner_iter, dataug_epoch_start=dataug_epoch_start, print_freq=20, loss_patience=None)

                            ####
                            print('-'*9)
                            times = [x["time"] for x in log]
                            out = {"Accuracy": max([x["acc"] for x in log]), "Time": (np.mean(times),np.std(times)), "Device": device_name, "Param_names": aug_model.TF_names(), "Log": log}
                            print(str(aug_model),": acc", out["Accuracy"], "in :", out["Time"][0], "+/-", out["Time"][1])
                            filename = "{}-{}epochs(dataug:{})-{}in_it-{}".format(str(aug_model),epochs,dataug_epoch_start,n_inner_iter,run)
                            with open(res_folder+"log/%s.json" % filename, "w+") as f:
                                json.dump(out, f, indent=True)
                                print('Log :\"',f.name, '\" saved !')

                            #plot_resV2(log, fig_name=res_folder+filename, param_names=tf_names)
                            print('-'*9)

    #'''    