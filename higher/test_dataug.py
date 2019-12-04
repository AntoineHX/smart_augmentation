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

    #'BRotate',
    #'BTranslateX',
    #'BTranslateY',
    #'BShearX',
    #'BShearY',
    #'BadTranslateX',
    #'BadTranslateX_neg',
    #'BadTranslateY',
    #'BadTranslateY_neg',

    #'BadColor',
    #'BadSharpness',
    #'BadContrast',
    #'BadBrightness',

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

    tasks={
        #'classic',
        'aug_dataset',
        #'aug_model'
    }
    n_inner_iter = 1
    epochs = 100
    dataug_epoch_start=0


    #### Classic ####
    if 'classic' in tasks:
        t0 = time.process_time()
        model = LeNet(3,10).to(device)
        #model = WideResNet(num_classes=10, wrn_size=16).to(device)
        #model = Augmented_model(Data_augV3(mix_dist=0.0), LeNet(3,10)).to(device)
        #model.augment(mode=False)

        print(str(model), 'on', device_name)
        log= train_classic(model=model, epochs=epochs)
        #log= train_classic_higher(model=model, epochs=epochs)

        ####
        print('-'*9)
        times = [x["time"] for x in log]
        out = {"Accuracy": max([x["acc"] for x in log]), "Time": (np.mean(times),np.std(times)), "Device": device_name, "Log": log}
        print(str(model),": acc", out["Accuracy"], "in:", out["Time"][0], "+/-", out["Time"][1])
        filename = "{}-{} epochs".format(str(model),epochs)
        with open("res/log/%s.json" % filename, "w+") as f:
            json.dump(out, f, indent=True)
            print('Log :\"',f.name, '\" saved !')

        plot_res(log, fig_name="res/"+filename)

        print('Execution Time : %.00f '%(time.process_time() - t0))
        print('-'*9)
    

    #### Augmented Dataset ####
    if 'aug_dataset' in tasks:

        xs, ys = next(iter(dl_train))
        viz_sample_data(imgs=xs, labels=ys, fig_name='samples/data_sample_{}'.format(str(data_train_aug)))
        t0 = time.process_time()
        model = LeNet(3,10).to(device)
        #model = WideResNet(num_classes=10, wrn_size=16).to(device)
        #model = Augmented_model(Data_augV3(mix_dist=0.0), LeNet(3,10)).to(device)
        #model.augment(mode=False)

        print(str(model), 'on', device_name)
        log= train_classic(model=model, epochs=epochs)
        #log= train_classic_higher(model=model, epochs=epochs)

        ####
        print('-'*9)
        times = [x["time"] for x in log]
        out = {"Accuracy": max([x["acc"] for x in log]), "Time": (np.mean(times),np.std(times)), "Device": device_name, "Log": log}
        print(str(model),": acc", out["Accuracy"], "in:", out["Time"][0], "+/-", out["Time"][1])
        filename = "{}-{}-{} epochs".format(str(data_train_aug),str(model),epochs)
        with open("res/log/%s.json" % filename, "w+") as f:
            json.dump(out, f, indent=True)
            print('Log :\"',f.name, '\" saved !')

        plot_res(log, fig_name="res/"+filename)

        print('Execution Time : %.00f '%(time.process_time() - t0))
        print('-'*9)
    

    #### Augmented Model ####
    if 'aug_model' in tasks:
        t0 = time.process_time()

        tf_dict = {k: TF.TF_dict[k] for k in tf_names}

        #aug_model = Augmented_model(Data_augV6(TF_dict=tf_dict, N_TF=1, mix_dist=0.0, fixed_prob=False, prob_set_size=2, fixed_mag=True, shared_mag=True), LeNet(3,10)).to(device)
        aug_model = Augmented_model(Data_augV5(TF_dict=tf_dict, N_TF=3, mix_dist=0.0, fixed_prob=False, fixed_mag=False, shared_mag=False), LeNet(3,10)).to(device)
        #aug_model = Augmented_model(Data_augV5(TF_dict=tf_dict, N_TF=2, mix_dist=0.5, fixed_mag=True, shared_mag=True), WideResNet(num_classes=10, wrn_size=160)).to(device)
        #aug_model = Augmented_model(RandAug(TF_dict=tf_dict, N_TF=2), LeNet(3,10)).to(device)
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