from model import *
from dataug import *
#from utils import *
from train_utils import *

tf_names = [
    ## Geometric TF ##
    'Identity',
    #'FlipUD',
    #'FlipLR',
    #'Rotate',
    #'TranslateX',
    #'TranslateY',
    #'ShearX',
    #'ShearY',

    ## Color TF (Expect image in the range of [0, 1]) ##
    #'Contrast',
    #'Color',
    #'Brightness',
    #'Sharpness',
    #'Posterize',
    #'Solarize', #=>Image entre [0,1] #Pas opti pour des batch

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

    'Random',
    #'RandBlend'
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
        #'aug_dataset',
        'aug_model'
    }
    n_inner_iter = 1
    epochs = 1
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

    model = LeNet(3,10)
    #model = MobileNetV2(num_classes=10)
    #model = ResNet(num_classes=10)
    #model = WideResNet(num_classes=10, wrn_size=32)

    #### Classic ####
    if 'classic' in tasks:
        t0 = time.process_time()
        model = model.to(device)

        print("{} on {} for {} epochs".format(str(model), device_name, epochs))
        #log= train_classic(model=model, opt_param=optim_param, epochs=epochs, print_freq=10)
        log= train_classic_higher(model=model, epochs=epochs)

        exec_time=time.process_time() - t0
        ####
        print('-'*9)
        times = [x["time"] for x in log]
        out = {"Accuracy": max([x["acc"] for x in log]), "Time": (np.mean(times),np.std(times), exec_time), 'Optimizer': optim_param['Inner'], "Device": device_name, "Log": log}
        print(str(model),": acc", out["Accuracy"], "in:", out["Time"][0], "+/-", out["Time"][1])
        filename = "{}-{} epochs".format(str(model),epochs)
        with open("res/log/%s.json" % filename, "w+") as f:
            json.dump(out, f, indent=True)
            print('Log :\"',f.name, '\" saved !')

        plot_res(log, fig_name="res/"+filename)

        print('Execution Time : %.00f '%(exec_time))
        print('-'*9)
    

    #### Augmented Dataset ####
    if 'aug_dataset' in tasks:

        t0 = time.process_time()

        #data_train_aug = AugmentedDataset("./data", train=True, download=download_data, transform=transform, subset=(0,int(len(data_train)/2)))
        #data_train_aug.augement_data(aug_copy=30)
        #print(data_train_aug)
        #dl_train = torch.utils.data.DataLoader(data_train_aug, batch_size=BATCH_SIZE, shuffle=True)

        #xs, ys = next(iter(dl_train))
        #viz_sample_data(imgs=xs, labels=ys, fig_name='samples/data_sample_{}'.format(str(data_train_aug)))

        #model = model.to(device)

        #print("{} on {} for {} epochs".format(str(model), device_name, epochs))
        #log= train_classic(model=model, epochs=epochs, print_freq=10)
        ##log= train_classic_higher(model=model, epochs=epochs)

        data_train_aug = AugmentedDatasetV2("./data", train=True, download=download_data, transform=transform, subset=(0,int(len(data_train)/2)))
        data_train_aug.augement_data(aug_copy=1)
        print(data_train_aug)
        unsup_ratio = 5
        dl_unsup = torch.utils.data.DataLoader(data_train_aug, batch_size=BATCH_SIZE*unsup_ratio, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

        unsup_xs, sup_xs, ys = next(iter(dl_unsup))
        viz_sample_data(imgs=sup_xs, labels=ys, fig_name='samples/data_sample_{}'.format(str(data_train_aug)))
        viz_sample_data(imgs=unsup_xs, labels=ys, fig_name='samples/data_sample_{}_unsup'.format(str(data_train_aug)))

        model = model.to(device)

        print("{} on {} for {} epochs".format(str(model), device_name, epochs))
        log= train_UDA(model=model, dl_unsup=dl_unsup, epochs=epochs, opt_param=optim_param, print_freq=10)

        exec_time=time.process_time() - t0
        ####
        print('-'*9)
        times = [x["time"] for x in log]
        out = {"Accuracy": max([x["acc"] for x in log]), "Time": (np.mean(times),np.std(times), exec_time), 'Optimizer': optim_param['Inner'], "Device": device_name, "Log": log}
        print(str(model),": acc", out["Accuracy"], "in:", out["Time"][0], "+/-", out["Time"][1])
        filename = "{}-{}-{} epochs".format(str(data_train_aug),str(model),epochs)
        with open("res/log/%s.json" % filename, "w+") as f:
            json.dump(out, f, indent=True)
            print('Log :\"',f.name, '\" saved !')

        plot_res(log, fig_name="res/"+filename)

        print('Execution Time : %.00f '%(exec_time))
        print('-'*9)
    

    #### Augmented Model ####
    if 'aug_model' in tasks:
        t0 = time.process_time()

        tf_dict = {k: TF.TF_dict[k] for k in tf_names}
        #aug_model = Augmented_model(Data_augV6(TF_dict=tf_dict, N_TF=1, mix_dist=0.0, fixed_prob=False, prob_set_size=2, fixed_mag=True, shared_mag=True), model).to(device)
        aug_model = Augmented_model(Data_augV5(TF_dict=tf_dict, N_TF=1, mix_dist=0.5, fixed_prob=False, fixed_mag=True, shared_mag=True), model).to(device)
        #aug_model = Augmented_model(RandAug(TF_dict=tf_dict, N_TF=2), model).to(device)

        print("{} on {} for {} epochs - {} inner_it".format(str(aug_model), device_name, epochs, n_inner_iter))
        log= run_dist_dataugV2(model=aug_model,
             epochs=epochs, 
             inner_it=n_inner_iter, 
             dataug_epoch_start=dataug_epoch_start, 
             opt_param=optim_param,
             print_freq=1, 
             KLdiv=True, 
             loss_patience=None)

        exec_time=time.process_time() - t0
        ####
        print('-'*9)
        times = [x["time"] for x in log]
        out = {"Accuracy": max([x["acc"] for x in log]), "Time": (np.mean(times),np.std(times), exec_time), 'Optimizer': optim_param, "Device": device_name, "Log": log}
        print(str(aug_model),": acc", out["Accuracy"], "in:", out["Time"][0], "+/-", out["Time"][1])
        filename = "{}-{} epochs (dataug:{})- {} in_it".format(str(aug_model),epochs,dataug_epoch_start,n_inner_iter)
        with open("res/log/%s.json" % filename, "w+") as f:
            json.dump(out, f, indent=True)
            print('Log :\"',f.name, '\" saved !')

        plot_resV2(log, fig_name="res/"+filename, param_names=tf_names)

        print('Execution Time : %.00f '%(exec_time))
        print('-'*9)