""" Script to run experiment on smart augmentation.

"""
import sys
from dataug import *
#from utils import *
from train_utils import *
from transformations import TF_loader
# from arg_parser import *

TF_loader=TF_loader()

torch.backends.cudnn.benchmark = True #Faster if same input size #Not recommended for reproductibility

#Increase reproductibility
torch.manual_seed(0)
np.random.seed(0)

##########################################
if __name__ == "__main__":

    args = parser.parse_args()
    print(args)

    res_folder=args.res_folder
    postfix=args.postfix

    if args.dtype == 'FP32':
        def_type=torch.float32
    elif args.dtype == 'FP16':
        # def_type=torch.float16 #Default : float32
        def_type=torch.bfloat16
    else:
        raise Exception('dtype not supported :', args.dtype)
    torch.set_default_dtype(def_type) #Default : float32


    device = torch.device(args.device) #Select device to use
    if device == torch.device('cpu'):
        device_name = 'CPU'
    else:
        device_name = torch.cuda.get_device_name(device)

    #Parameters
    n_inner_iter = args.K
    epochs = args.epochs
    dataug_epoch_start=0
    Nb_TF_seq= args.N
    optim_param={
        'Meta':{
            'optim':'Adam',
            'lr':args.mlr,
            'epoch_start': args.meta_epoch_start, #0 / 2 (Resnet?)
            'reg_factor': args.mag_reg,
            'scheduler': None, #None, 'multiStep'
        },
        'Inner':{
            'optim': 'SGD',
            'lr':args.lr, #1e-2/1e-1 (ResNet)
            'momentum':args.momentum, #0.9
            'weight_decay':args.decay, #0.0005
            'nesterov':args.nesterov, #False (True: Bad behavior w/ Data_aug)
            'scheduler': args.scheduler, #None, 'cosine', 'multiStep', 'exponential'
            'warmup':{
                'multiplier': args.warmup, #2 #+ batch_size => + mutliplier #No warmup = 0
                'epochs': 5
            }
        }
    }

    #Info params
    F1=True
    sample_save=None
    print_f= epochs/4

    #Load network
    model, model_name= load_model(args.model, num_classes=len(dl_train.dataset.classes), pretrained=args.pretrained)

    #### Classic ####
    if not args.augment:
        if device_name != 'CPU':
            torch.cuda.reset_max_memory_allocated() #reset_peak_stats
            torch.cuda.reset_max_memory_cached() #reset_peak_stats
        t0 = time.perf_counter()

        model = model.to(device)


        print("{} on {} for {} epochs{}".format(model_name, device_name, epochs, postfix))
        #print("RandAugment(N{}-M{:.2f})-{} on {} for {} epochs{}".format(rand_aug['N'],rand_aug['M'],model_name, device_name, epochs, postfix))
        log= train_classic(model=model, opt_param=optim_param, epochs=epochs, print_freq=print_f)
        #log= train_classic_higher(model=model, epochs=epochs)

        exec_time=time.perf_counter() - t0
        
        if device_name != 'CPU':
            max_allocated = torch.cuda.max_memory_allocated()/(1024.0 * 1024.0)
            max_cached = torch.cuda.max_memory_cached()/(1024.0 * 1024.0) #torch.cuda.max_memory_reserved() #MB
        else:
            max_allocated = 0.0
            max_cached=0.0
        ####
        print('-'*9)
        times = [x["time"] for x in log]
        out = {"Accuracy": max([x["acc"] for x in log]), 
            "Time": (np.mean(times),np.std(times), exec_time), 
            'Optimizer': optim_param['Inner'], 
            "Device": device_name, 
            "Memory": [max_allocated, max_cached], 
            #"Rand_Aug": rand_aug, 
            "Log": log}
        print(model_name,": acc", out["Accuracy"], "in:", out["Time"][0], "+/-", out["Time"][1])
        filename = "{}-{} epochs".format(model_name,epochs)+postfix
        #print("RandAugment-",model_name,": acc", out["Accuracy"], "in:", out["Time"][0], "+/-", out["Time"][1])
        #filename = "RandAugment(N{}-M{:.2f})-{}-{} epochs".format(rand_aug['N'],rand_aug['M'],model_name,epochs)+postfix
        with open(res_folder+"log/%s.json" % filename, "w+") as f:
            try:
                json.dump(out, f, indent=True)
                print('Log :\"',f.name, '\" saved !')
            except:
                print("Failed to save logs :",f.name)
                print(sys.exc_info()[1])

        try:
            plot_resV2(log, fig_name=res_folder+filename, f1=F1)
        except:
            print("Failed to plot res")
            print(sys.exc_info()[1])

        print('Execution Time (s): %.00f '%(exec_time))
        print('-'*9)

    #### Augmented Model ####
    else:
        # tf_config='../config/invScale_wide_tf_config.json'#'../config/invScale_wide_tf_config.json'#'../config/base_tf_config.json'
        tf_dict, tf_ignore_mag =TF_loader.load_TF_dict(args.tf_config)

        if device_name != 'CPU':
            torch.cuda.reset_max_memory_allocated() #reset_peak_stats
            torch.cuda.reset_max_memory_cached() #reset_peak_stats
        t0 = time.perf_counter()

        model = Higher_model(model, model_name) #run_dist_dataugV3
        dataug_mod = 'Data_augV8' if args.learn_seq else 'Data_augV5'
        if n_inner_iter !=0:
            aug_model = Augmented_model(
                globals()[dataug_mod](TF_dict=tf_dict, 
                    N_TF=Nb_TF_seq, 
                    temp=args.temp, 
                    fixed_prob=False, 
                    fixed_mag=args.fixed_mag, 
                    shared_mag=args.shared_mag, 
                    TF_ignore_mag=tf_ignore_mag), model).to(device)
        else:
            aug_model = Augmented_model(RandAug(TF_dict=tf_dict, N_TF=Nb_TF_seq), model).to(device)

        print("{} on {} for {} epochs - {} inner_it{}".format(str(aug_model), device_name, epochs, n_inner_iter, postfix))
        log, aug_acc = run_dist_dataugV3(model=aug_model,
             epochs=epochs, 
             inner_it=n_inner_iter, 
             dataug_epoch_start=dataug_epoch_start, 
             opt_param=optim_param,
             unsup_loss=1, 
             augment_loss=args.augment_loss,
             hp_opt=False, #False #['lr', 'momentum', 'weight_decay']
             print_freq=print_f, 
             save_sample_freq=sample_save)

        exec_time=time.perf_counter() - t0
        if device_name != 'CPU':
            max_allocated = torch.cuda.max_memory_allocated()/(1024.0 * 1024.0)
            max_cached = torch.cuda.max_memory_cached()/(1024.0 * 1024.0) #torch.cuda.max_memory_reserved() #MB
        else:
            max_allocated = 0.0
            max_cached = 0.0
        ####
        print('-'*9)
        times = [x["time"] for x in log]
        out = {"Accuracy": max([x["acc"] for x in log]), 
            "Aug_Accuracy": [args.augment_loss, aug_acc],
            "Time": (np.mean(times),np.std(times), exec_time), 
            'Optimizer': optim_param, 
            "Device": device_name, 
            "Memory": [max_allocated, max_cached], 
            "TF_config": args.tf_config,
            "Param_names": aug_model.TF_names(), 
            "Log": log}
        print(str(aug_model),": acc", out["Accuracy"], "/ aug_acc", out["Aug_Accuracy"][1] , "in:", out["Time"][0], "+/-", out["Time"][1])
        filename = "{}-{}_epochs-{}_in_it-AL{}".format(str(aug_model),epochs,n_inner_iter,args.augment_loss)+postfix
        with open(res_folder+"log/%s.json" % filename, "w+") as f:
            try:
                json.dump(out, f, indent=True)
                print('Log :\"',f.name, '\" saved !')
            except:
                print("Failed to save logs :",f.name)
                print(sys.exc_info()[1])
        try:
            plot_resV2(log, fig_name=res_folder+filename, param_names=aug_model.TF_names(), f1=F1)
        except:
            print("Failed to plot res")
            print(sys.exc_info()[1])

        print('Execution Time (s): %.00f '%(exec_time))
        print('-'*9)