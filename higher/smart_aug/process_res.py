from utils import *

if __name__ == "__main__":

    '''
    # files=[
    #     "../res/log/Aug_mod(Data_augV5(Mix0.5-18TFx3-Mag)-efficientnet-b1)-200 epochs (dataug 0)- 1 in_it__AL2.json",
    #     #"res/brutus-tests/log/Aug_mod(Data_augV5(Uniform-14TFx3-MagFxSh)-LeNet)-150epochs(dataug:0)-10in_it-0.json",
    #     #"res/brutus-tests/log/Aug_mod(Data_augV5(Uniform-14TFx3-MagFxSh)-LeNet)-150epochs(dataug:0)-10in_it-1.json",
    #     #"res/brutus-tests/log/Aug_mod(Data_augV5(Uniform-14TFx3-MagFxSh)-LeNet)-150epochs(dataug:0)-10in_it-2.json",
    #     #"res/log/Aug_mod(RandAugUDA(18TFx2-Mag1)-LeNet)-100 epochs (dataug:0)- 0 in_it.json",
    # ]
    files = ["../res/benchmark/CIFAR10/log/RandAugment(N%d-M%.2f)-%s-200 epochs -%s.json"%(3,1,'resnet18', str(run)) for run in range(3)]
    #files = ["../res/benchmark/CIFAR10/log/Aug_mod(RandAug(18TFx%d-Mag%d)-%s)-200 epochs (dataug:0)- 0 in_it-%s.json"%(2,1,'resnet18', str(run)) for run in range(3)]
    #files = ["../res/benchmark/CIFAR10/log/Aug_mod(Data_augV5(Mix%.1f-18TFx%d-Mag)-%s)-200 epochs (dataug:0)- 1 in_it-%s.json"%(0.5,3,'resnet18', str(run)) for run in range(3)]


    for idx, file in enumerate(files):
        #legend+=str(idx)+'-'+file+'\n'
        with open(file) as json_file:
            data = json.load(json_file)
            plot_resV2(data['Log'], fig_name=file.replace("/log","").replace(".json",""), param_names=data['Param_names'], f1=True)
            #plot_TF_influence(data['Log'], param_names=data['Param_names'])
    '''

    #Res print
    # '''
    nb_run=3
    accs = []
    aug_accs = []
    f1_max = []
    f1_min = []
    times = []
    mem = []

    files = ["../res/benchmark/log/Aug_mod(Data_augV5(T0.5-19TFx3-Mag)-resnet18)-200 epochs (dataug 0)- 1 in_it__%s.json"%(str(run)) for run in range(1, nb_run+1)]

    for idx, file in enumerate(files):
        #legend+=str(idx)+'-'+file+'\n'
        with open(file) as json_file:
            data = json.load(json_file)
        accs.append(data['Accuracy'])
        aug_accs.append(data['Aug_Accuracy'][1])
        times.append(data['Time'][0])
        mem.append(data['Memory'][1])

        acc_idx = [x['acc'] for x in data['Log']].index(data['Accuracy'])
        f1_max.append(max(data['Log'][acc_idx]['f1'])*100)
        f1_min.append(min(data['Log'][acc_idx]['f1'])*100)
        print(idx, accs[-1], aug_accs[-1])

    print(files[0])
    print("Acc : %.2f ~ %.2f / Aug_Acc %d: %.2f ~ %.2f"%(np.mean(accs), np.std(accs), data['Aug_Accuracy'][0], np.mean(aug_accs), np.std(aug_accs)))
    print("F1 max : %.2f ~ %.2f / F1 min : %.2f ~ %.2f"%(np.mean(f1_max), np.std(f1_max), np.mean(f1_min), np.std(f1_min)))
    print("Time (h): %.2f ~ %.2f"%(np.mean(times)/3600, np.std(times)/3600))
    print("Mem (MB): %d ~ %d"%(np.mean(mem), np.std(mem)))
    # '''

    ## Loss , Acc, Proba = f(epoch) ##
    #plot_compare(filenames=files, fig_name="res/compare")

    '''
    ## Acc, Time, Epochs = f(n_tf) ##
    #fig_name="res/TF_nb_tests_compare"
    fig_name="res/TF_seq_tests_compare"
    inner_its = [0, 10]
    dataug_epoch_starts= [0]
    TF_nb = 14#[len(TF.TF_dict)] #range(10,len(TF.TF_dict)+1) #[len(TF.TF_dict)]
    N_seq_TF= [1, 2, 3, 4, 6] #[1]

    fig, ax = plt.subplots(ncols=3, figsize=(30, 8))
    for in_it in inner_its:
        for dataug in dataug_epoch_starts:

            #n_tf = TF_nb
            #filenames =["res/TF_nb_tests/log/Aug_mod(Data_augV4(Uniform-{} TF)-LeNet)-200 epochs (dataug:{})- {} in_it.json".format(n_tf, dataug, in_it) for n_tf in TF_nb]
            #filenames =["res/TF_nb_tests/log/Aug_mod(Data_augV4(Uniform-{} TF x {})-LeNet)-200 epochs (dataug:{})- {} in_it.json".format(n_tf, 1, dataug, in_it) for n_tf in TF_nb]

            n_tf = N_seq_TF
            #filenames =["res/TF_nb_tests/log/Aug_mod(Data_augV4(Uniform-{} TF x {})-LeNet)-200 epochs (dataug:{})- {} in_it.json".format(TF_nb, n_tf, dataug, in_it) for n_tf in N_seq_TF]
            filenames =["res/TF_nb_tests/log/Aug_mod(Data_augV4(Uniform-{} TF x {})-LeNet)-200 epochs (dataug:{})- {} in_it.json".format(TF_nb, n_tf, dataug, in_it) for n_tf in N_seq_TF]


            all_data=[]
            #legend=""
            for idx, file in enumerate(filenames):
                #legend+=str(idx)+'-'+file+'\n'
                with open(file) as json_file:
                    data = json.load(json_file)
                    all_data.append(data)
            
            acc = [x["Accuracy"] for x in all_data]
            epochs = [len(x["Log"]) for x in all_data]
            time = [x["Time"][0] for x in all_data]
            #for i in range(len(time)): time[i] *= epochs[i] #Estimation temps total

            ax[0].plot(n_tf, acc, label="{} in_it/{} dataug".format(in_it,dataug))
            ax[1].plot(n_tf, time, label="{} in_it/{} dataug".format(in_it,dataug))
            ax[2].plot(n_tf, epochs, label="{} in_it/{} dataug".format(in_it,dataug))


            #for data in all_data:
                #print(np.mean([x["param"] for x in data["Log"]], axis=0))
                #print(len(data["Param_names"]), np.argsort(np.argsort(np.mean([x["param"] for x in data["Log"]], axis=0))))


    ax[0].set_title('Acc')
    ax[1].set_title('Time')
    ax[2].set_title('Epochs')
    for a in ax: a.legend()

    fig_name = fig_name.replace('.',',')
    plt.savefig(fig_name, bbox_inches='tight')
    plt.close()
    '''



    '''
    #HP search
    inner_its = [1]
    dist_mix = [0.3, 0.5, 0.8, 1.0] #Uniform
    N_seq_TF= [3]
    nb_run= 3

    for n_inner_iter in inner_its:
            for n_tf in N_seq_TF:
                for dist in dist_mix:

                    files = ["../res/HPsearch/log/Aug_mod(Data_augV5(Mix%.1f-14TFx%d-Mag)-ResNet)-200 epochs (dataug:0)- 1 in_it-%s.json"%(dist, n_tf, str(run)) for run in range(nb_run)]
                    #files = ["../res/HPsearch/log/Aug_mod(Data_augV5(Uniform-14TFx%d-MagFxSh)-ResNet)-200 epochs (dataug:0)- 1 in_it-%s.json"%(n_tf, str(run)) for run in range(nb_run)]
                    accs = []
                    times = []
                    for idx, file in enumerate(files):
                        #legend+=str(idx)+'-'+file+'\n'
                        with open(file) as json_file:
                            data = json.load(json_file)
                        accs.append(data['Accuracy'])
                        times.append(data['Time'][0])
                        print(idx, data['Accuracy'])

                    print(files[0], 'acc', np.mean(accs), '+-',np.std(accs), ',t', np.mean(times))
    '''

    '''
    #Benchmark
    model_list=['resnet18', 'resnet50','wide_resnet50_2']
    nb_run= 3

    for model_name in model_list:

        files = ["../res/benchmark/CIFAR100/log/RandAugment(N%d-M%.2f)-%s-200 epochs -%s.json"%(3,0.17,model_name, str(run)) for run in range(nb_run)]
        #files = ["../res/benchmark/CIFAR10/log/%s-200 epochs -%s.json"%(model_name, str(run)) for run in range(nb_run)]

        accs = []
        times = []
        mem_alloc = []
        mem_cach = []
        for idx, file in enumerate(files):
            #legend+=str(idx)+'-'+file+'\n'
            with open(file) as json_file:
                data = json.load(json_file)
            accs.append(data['Accuracy'])
            times.append(data['Time'][0])
            mem_cach.append(data['Memory'])
            print(idx, data['Accuracy'])

        print(files[0], 'acc', np.mean(accs), '+-',np.std(accs), ',t', np.mean(times), 'Mem', np.mean(mem_cach))
    '''