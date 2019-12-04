from utils import *

if __name__ == "__main__":

    '''
    files=[
        #"res/good_TF_tests/log/Aug_mod(Data_augV5(Mix0.5-14TFx2-MagFxSh)-LeNet)-100 epochs (dataug:0)- 0 in_it.json",
        #"res/good_TF_tests/log/Aug_mod(Data_augV5(Uniform-14TFx2-MagFxSh)-LeNet)-100 epochs (dataug:0)- 0 in_it.json",
        "res/brutus-tests/log/Aug_mod(Data_augV5(Uniform-14TFx2-MagFxSh)-LeNet)-150epochs(dataug:0)-0in_it-0.json",
        #"res/log/Aug_mod(RandAugUDA(18TFx2-Mag1)-LeNet)-100 epochs (dataug:0)- 0 in_it.json",
    ]

    for idx, file in enumerate(files):
        #legend+=str(idx)+'-'+file+'\n'
        with open(file) as json_file:
            data = json.load(json_file)
            plot_resV2(data['Log'], fig_name=file.replace('.json','').replace('log/',''), param_names=data['Param_names'])
            #plot_TF_influence(data['Log'], param_names=data['Param_names'])
    '''
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

    #Res print
    #'''
    nb_run=3
    accs = []
    times = []
    files = ["res/brutus-tests/log/Aug_mod(Data_augV5(Mix1.0-18TFx3-MagFxSh)-LeNet)-150epochs(dataug:0)-1in_it-%s.json"%str(run) for run in range(nb_run)]
    
    for idx, file in enumerate(files):
        #legend+=str(idx)+'-'+file+'\n'
        with open(file) as json_file:
            data = json.load(json_file)
        accs.append(data['Accuracy'])
        times.append(data['Time'][0])

    print(files[0], np.mean(accs), np.std(accs), np.mean(times))
    #'''