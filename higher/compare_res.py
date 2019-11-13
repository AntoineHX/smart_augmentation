from utils import *

if __name__ == "__main__":

    #### Comparison ####

    ## Loss , Acc, Proba = f(epoch) ##
    files=[
        #"res/log/LeNet-100 epochs.json",
        #"res/log/Aug_mod(Data_augV4(Uniform-4 TF)-LeNet)-100 epochs (dataug:0)- 0 in_it.json",
        #"res/log/Aug_mod(Data_augV4(Uniform-4 TF)-LeNet)-100 epochs (dataug:50)- 0 in_it.json",
        #"res/log/Aug_mod(Data_augV4(Uniform-3 TF)-LeNet)-100 epochs (dataug:0)- 0 in_it.json",
        #"res/log/Aug_mod(Data_augV3(Uniform-3 TF)-LeNet)-100 epochs (dataug:50)- 10 in_it.json",
        #"res/log/Aug_mod(Data_augV4(Mix 0,5-3 TF)-LeNet)-100 epochs (dataug:0)- 1 in_it.json",
        #"res/log/Aug_mod(Data_augV4(Mix 0.5-3 TF)-LeNet)-100 epochs (dataug:50)- 10 in_it.json",
        #"res/log/Aug_mod(Data_augV4(Uniform-3 TF)-LeNet)-100 epochs (dataug:0)- 10 in_it.json",
        #"res/log/Aug_mod(Data_augV4(Uniform-10 TF)-LeNet)-100 epochs (dataug:50)- 10 in_it.json",
        #"res/log/Aug_mod(Data_augV4(Uniform-10 TF)-LeNet)-100 epochs (dataug:50)- 0 in_it.json",
    ]
    #plot_compare(filenames=files, fig_name="res/compare")

    ## Acc, Time, Epochs = f(n_tf) ##
    fig_name="res/TF_seq_tests_compare"
    inner_its = [10]
    dataug_epoch_starts= [0]
    TF_nb = 14 #range(1,14+1)
    N_seq_TF= [1, 2, 3, 4]

    fig, ax = plt.subplots(ncols=3, figsize=(30, 8))
    for in_it in inner_its:
        for dataug in dataug_epoch_starts:
            #filenames =["res/TF_nb_tests/log/Aug_mod(Data_augV4(Uniform-{} TF)-LeNet)-200 epochs (dataug:{})- {} in_it.json".format(n_tf, dataug, in_it) for n_tf in TF_nb]
            filenames =["res/TF_nb_tests/log/Aug_mod(Data_augV4(Uniform-{} TF x {})-LeNet)-200 epochs (dataug:{})- {} in_it.json".format(TF_nb, n_tf, dataug, in_it) for n_tf in N_seq_TF]
            filenames =["res/TF_nb_tests/log/Aug_mod(Data_augV4(Uniform-{} TF x {})-LeNet)-100 epochs (dataug:{})- {} in_it.json".format(TF_nb, n_tf, dataug, in_it) for n_tf in N_seq_TF]


            all_data=[]
            #legend=""
            for idx, file in enumerate(filenames):
                #legend+=str(idx)+'-'+file+'\n'
                with open(file) as json_file:
                    data = json.load(json_file)
                    all_data.append(data)

            n_tf = N_seq_TF
            #n_tf = [len(x["Param_names"]) for x in all_data]
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