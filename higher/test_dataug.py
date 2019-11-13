from torch.utils.data import SubsetRandomSampler
import torch.optim as optim
import torchvision
import higher

from model import *
from dataug import *
from utils import *

BATCH_SIZE = 300
#TEST_SIZE = 300
TEST_SIZE = 10000

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

#ATTENTION : Dataug (Kornia) Expect image in the range of [0, 1]
#transform_train = torchvision.transforms.Compose([
#    torchvision.transforms.RandomHorizontalFlip(),
#    torchvision.transforms.ToTensor(),
    #torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #CIFAR10
#])
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    #torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #CIFAR10
])
'''
data_train = torchvision.datasets.MNIST(
    "./data", train=True, download=True, 
    transform=torchvision.transforms.Compose([
            #torchvision.transforms.RandomAffine(degrees=180, translate=None, scale=None, shear=None, resample=False, fillcolor=0),
            torchvision.transforms.ToTensor()
        ])
)
data_test = torchvision.datasets.MNIST(
    "./data", train=False, download=True, transform=torchvision.transforms.ToTensor()
)
'''
data_train = torchvision.datasets.CIFAR10(
    "./data", train=True, download=True, transform=transform
)
#data_val = torchvision.datasets.CIFAR10(
#    "./data", train=True, download=True, transform=transform
#)
data_test = torchvision.datasets.CIFAR10(
    "./data", train=False, download=True, transform=transform
)
#'''
train_subset_indices=range(int(len(data_train)/2))
#train_subset_indices=range(BATCH_SIZE*10)
val_subset_indices=range(int(len(data_train)/2),len(data_train))

dl_train = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=False, sampler=SubsetRandomSampler(train_subset_indices))
dl_val = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=False, sampler=SubsetRandomSampler(val_subset_indices))
dl_test = torch.utils.data.DataLoader(data_test, batch_size=TEST_SIZE, shuffle=False)

device = torch.device('cuda')

if device == torch.device('cpu'):
    device_name = 'CPU'
else:
    device_name = torch.cuda.get_device_name(device)


def test(model):
    model.eval()
    for i, (features, labels) in enumerate(dl_test):
        features,labels = features.to(device), labels.to(device)

        pred = model.forward(features)
        return pred.argmax(dim=1).eq(labels).sum().item() / TEST_SIZE * 100

def compute_vaLoss(model, dl_val_it):
    try:
        xs_val, ys_val = next(dl_val_it)
    except StopIteration: #Fin epoch val
        dl_val_it = iter(dl_val)
        xs_val, ys_val = next(dl_val_it)
    xs_val, ys_val = xs_val.to(device), ys_val.to(device)

    try:
        model.augment(mode=False) #Validation sans transfornations !
    except:
        pass
    return F.cross_entropy(model(xs_val), ys_val)

def train_classic(model, epochs=1):
    #opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

    model.train()
    dl_val_it = iter(dl_val)
    log = []
    for epoch in range(epochs):
        #print_torch_mem("Start epoch")
        t0 = time.process_time()
        for i, (features, labels) in enumerate(dl_train):
            #print_torch_mem("Start iter")
            features,labels = features.to(device), labels.to(device)

            optim.zero_grad()
            pred = model.forward(features)
            loss = F.cross_entropy(pred,labels)
            loss.backward()
            optim.step()

        #### Tests ####
        tf = time.process_time()
        try:
            xs_val, ys_val = next(dl_val_it)
        except StopIteration: #Fin epoch val
            dl_val_it = iter(dl_val)
            xs_val, ys_val = next(dl_val_it)
        xs_val, ys_val = xs_val.to(device), ys_val.to(device)

        val_loss = F.cross_entropy(model(xs_val), ys_val)
        accuracy=test(model)
        model.train()
        #### Log ####
        data={
            "epoch": epoch,
            "train_loss": loss.item(),
            "val_loss": val_loss.item(),
            "acc": accuracy,
            "time": tf - t0,

            "param": None,
        }
        log.append(data)

    return log

def train_classic_higher(model, epochs=1):
    #opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

    model.train()
    dl_val_it = iter(dl_val)
    log = []

    fmodel = higher.patch.monkeypatch(model, device=None, copy_initial_weights=True)
    diffopt = higher.optim.get_diff_optim(optim, model.parameters(),fmodel=fmodel,track_higher_grads=False)
    #with higher.innerloop_ctx(model, optim, copy_initial_weights=True, track_higher_grads=False) as (fmodel, diffopt):

    for epoch in range(epochs):
        #print_torch_mem("Start epoch "+str(epoch))
        #print("Fast param ",len(fmodel._fast_params))
        t0 = time.process_time()
        for i, (features, labels) in enumerate(dl_train):
            #print_torch_mem("Start iter")
            features,labels = features.to(device), labels.to(device)

            #optim.zero_grad()
            pred = fmodel.forward(features)
            loss = F.cross_entropy(pred,labels)
            #.backward()
            #optim.step()
            diffopt.step(loss) #(opt.zero_grad, loss.backward, opt.step)

        model_copy(src=fmodel, dst=model, patch_copy=False)
        optim_copy(dopt=diffopt, opt=optim)
        fmodel = higher.patch.monkeypatch(model, device=None, copy_initial_weights=True)
        diffopt = higher.optim.get_diff_optim(optim, model.parameters(),fmodel=fmodel,track_higher_grads=False)

        #### Tests ####
        tf = time.process_time()
        try:
            xs_val, ys_val = next(dl_val_it)
        except StopIteration: #Fin epoch val
            dl_val_it = iter(dl_val)
            xs_val, ys_val = next(dl_val_it)
        xs_val, ys_val = xs_val.to(device), ys_val.to(device)

        val_loss = F.cross_entropy(model(xs_val), ys_val)
        accuracy=test(model)
        model.train()
        #### Log ####
        data={
            "epoch": epoch,
            "train_loss": loss.item(),
            "val_loss": val_loss.item(),
            "acc": accuracy,
            "time": tf - t0,

            "param": None,
        }
        log.append(data)

    return log

def train_classic_tests(model, epochs=1):
    #opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

    countcopy=0
    model.train()
    dl_val_it = iter(dl_val)
    log = []

    fmodel = higher.patch.monkeypatch(model, device=None, copy_initial_weights=True)
    doptim = higher.optim.get_diff_optim(optim, model.parameters(), fmodel=fmodel, track_higher_grads=False)
    for epoch in range(epochs):
        print_torch_mem("Start epoch")
        print(len(fmodel._fast_params))
        t0 = time.process_time()
        #with higher.innerloop_ctx(model, optim, copy_initial_weights=True, track_higher_grads=True) as (fmodel, doptim):

        #fmodel = higher.patch.monkeypatch(model, device=None, copy_initial_weights=True)
        #doptim = higher.optim.get_diff_optim(optim, model.parameters(), track_higher_grads=True)

        for i, (features, labels) in enumerate(dl_train):
            features,labels = features.to(device), labels.to(device)

            #with higher.innerloop_ctx(model, optim, copy_initial_weights=True, track_higher_grads=False) as (fmodel, doptim):

            
            #optim.zero_grad()
            pred = fmodel.forward(features)
            loss = F.cross_entropy(pred,labels)
            doptim.step(loss) #(opt.zero_grad, loss.backward, opt.step)
            #loss.backward()
            #new_params = doptim.step(loss, params=fmodel.parameters())
            #fmodel.update_params(new_params)

            
            #print('Fast param',len(fmodel._fast_params))
            #print('opt state', type(doptim.state[0][0]['momentum_buffer']), doptim.state[0][2]['momentum_buffer'].shape)
            
            if False or (len(fmodel._fast_params)>1):
                print("fmodel fast param",len(fmodel._fast_params))
                '''
                #val_loss = F.cross_entropy(fmodel(features), labels)

                #print_graph(val_loss)

                #val_loss.backward()
                #print('bip')

                tmp = fmodel.parameters()
                
                #print(list(tmp)[1])
                tmp = [higher.utils._copy_tensor(t,safe_copy=True) if isinstance(t, torch.Tensor) else t for t in tmp]
                #print(len(tmp))

                #fmodel._fast_params.clear()
                del fmodel._fast_params
                fmodel._fast_params=None
                
                fmodel.fast_params=tmp # Surcharge la memoire          
                #fmodel.update_params(tmp) #Meilleur perf / Surcharge la memoire avec trach higher grad

                #optim._fmodel=fmodel
                '''
            

                countcopy+=1
                model_copy(src=fmodel, dst=model, patch_copy=False)
                fmodel = higher.patch.monkeypatch(model, device=None, copy_initial_weights=True)
                #doptim.detach_dyn()
                #tmp = doptim.state
                #tmp = doptim.state_dict()
                #for k, v in tmp['state'].items():
                #    print('dict',k, type(v))

                a = optim.param_groups[0]['params'][0]
                state = optim.state[a]
                #state['momentum_buffer'] = None
                #print('opt state', type(optim.state[a]), len(optim.state[a]))
                #optim.load_state_dict(tmp)


                for group_idx, group in enumerate(optim.param_groups):
                   # print('gp idx',group_idx)
                    for p_idx, p in enumerate(group['params']):
                        optim.state[p]=doptim.state[group_idx][p_idx]

                #print('opt state', type(optim.state[a]['momentum_buffer']), optim.state[a]['momentum_buffer'][0:10])
                #print('dopt state', type(doptim.state[0][0]['momentum_buffer']), doptim.state[0][0]['momentum_buffer'][0:10])
                '''
                for a in tmp:
                    #print(type(a), len(a))
                    for nb, b in a.items():
                        #print(nb, type(b), len(b))
                        for n, state in b.items():
                            #print(n, type(states))
                            #print(state.grad_fn)
                            state = torch.tensor(state.data).requires_grad_()
                            #print(state.grad_fn)
                '''


                doptim = higher.optim.get_diff_optim(optim, model.parameters(), track_higher_grads=True)
                #doptim.state = tmp


        countcopy+=1
        model_copy(src=fmodel, dst=model)
        optim_copy(dopt=diffopt, opt=inner_opt) 

        #### Tests ####
        tf = time.process_time()
        try:
            xs_val, ys_val = next(dl_val_it)
        except StopIteration: #Fin epoch val
            dl_val_it = iter(dl_val)
            xs_val, ys_val = next(dl_val_it)
        xs_val, ys_val = xs_val.to(device), ys_val.to(device)

        val_loss = F.cross_entropy(model(xs_val), ys_val)
        accuracy=test(model)
        model.train()
        #### Log ####
        data={
            "epoch": epoch,
            "train_loss": loss.item(),
            "val_loss": val_loss.item(),
            "acc": accuracy,
            "time": tf - t0,

            "param": None,
        }
        log.append(data)

    #countcopy+=1
    #model_copy(src=fmodel, dst=model, patch_copy=False)
    #optim.load_state_dict(doptim.state_dict()) #Besoin sauver etat otpim ?

    print("Copy ", countcopy)
    return log

def run_simple_dataug(inner_it, epochs=1):

    dl_train_it = iter(dl_train)
    dl_val_it = iter(dl_val)

    #aug_model = nn.Sequential(
    #    Data_aug(),
    #    LeNet(1,10),
    #    )
    aug_model = Augmented_model(Data_aug(), LeNet(1,10)).to(device)
    print(str(aug_model))
    meta_opt = torch.optim.Adam(aug_model['data_aug'].parameters(), lr=1e-2)
    inner_opt = torch.optim.SGD(aug_model['model'].parameters(), lr=1e-2, momentum=0.9)

    log = []
    t0 = time.process_time()

    epoch = 0
    while epoch < epochs:
        meta_opt.zero_grad()
        aug_model.train()
        with higher.innerloop_ctx(aug_model, inner_opt, copy_initial_weights=True, track_higher_grads=True) as (fmodel, diffopt): #effet copy_initial_weight pas clair...

            for i in range(n_inner_iter):
                try:
                    xs, ys = next(dl_train_it)
                except StopIteration: #Fin epoch train
                    tf = time.process_time()
                    epoch +=1
                    dl_train_it = iter(dl_train)
                    xs, ys = next(dl_train_it)

                    accuracy=test(aug_model)
                    aug_model.train()

                    #### Print ####
                    print('-'*9)
                    print('Epoch %d/%d'%(epoch,epochs))
                    print('train loss',loss.item(), '/ val loss', val_loss.item())
                    print('acc', accuracy)
                    print('mag', aug_model['data_aug']['mag'].item())

                    #### Log ####
                    data={
                        "epoch": epoch,
                        "train_loss": loss.item(),
                        "val_loss": val_loss.item(),
                        "acc": accuracy,
                        "time": tf - t0,

                        "param": aug_model['data_aug']['mag'].item(),
                    }
                    log.append(data)
                    t0 = time.process_time()

                xs, ys = xs.to(device), ys.to(device)

                logits = fmodel(xs)  # modified `params` can also be passed as a kwarg

                loss = F.cross_entropy(logits, ys)  # no need to call loss.backwards()
                #loss.backward(retain_graph=True)
                #print(fmodel['model']._params['b4'].grad)
                #print('mag', fmodel['data_aug']['mag'].grad)

                diffopt.step(loss)  # note that `step` must take `loss` as an argument!
                # The line above gets P[t+1] from P[t] and loss[t]. `step` also returns
                # these new parameters, as an alternative to getting them from
                # `fmodel.fast_params` or `fmodel.parameters()` after calling
                # `diffopt.step`.

                # At this point, or at any point in the iteration, you can take the
                # gradient of `fmodel.parameters()` (or equivalently
                # `fmodel.fast_params`) w.r.t. `fmodel.parameters(time=0)` (equivalently
                # `fmodel.init_fast_params`). i.e. `fast_params` will always have
                # `grad_fn` as an attribute, and be part of the gradient tape.
            
            # At the end of your inner loop you can obtain these e.g. ...
            #grad_of_grads = torch.autograd.grad(
            #    meta_loss_fn(fmodel.parameters()), fmodel.parameters(time=0))
            try:
                xs_val, ys_val = next(dl_val_it)
            except StopIteration: #Fin epoch val
                dl_val_it = iter(dl_val)
                xs_val, ys_val = next(dl_val_it)
            xs_val, ys_val = xs_val.to(device), ys_val.to(device)

            fmodel.augment(mode=False)
            val_logits = fmodel(xs_val) #Validation sans transfornations !
            val_loss = F.cross_entropy(val_logits, ys_val)
            #print('val_loss',val_loss.item())
            val_loss.backward()

            #print('mag', fmodel['data_aug']['mag'], '/', fmodel['data_aug']['mag'].grad)

            #model=copy.deepcopy(fmodel)
            aug_model.load_state_dict(fmodel.state_dict()) #Do not copy gradient ! 
            #Copie des gradients
            for paramName, paramValue, in fmodel.named_parameters():
              for netCopyName, netCopyValue, in aug_model.named_parameters():
                if paramName == netCopyName:
                  netCopyValue.grad = paramValue.grad

            #print('mag', aug_model['data_aug']['mag'], '/', aug_model['data_aug']['mag'].grad)
            meta_opt.step()

    plot_res(log, fig_name="res/{}-{} epochs- {} in_it".format(str(aug_model),epochs,inner_it))
    print('-'*9)
    times = [x["time"] for x in log]
    print(str(aug_model),": acc", max([x["acc"] for x in log]), "in (ms):", np.mean(times), "+/-", np.std(times))

def run_dist_dataug(model, epochs=1, inner_it=1, dataug_epoch_start=0):

    dl_train_it = iter(dl_train)
    dl_val_it = iter(dl_val)
    
    meta_opt = torch.optim.Adam(model['data_aug'].parameters(), lr=1e-3)
    inner_opt = torch.optim.SGD(model['model'].parameters(), lr=1e-2, momentum=0.9)

    high_grad_track = True
    if dataug_epoch_start>0:
        model.augment(mode=False)
        high_grad_track = False

    model.train()

    log = []
    t0 = time.process_time()

    countcopy=0
    val_loss=torch.tensor(0)
    opt_param=None

    epoch = 0
    while epoch < epochs:
        meta_opt.zero_grad()
        with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=True, override=opt_param, track_higher_grads=high_grad_track) as (fmodel, diffopt): #effet copy_initial_weight pas clair...

            for i in range(n_inner_iter):
                try:
                    xs, ys = next(dl_train_it)
                except StopIteration: #Fin epoch train
                    tf = time.process_time()
                    epoch +=1
                    dl_train_it = iter(dl_train)
                    xs, ys = next(dl_train_it)

                    #viz_sample_data(imgs=xs, labels=ys, fig_name='samples/data_sample_epoch{}_noTF'.format(epoch))
                    #viz_sample_data(imgs=aug_model['data_aug'](xs), labels=ys, fig_name='samples/data_sample_epoch{}'.format(epoch))

                    accuracy=test(model)
                    model.train()

                    #### Print ####
                    print('-'*9)
                    print('Epoch : %d/%d'%(epoch,epochs))
                    print('Train loss :',loss.item(), '/ val loss', val_loss.item())
                    print('Accuracy :', accuracy)
                    print('Data Augmention : {} (Epoch {})'.format(model._data_augmentation, dataug_epoch_start))
                    print('TF Proba :', model['data_aug']['prob'].data)
                    #print('proba grad',aug_model['data_aug']['prob'].grad)
                    #############
                    #### Log ####
                    data={
                        "epoch": epoch,
                        "train_loss": loss.item(),
                        "val_loss": val_loss.item(),
                        "acc": accuracy,
                        "time": tf - t0,

                        "param": [p for p in model['data_aug']['prob']],
                    }
                    log.append(data)
                    #############

                    if epoch == dataug_epoch_start:
                        print('Starting Data Augmention...')
                        model.augment(mode=True)
                        high_grad_track = True

                    t0 = time.process_time()

                xs, ys = xs.to(device), ys.to(device)

                '''
                #Methode exacte
                final_loss = 0
                for tf_idx in range(fmodel['data_aug']._nb_tf):
                    fmodel['data_aug'].transf_idx=tf_idx
                    logits = fmodel(xs)
                    loss = F.cross_entropy(logits, ys)
                    #loss.backward(retain_graph=True)
                    #print('idx', tf_idx)
                    #print(fmodel['data_aug']['prob'][tf_idx], fmodel['data_aug']['prob'][tf_idx].grad)
                    final_loss += loss*fmodel['data_aug']['prob'][tf_idx] #Take it in the forward function ?
                
                loss = final_loss 
                '''
                #Methode uniforme
                logits = fmodel(xs)  # modified `params` can also be passed as a kwarg
                loss = F.cross_entropy(logits, ys, reduction='none')  # no need to call loss.backwards()
                if fmodel._data_augmentation: #Weight loss
                    w_loss = fmodel['data_aug'].loss_weight().to(device)
                    loss = loss * w_loss
                loss = loss.mean()
                #'''

                #to visualize computational graph
                #print_graph(loss)

                #loss.backward(retain_graph=True)
                #print(fmodel['model']._params['b4'].grad)
                #print('prob grad', fmodel['data_aug']['prob'].grad)

                diffopt.step(loss) #(opt.zero_grad, loss.backward, opt.step)

            try:
                xs_val, ys_val = next(dl_val_it)
            except StopIteration: #Fin epoch val
                dl_val_it = iter(dl_val)
                xs_val, ys_val = next(dl_val_it)
            xs_val, ys_val = xs_val.to(device), ys_val.to(device)

            fmodel.augment(mode=False) #Validation sans transfornations !
            val_loss = F.cross_entropy(fmodel(xs_val), ys_val)

            #print_graph(val_loss)

            val_loss.backward()
            
            countcopy+=1
            model_copy(src=fmodel, dst=model)
            optim_copy(dopt=diffopt, opt=inner_opt)
            
            meta_opt.step()
            model['data_aug'].adjust_prob() #Contrainte sum(proba)=1

    print("Copy ", countcopy)
    return log

def run_dist_dataugV2(model, epochs=1, inner_it=0, dataug_epoch_start=0, print_freq=1, loss_patience=None):

    log = []
    countcopy=0
    val_loss=torch.tensor(0) #Necessaire si pas de metastep sur une epoch
    dl_val_it = iter(dl_val)

    meta_opt = torch.optim.Adam(model['data_aug'].parameters(), lr=1e-2)
    inner_opt = torch.optim.SGD(model['model'].parameters(), lr=1e-2, momentum=0.9)

    high_grad_track = True
    if inner_it == 0:
        high_grad_track=False
    if dataug_epoch_start!=0:
        model.augment(mode=False)
        high_grad_track = False

    val_loss_monitor= None
    if loss_patience != None :
        if dataug_epoch_start==-1: val_loss_monitor = loss_monitor(patience=loss_patience, end_train=2) #1st limit = dataug start
        else: val_loss_monitor = loss_monitor(patience=loss_patience) #Val loss monitor

    model.train()
    
    fmodel = higher.patch.monkeypatch(model, device=None, copy_initial_weights=True)
    diffopt = higher.optim.get_diff_optim(inner_opt, model.parameters(),fmodel=fmodel,track_higher_grads=high_grad_track)

    for epoch in range(1, epochs+1):
        #print_torch_mem("Start epoch "+str(epoch))
        #print(high_grad_track, fmodel._data_augmentation, len(fmodel._fast_params))
        t0 = time.process_time()
        #with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=True, override=opt_param, track_higher_grads=high_grad_track) as (fmodel, diffopt):

        for i, (xs, ys) in enumerate(dl_train):
            xs, ys = xs.to(device), ys.to(device)
            '''
            #Methode exacte
            final_loss = 0
            for tf_idx in range(fmodel['data_aug']._nb_tf):
                fmodel['data_aug'].transf_idx=tf_idx
                logits = fmodel(xs)
                loss = F.cross_entropy(logits, ys)
                #loss.backward(retain_graph=True)
                #print('idx', tf_idx)
                #print(fmodel['data_aug']['prob'][tf_idx], fmodel['data_aug']['prob'][tf_idx].grad)
                final_loss += loss*fmodel['data_aug']['prob'][tf_idx] #Take it in the forward function ?
            
            loss = final_loss 
            '''
            #Methode uniforme
            
            logits = fmodel(xs)  # modified `params` can also be passed as a kwarg
            loss = F.cross_entropy(logits, ys, reduction='none')  # no need to call loss.backwards()
            #PAS PONDERE LOSS POUR DIST MIX
            if fmodel._data_augmentation: # and not fmodel['data_aug']._mix_dist: #Weight loss
                w_loss = fmodel['data_aug'].loss_weight().to(device)
                loss = loss * w_loss
            loss = loss.mean()
            #'''

            #to visualize computational graph
            #print_graph(loss)

            #loss.backward(retain_graph=True)
            #print(fmodel['model']._params['b4'].grad)
            #print('prob grad', fmodel['data_aug']['prob'].grad)

            diffopt.step(loss) #(opt.zero_grad, loss.backward, opt.step)

            if(high_grad_track and i%inner_it==0): #Perform Meta step
                #print("meta")
                #Peu utile si high_grad_track = False
                val_loss = compute_vaLoss(model=fmodel, dl_val_it=dl_val_it)

                #print_graph(val_loss)

                val_loss.backward()
            
                countcopy+=1
                model_copy(src=fmodel, dst=model)
                optim_copy(dopt=diffopt, opt=inner_opt)

                meta_opt.step()
                model['data_aug'].adjust_prob(soft=False) #Contrainte sum(proba)=1

                fmodel = higher.patch.monkeypatch(model, device=None, copy_initial_weights=True)
                diffopt = higher.optim.get_diff_optim(inner_opt, model.parameters(),fmodel=fmodel, track_higher_grads=high_grad_track)

        tf = time.process_time()

        #viz_sample_data(imgs=xs, labels=ys, fig_name='samples/data_sample_epoch{}_noTF'.format(epoch))
        #viz_sample_data(imgs=aug_model['data_aug'](xs), labels=ys, fig_name='samples/data_sample_epoch{}'.format(epoch))
        
        if(not high_grad_track): 
            countcopy+=1
            model_copy(src=fmodel, dst=model)
            optim_copy(dopt=diffopt, opt=inner_opt)
            val_loss = compute_vaLoss(model=fmodel, dl_val_it=dl_val_it)

            #Necessaire pour reset higher (Accumule les fast_param meme avec track_higher_grads = False)
            fmodel = higher.patch.monkeypatch(model, device=None, copy_initial_weights=True)
            diffopt = higher.optim.get_diff_optim(inner_opt, model.parameters(),fmodel=fmodel, track_higher_grads=high_grad_track)

        accuracy=test(model)
        model.train()

        #### Print ####
        if(print_freq and epoch%print_freq==0):
            print('-'*9)
            print('Epoch : %d/%d'%(epoch,epochs))
            print('Time : %.00f ms'%(tf - t0))
            print('Train loss :',loss.item(), '/ val loss', val_loss.item())
            print('Accuracy :', accuracy)
            print('Data Augmention : {} (Epoch {})'.format(model._data_augmentation, dataug_epoch_start))
            print('TF Proba :', model['data_aug']['prob'].data)
            #print('proba grad',aug_model['data_aug']['prob'].grad)
        #############
        #### Log ####
        data={
            "epoch": epoch,
            "train_loss": loss.item(),
            "val_loss": val_loss.item(),
            "acc": accuracy,
            "time": tf - t0,

            "param": [p.item() for p in model['data_aug']['prob']],
        }
        log.append(data)
        #############
        if val_loss_monitor : 
            val_loss_monitor.register(val_loss.item())
            if val_loss_monitor.end_training(): break #Stop training


        if not model.is_augmenting() and (epoch == dataug_epoch_start or (val_loss_monitor and val_loss_monitor.limit_reached()==1)):
            print('Starting Data Augmention...')
            dataug_epoch_start = epoch
            model.augment(mode=True)
            if inner_it != 0: high_grad_track = True

    #print("Copy ", countcopy)
    return log

##########################################
if __name__ == "__main__":

    n_inner_iter = 0
    epochs = 100
    dataug_epoch_start=0

    #### Classic ####
    '''
    model = LeNet(3,10).to(device)
    #model = torchvision.models.resnet18()
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
    '''
    tf_dict = {k: TF.TF_dict[k] for k in tf_names}
    #tf_dict = TF.TF_dict
    aug_model = Augmented_model(Data_augV4(TF_dict=tf_dict, N_TF=2, mix_dist=0.0), LeNet(3,10)).to(device)
    print(str(aug_model), 'on', device_name)
    #run_simple_dataug(inner_it=n_inner_iter, epochs=epochs)
    log= run_dist_dataugV2(model=aug_model, epochs=epochs, inner_it=n_inner_iter, dataug_epoch_start=dataug_epoch_start, print_freq=10, loss_patience=10)

    ####
    plot_res(log, fig_name="res/{}-{} epochs (dataug:{})- {} in_it".format(str(aug_model),epochs,dataug_epoch_start,n_inner_iter))
    print('-'*9)
    times = [x["time"] for x in log]
    out = {"Accuracy": max([x["acc"] for x in log]), "Time": (np.mean(times),np.std(times)), "Device": device_name, "Param_names": aug_model.TF_names(), "Log": log}
    print(str(aug_model),": acc", out["Accuracy"], "in (ms):", out["Time"][0], "+/-", out["Time"][1])
    with open("res/log/%s.json" % "{}-{} epochs (dataug:{})- {} in_it".format(str(aug_model),epochs,dataug_epoch_start,n_inner_iter), "w+") as f:
        json.dump(out, f, indent=True)
        print('Log :\"',f.name, '\" saved !')
    print('-'*9)
    '''
    #### TF number tests ####
    #'''
    res_folder="res/TF_nb_tests/"
    epochs= 100
    inner_its = [10]
    dataug_epoch_starts= [0]
    TF_nb = [len(TF.TF_dict)] #range(1,len(TF.TF_dict)+1)
    N_seq_TF= [1, 2, 3, 4]
    
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
                print("---Starting N_TF", n_tf,"---")
                for i in TF_nb:
                    keys = list(TF.TF_dict.keys())[0:i]
                    ntf_dict = {k: TF.TF_dict[k] for k in keys}

                    aug_model = Augmented_model(Data_augV4(TF_dict=ntf_dict, N_TF=n_tf, mix_dist=0.0), LeNet(3,10)).to(device)
                    print(str(aug_model), 'on', device_name)
                    #run_simple_dataug(inner_it=n_inner_iter, epochs=epochs)
                    log= run_dist_dataugV2(model=aug_model, epochs=epochs, inner_it=n_inner_iter, dataug_epoch_start=dataug_epoch_start, print_freq=10, loss_patience=None)

                    ####
                    plot_res(log, fig_name=res_folder+"{}-{} epochs (dataug:{})- {} in_it".format(str(aug_model),epochs,dataug_epoch_start,n_inner_iter))
                    print('-'*9)
                    times = [x["time"] for x in log]
                    out = {"Accuracy": max([x["acc"] for x in log]), "Time": (np.mean(times),np.std(times)), "Device": device_name, "Param_names": aug_model.TF_names(), "Log": log}
                    print(str(aug_model),": acc", out["Accuracy"], "in (ms):", out["Time"][0], "+/-", out["Time"][1])
                    with open(res_folder+"log/%s.json" % "{}-{} epochs (dataug:{})- {} in_it".format(str(aug_model),epochs,dataug_epoch_start,n_inner_iter), "w+") as f:
                        json.dump(out, f, indent=True)
                        print('Log :\"',f.name, '\" saved !')
                    print('-'*9)

    #'''

    