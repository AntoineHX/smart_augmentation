import torch
#import torch.optim
import torchvision
import higher

from datasets import *
from utils import *

def train_classic_tests(model, epochs=1):
    device = next(model.parameters()).device
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
        accuracy, _ =test(model)
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
    device = next(model.parameters()).device
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

                    accuracy, _ =test(model)
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
    device = next(model.parameters()).device
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

                    accuracy, _ =test(model)
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
            model['data_aug'].adjust_param() #Contrainte sum(proba)=1

    print("Copy ", countcopy)
    return log

def run_dist_dataugV2(model, opt_param, epochs=1, inner_it=0, dataug_epoch_start=0, print_freq=1, KLdiv=False, loss_patience=None, save_sample=False):
    device = next(model.parameters()).device
    log = []
    countcopy=0
    val_loss=torch.tensor(0) #Necessaire si pas de metastep sur une epoch
    dl_val_it = iter(dl_val)

    #if inner_it!=0: 
    meta_opt = torch.optim.Adam(model['data_aug'].parameters(), lr=opt_param['Meta']['lr']) #lr=1e-2
    inner_opt = torch.optim.SGD(model['model'].parameters(), lr=opt_param['Inner']['lr'], momentum=opt_param['Inner']['momentum']) #lr=1e-2 / momentum=0.9

    high_grad_track = True
    if inner_it == 0:
        high_grad_track=False
    if dataug_epoch_start!=0:
        model.augment(mode=False)
        high_grad_track = False

    val_loss_monitor= None
    if loss_patience != None :
        if dataug_epoch_start==-1: val_loss_monitor = loss_monitor(patience=loss_patience, end_train=2) #1st limit = dataug start
        else: val_loss_monitor = loss_monitor(patience=loss_patience) #Val loss monitor (Not on val data : used by Dataug... => Test data)

    model.train()
    
    fmodel = higher.patch.monkeypatch(model, device=None, copy_initial_weights=True)
    diffopt = higher.optim.get_diff_optim(inner_opt, model.parameters(),fmodel=fmodel, track_higher_grads=high_grad_track)

    meta_opt.zero_grad()

    for epoch in range(1, epochs+1):
        #print_torch_mem("Start epoch "+str(epoch))
        #print(high_grad_track, fmodel._data_augmentation, len(fmodel._fast_params))
        t0 = time.process_time()
        #with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=True, override=opt_param, track_higher_grads=high_grad_track) as (fmodel, diffopt):

        for i, (xs, ys) in enumerate(dl_train):
            xs, ys = xs.to(device), ys.to(device)
            
            #Methode exacte
            #final_loss = 0
            #for tf_idx in range(fmodel['data_aug']._nb_tf):
            #    fmodel['data_aug'].transf_idx=tf_idx
            #    logits = fmodel(xs)
            #    loss = F.cross_entropy(logits, ys)
            #    #loss.backward(retain_graph=True)
            #    final_loss += loss*fmodel['data_aug']['prob'][tf_idx] #Take it in the forward function ? 
            #loss = final_loss 
            
            if(not KLdiv):
            #Methode uniforme
                logits = fmodel(xs)  # modified `params` can also be passed as a kwarg
                loss = F.cross_entropy(F.log_softmax(logits, dim=1), ys, reduction='none')  # no need to call loss.backwards()

                if fmodel._data_augmentation: #Weight loss
                    w_loss = fmodel['data_aug'].loss_weight()#.to(device)
                    loss = loss * w_loss
                loss = loss.mean()
            
            else:
            #Methode KL div
                if fmodel._data_augmentation :
                    fmodel.augment(mode=False)
                    sup_logits = fmodel(xs)
                    fmodel.augment(mode=True)
                else:
                    sup_logits = fmodel(xs)
                log_sup=F.log_softmax(sup_logits, dim=1)
                loss = F.cross_entropy(log_sup, ys)

                if fmodel._data_augmentation:
                    aug_logits = fmodel(xs)
                    log_aug=F.log_softmax(aug_logits, dim=1)
                    
                    w_loss = fmodel['data_aug'].loss_weight() #Weight loss

                    #if epoch>50: #debut differe ?
                    #KL div w/ logits - Similarite predictions (distributions)
                    aug_loss = F.softmax(sup_logits, dim=1)*(log_sup-log_aug)
                    aug_loss = aug_loss.sum(dim=-1)
                    #aug_loss = F.kl_div(aug_logits, sup_logits, reduction='none')
                    aug_loss = (w_loss * aug_loss).mean()

                    aug_loss += (F.cross_entropy(log_aug, ys , reduction='none') * w_loss).mean()

                    unsupp_coeff = 1
                    loss += aug_loss * unsupp_coeff

            #to visualize computational graph
            #print_graph(loss)

            #loss.backward(retain_graph=True)
            #print(fmodel['model']._params['b4'].grad)
            #print('prob grad', fmodel['data_aug']['prob'].grad)

            #t = time.process_time()
            diffopt.step(loss) #(opt.zero_grad, loss.backward, opt.step)
            #print(len(fmodel._fast_params),"step", time.process_time()-t)

            if(high_grad_track and i>0 and i%inner_it==0): #Perform Meta step
                #print("meta")

                val_loss = compute_vaLoss(model=fmodel, dl_it=dl_val_it, dl=dl_val) #+ fmodel['data_aug'].reg_loss()          
                #print_graph(val_loss)

                #t = time.process_time()
                val_loss.backward()
                #print("meta", time.process_time()-t)
                #print('proba grad',model['data_aug']['prob'].grad)
                if model['data_aug']['prob'].grad is None or model['data_aug']['mag'] is None:
                    print("Warning no grad (iter",i,") :\n Prob-",model['data_aug']['prob'].grad,"\n Mag-", model['data_aug']['mag'].grad)

                countcopy+=1
                model_copy(src=fmodel, dst=model)
                optim_copy(dopt=diffopt, opt=inner_opt)

                torch.nn.utils.clip_grad_norm_(model['data_aug'].parameters(), max_norm=10, norm_type=2) #Prevent exploding grad with RNN
                
                #if epoch>50:
                meta_opt.step()
                model['data_aug'].adjust_param(soft=False) #Contrainte sum(proba)=1
                try: #Dataugv6
                    model['data_aug'].next_TF_set()
                except:
                    pass

                fmodel = higher.patch.monkeypatch(model, device=None, copy_initial_weights=True)
                diffopt = higher.optim.get_diff_optim(inner_opt, model.parameters(),fmodel=fmodel, track_higher_grads=high_grad_track)

                meta_opt.zero_grad()

        tf = time.process_time()

        #viz_sample_data(imgs=xs, labels=ys, fig_name='samples/data_sample_epoch{}_noTF'.format(epoch))
        #viz_sample_data(imgs=model['data_aug'](xs), labels=ys, fig_name='samples/data_sample_epoch{}'.format(epoch), weight_labels=model['data_aug'].loss_weight())
        
        if(not high_grad_track): 
            countcopy+=1
            model_copy(src=fmodel, dst=model)
            optim_copy(dopt=diffopt, opt=inner_opt)
            val_loss = compute_vaLoss(model=fmodel, dl_it=dl_val_it, dl=dl_val)

            #Necessaire pour reset higher (Accumule les fast_param meme avec track_higher_grads = False)
            fmodel = higher.patch.monkeypatch(model, device=None, copy_initial_weights=True)
            diffopt = higher.optim.get_diff_optim(inner_opt, model.parameters(),fmodel=fmodel, track_higher_grads=high_grad_track)

        accuracy, test_loss =test(model)
        model.train()

        #### Log ####
        #print(type(model['data_aug']) is dataug.Data_augV5)
        param = [{'p': p.item(), 'm':model['data_aug']['mag'].item()} for p in model['data_aug']['prob']] if model['data_aug']._shared_mag else [{'p': p.item(), 'm': m.item()} for p, m in zip(model['data_aug']['prob'], model['data_aug']['mag'])]
        data={
            "epoch": epoch,
            "train_loss": loss.item(),
            "val_loss": val_loss.item(),
            "acc": accuracy,
            "time": tf - t0,

            "param": param #if isinstance(model['data_aug'], Data_augV5) 
            #else [p.item() for p in model['data_aug']['prob']],
        }
        log.append(data)
        #############
        #### Print ####
        if(print_freq and epoch%print_freq==0):
            print('-'*9)
            print('Epoch : %d/%d'%(epoch,epochs))
            print('Time : %.00f'%(tf - t0))
            print('Train loss :',loss.item(), '/ val loss', val_loss.item())
            print('Accuracy :', max([x["acc"] for x in log]))
            print('Data Augmention : {} (Epoch {})'.format(model._data_augmentation, dataug_epoch_start))
            print('TF Proba :', model['data_aug']['prob'].data)
            #print('proba grad',model['data_aug']['prob'].grad)
            print('TF Mag :', model['data_aug']['mag'].data)
            #print('Mag grad',model['data_aug']['mag'].grad)
            #print('Reg loss:', model['data_aug'].reg_loss().item())
            #print('Aug loss', aug_loss.item())
        #############
        if val_loss_monitor : 
            model.eval()
            val_loss_monitor.register(test_loss)#val_loss.item())
            if val_loss_monitor.end_training(): break #Stop training
            model.train()

        if not model.is_augmenting() and (epoch == dataug_epoch_start or (val_loss_monitor and val_loss_monitor.limit_reached()==1)):
            print('Starting Data Augmention...')
            dataug_epoch_start = epoch
            model.augment(mode=True)
            if inner_it != 0: high_grad_track = True

    try:
        viz_sample_data(imgs=xs, labels=ys, fig_name='samples/data_sample_epoch{}_noTF'.format(epoch))
        viz_sample_data(imgs=model['data_aug'](xs), labels=ys, fig_name='samples/data_sample_epoch{}'.format(epoch), weight_labels=model['data_aug'].loss_weight())
    except:
        print("Couldn't save finals samples")
        pass

    #print("Copy ", countcopy)
    return log
