import torch
#import torch.optim
import torchvision
import higher

from datasets import *
from utils import *

def test(model):
    device = next(model.parameters()).device
    model.eval()

    #for i, (features, labels) in enumerate(dl_test):
    #    features,labels = features.to(device), labels.to(device)

    #    pred = model.forward(features)
    #    return pred.argmax(dim=1).eq(labels).sum().item() / dl_test.batch_size * 100

    correct = 0
    total = 0
    loss = []
    with torch.no_grad():
        for features, labels in dl_test:
            features,labels = features.to(device), labels.to(device)

            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss.append(F.cross_entropy(outputs, labels).item())

    accuracy = 100 * correct / total

    return accuracy, np.mean(loss)

def compute_vaLoss(model, dl_it, dl):
    device = next(model.parameters()).device
    try:
        xs, ys = next(dl_it)
    except StopIteration: #Fin epoch val
        dl_it = iter(dl)
        xs, ys = next(dl_it)
    xs, ys = xs.to(device), ys.to(device)

    model.eval() #Validation sans transfornations !
    return F.cross_entropy(F.log_softmax(model(xs), dim=1), ys)

def train_classic(model, opt_param, epochs=1, print_freq=1):
    device = next(model.parameters()).device
    #opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    optim = torch.optim.SGD(model.parameters(), lr=opt_param['Inner']['lr'], momentum=opt_param['Inner']['momentum']) #lr=1e-2 / momentum=0.9

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
            logits = model.forward(features)
            pred = F.log_softmax(logits, dim=1)
            loss = F.cross_entropy(pred,labels)
            loss.backward()
            optim.step()

        #### Tests ####
        tf = time.process_time()

        val_loss = compute_vaLoss(model=model, dl_it=dl_val_it, dl=dl_val)
        accuracy, _ =test(model)
        model.train()

        #### Print ####
        if(print_freq and epoch%print_freq==0):
            print('-'*9)
            print('Epoch : %d/%d'%(epoch,epochs))
            print('Time : %.00f'%(tf - t0))
            print('Train loss :',loss.item(), '/ val loss', val_loss.item())
            print('Accuracy :', accuracy)

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
    device = next(model.parameters()).device
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
            logits = model.forward(features)
            pred = F.log_softmax(logits, dim=1)
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

    return log

def train_UDA(model, dl_unsup, opt_param, epochs=1, print_freq=1):

    device = next(model.parameters()).device
    #opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt = torch.optim.SGD(model.parameters(), lr=opt_param['Inner']['lr'], momentum=opt_param['Inner']['momentum']) #lr=1e-2 / momentum=0.9


    model.train()
    dl_val_it = iter(dl_val)
    dl_unsup_it =iter(dl_unsup)
    log = []
    for epoch in range(epochs):
        #print_torch_mem("Start epoch")
        t0 = time.process_time()
        for i, (features, labels) in enumerate(dl_train):
            #print_torch_mem("Start iter")
            features,labels = features.to(device), labels.to(device)

            optim.zero_grad()
            #Supervised
            logits = model.forward(features)
            pred = F.log_softmax(logits, dim=1)
            sup_loss = F.cross_entropy(pred,labels)

            #Unsupervised
            try:
                aug_xs, origin_xs, ys = next(dl_unsup_it)
            except StopIteration: #Fin epoch val
                dl_unsup_it =iter(dl_unsup)
                aug_xs, origin_xs, ys = next(dl_unsup_it)
            aug_xs, origin_xs, ys = aug_xs.to(device), origin_xs.to(device), ys.to(device)

            #print(aug_xs.shape, origin_xs.shape, ys.shape)
            sup_logits = model.forward(origin_xs)
            unsup_logits = model.forward(aug_xs)

            log_sup=F.log_softmax(sup_logits, dim=1)
            log_unsup=F.log_softmax(unsup_logits, dim=1)
            #KL div w/ logits
            unsup_loss = F.softmax(sup_logits, dim=1)*(log_sup-log_unsup)
            unsup_loss=unsup_loss.sum(dim=-1).mean()

            #print(unsup_loss)
            unsupp_coeff = 1
            loss = sup_loss + unsup_loss * unsupp_coeff

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
        accuracy, _ =test(model)
        model.train()

        #### Print ####
        if(print_freq and epoch%print_freq==0):
            print('-'*9)
            print('Epoch : %d/%d'%(epoch,epochs))
            print('Time : %.00f'%(tf - t0))
            print('Train loss :',loss.item(), '/ val loss', val_loss.item())
            print('Sup Loss :', sup_loss.item(), '/ unsup_loss :', unsup_loss.item())
            print('Accuracy :', accuracy)

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

def run_dist_dataugV3(model, opt_param, epochs=1, inner_it=0, dataug_epoch_start=0, print_freq=1, KLdiv=False, hp_opt=False, save_sample=False):
    device = next(model.parameters()).device
    log = []
    dl_val_it = iter(dl_val)
    val_loss=None

    high_grad_track = True
    if inner_it == 0: #No HP optimization
        high_grad_track=False
    if dataug_epoch_start!=0: #Augmentation de donnee differee
        model.augment(mode=False)
        high_grad_track = False

    ## Optimizers ##
    #Inner Opt
    inner_opt = torch.optim.SGD(model['model']['original'].parameters(), lr=opt_param['Inner']['lr'], momentum=opt_param['Inner']['momentum']) #lr=1e-2 / momentum=0.9

    diffopt = model['model'].get_diffopt(
        inner_opt, 
        grad_callback=(lambda grads: clip_norm(grads, max_norm=10)),
        track_higher_grads=high_grad_track)

    #Meta Opt
    hyper_param = list(model['data_aug'].parameters())
    if hp_opt : 
        for param_group in diffopt.param_groups: 
            for param in list(opt_param['Inner'].keys())[1:]:
                param_group[param]=torch.tensor(param_group[param]).to(device).requires_grad_()
                hyper_param += [param_group[param]]
    meta_opt = torch.optim.Adam(hyper_param, lr=opt_param['Meta']['lr']) #lr=1e-2

    model.train()
    meta_opt.zero_grad()

    for epoch in range(1, epochs+1):
        t0 = time.process_time()
       
        for i, (xs, ys) in enumerate(dl_train):
            xs, ys = xs.to(device), ys.to(device)
            
            if(not KLdiv):
            #Methode uniforme
                logits = model(xs)  # modified `params` can also be passed as a kwarg
                loss = F.cross_entropy(F.log_softmax(logits, dim=1), ys, reduction='none')  # no need to call loss.backwards()

                if model._data_augmentation: #Weight loss
                    w_loss = model['data_aug'].loss_weight()#.to(device)
                    loss = loss * w_loss
                loss = loss.mean()
            
            else:
            #Methode KL div
                # Supervised loss (classic)
                if model.is_augmenting()  :
                    model.augment(mode=False)
                    sup_logits = model(xs)
                    model.augment(mode=True)
                else:
                    sup_logits = model(xs)
                log_sup=F.log_softmax(sup_logits, dim=1)
                loss = F.cross_entropy(log_sup, ys)

                # Unsupervised loss (KLdiv)
                if model.is_augmenting() :
                    aug_logits = model(xs)
                    log_aug=F.log_softmax(aug_logits, dim=1)
                    aug_loss=0
                    w_loss = model['data_aug'].loss_weight() #Weight loss

                    #KL div w/ logits - Similarite predictions (distributions)
                    aug_loss = F.softmax(sup_logits, dim=1)*(log_sup-log_aug)
                    aug_loss = aug_loss.sum(dim=-1)
                    aug_loss = (w_loss * aug_loss).mean()
                    aug_loss += (F.cross_entropy(log_aug, ys , reduction='none') * w_loss).mean()

                    unsupp_coeff = 1
                    loss += aug_loss * unsupp_coeff
            
            #print_graph(loss) #to visualize computational graph

            #t = time.process_time()
            diffopt.step(loss) #(opt.zero_grad, loss.backward, opt.step)
            #print(len(model['model']['functional']._fast_params),"step", time.process_time()-t)


            if(high_grad_track and i>0 and i%inner_it==0): #Perform Meta step
                #print("meta")
                val_loss = compute_vaLoss(model=model, dl_it=dl_val_it, dl=dl_val) + model['data_aug'].reg_loss()
                #print_graph(val_loss) #to visualize computational graph
                val_loss.backward()

                torch.nn.utils.clip_grad_norm_(model['data_aug'].parameters(), max_norm=10, norm_type=2) #Prevent exploding grad with RNN

                meta_opt.step()

                #Adjust Hyper-parameters
                model['data_aug'].adjust_param(soft=False) #Contrainte sum(proba)=1
                if hp_opt: 
                    for param_group in diffopt.param_groups: 
                        for param in list(opt_param['Inner'].keys())[1:]:
                            param_group[param].data = param_group[param].data.clamp(min=1e-4)

                #Reset gradients
                diffopt.detach_()
                model['model'].detach_()
                meta_opt.zero_grad()
                
        tf = time.process_time()

        if save_sample: #Data sample saving
                try:
                    viz_sample_data(imgs=xs, labels=ys, fig_name='samples/data_sample_epoch{}_noTF'.format(epoch))
                    viz_sample_data(imgs=model['data_aug'](xs), labels=ys, fig_name='samples/data_sample_epoch{}'.format(epoch))
                except:
                    print("Couldn't save samples epoch"+epoch)
                    pass


        if(not val_loss): #Compute val loss for logs
            val_loss = compute_vaLoss(model=model, dl_it=dl_val_it, dl=dl_val)

        # Test model
        accuracy, test_loss =test(model)
        model.train()

        #### Log ####
        param = [{'p': p.item(), 'm':model['data_aug']['mag'].item()} for p in model['data_aug']['prob']] if model['data_aug']._shared_mag else [{'p': p.item(), 'm': m.item()} for p, m in zip(model['data_aug']['prob'], model['data_aug']['mag'])]
        data={
            "epoch": epoch,
            "train_loss": loss.item(),
            "val_loss": val_loss.item(),
            "acc": accuracy,
            "time": tf - t0,

            "mix_dist": model['data_aug']['mix_dist'].item(),
            "param": param,
        }
        if hp_opt : data["opt_param"]=[{'lr': p_grp['lr'].item(), 'momentum': p_grp['momentum'].item()} for p_grp in diffopt.param_groups]
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
            if not model['data_aug']._fixed_prob: print('TF Proba :', model['data_aug']['prob'].data)
            #print('proba grad',model['data_aug']['prob'].grad)
            if not model['data_aug']._fixed_mag: print('TF Mag :', model['data_aug']['mag'].data)
            #print('Mag grad',model['data_aug']['mag'].grad)
            if not model['data_aug']._fixed_mix: print('Mix:', model['data_aug']['mix_dist'].item())
            #print('Reg loss:', model['data_aug'].reg_loss().item())

            if hp_opt : 
                for param_group in diffopt.param_groups:
                    print('Opt param - lr:', param_group['lr'].item(),'- momentum:', param_group['momentum'].item())
        #############

        #Augmentation de donnee differee
        if not model.is_augmenting() and (epoch == dataug_epoch_start):
            print('Starting Data Augmention...')
            dataug_epoch_start = epoch
            model.augment(mode=True)
            if inner_it != 0: #Rebuild diffopt if needed
                high_grad_track = True
                diffopt = model['model'].get_diffopt(
                    inner_opt, 
                    grad_callback=(lambda grads: clip_norm(grads, max_norm=10)),
                    track_higher_grads=high_grad_track)


    #Data sample saving
    try:
        viz_sample_data(imgs=xs, labels=ys, fig_name='samples/data_sample_epoch{}_noTF'.format(epoch))
        viz_sample_data(imgs=model['data_aug'](xs), labels=ys, fig_name='samples/data_sample_epoch{}'.format(epoch))
    except:
        print("Couldn't save finals samples")
        pass

    return log