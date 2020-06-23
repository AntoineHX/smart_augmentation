""" Utilities function for training.

"""
import sys
import torch
#import torch.optim
#import torchvision
import higher
import higher_patch

from datasets import *
from utils import *

from transformations import Normalizer, translate, zero_stack
norm = Normalizer(MEAN, STD)
confmat = ConfusionMatrix(num_classes=len(dl_test.dataset.classes))

max_grad = 1 #Max gradient value #Limite catastrophic drop

def test(model, augment=0):
    """Evaluate a model on test data.

        Args:
            model (nn.Module): Model to test.
            augment (int): Number of augmented example for each sample. (Default : 0)

        Returns:
            (float, Tensor) Returns the accuracy and F1 score of the model.
    """
    device = next(model.parameters()).device
    model.eval()
    # model['model']['functional'].set_mode('mixed') #ABN

    #for i, (features, labels) in enumerate(dl_test):
    #    features,labels = features.to(device), labels.to(device)

    #    pred = model.forward(features)
    #    return pred.argmax(dim=1).eq(labels).sum().item() / dl_test.batch_size * 100

    correct = 0
    total = 0
    #loss = []
    global confmat
    confmat.reset()
    with torch.no_grad():
        for features, labels in dl_test:
            features,labels = features.to(device), labels.to(device)

            if augment>0: #Test Time Augmentation
                model.augment(True)
                # V2
                features=torch.cat([features for _ in range(augment)], dim=0) # (B,C,H,W)=>(B*augment,C,H,W)
                outputs=model(features)
                outputs=torch.cat([o.unsqueeze(dim=0) for o in outputs.chunk(chunks=augment, dim=0)],dim=0) # (B*augment,nb_class)=>(augment,B,nb_class)

                w_losses=model['data_aug'].loss_weight(batch_norm=False) #(B*augment) if Dataug
                if w_losses.shape[0]==1: #RandAugment
                    outputs=torch.sum(outputs, axis=0)/augment #mean
                else: #Dataug
                    w_losses=torch.cat([w.unsqueeze(dim=0) for w in w_losses.chunk(chunks=augment, dim=0)], dim=0) #(augment, B)
                    w_losses = w_losses / w_losses.sum(axis=0, keepdim=True) #sum(w_losses)=1 pour un même echantillons
                
                    outputs=torch.sum(outputs*w_losses.unsqueeze(dim=2).expand_as(outputs), axis=0)/augment #Weighted mean
            else:
                outputs = model(features)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            #loss.append(F.cross_entropy(outputs, labels).item())
            confmat.update(labels, predicted)

    accuracy = 100 * correct / total

    #print(confmat)
    #from sklearn.metrics import f1_score
    #f1 = f1_score(labels.data.to('cpu'), predicted.data.to('cpu'), average="macro")

    return accuracy, confmat.f1_metric(average=None)

def compute_vaLoss(model, dl_it, dl):
    """Evaluate a model on a batch of data.

        Args: 
            model (nn.Module): Model to evaluate.
            dl_it (Iterator): Data loader iterator.
            dl (DataLoader): Data loader.

        Returns:
            (Tensor) Loss on a single batch of data.
    """
    device = next(model.parameters()).device
    try:
        xs, ys = next(dl_it)
    except StopIteration: #Fin epoch val
        dl_it = iter(dl)
        xs, ys = next(dl_it)
    xs, ys = xs.to(device), ys.to(device)

    model.eval() #Validation sans transformations !
    # model['model']['functional'].set_mode('mixed') #ABN
    # return F.cross_entropy(F.log_softmax(model(xs), dim=1), ys)
    return F.cross_entropy(model(xs), ys)

def mixed_loss(xs, ys, model, unsup_factor=1, augment=1):
    """Evaluate a model on a batch of data.

        Compute a combinaison of losses:
            + Supervised Cross-Entropy loss from original data.
            + Unsupervised Cross-Entropy loss from augmented data.
            + KL divergence loss encouraging similarity between original and augmented prediction.

        If unsup_factor is equal to 0 or if there isn't data augmentation, only the supervised loss is computed.

        Inspired by UDA, see: https://github.com/google-research/uda/blob/master/image/main.py

        Args:
            xs (Tensor): Batch of data.
            ys (Tensor): Batch of labels.
            model (nn.Module): Augmented model (see dataug.py).
            unsup_factor (float): Factor by which unsupervised CE and KL div loss are multiplied.
            augment (int): Number of augmented example for each sample. (Default : 1)

        Returns:
            (Tensor) Mixed loss if there's data augmentation, just supervised CE loss otherwise.
    """

    #TODO: add test to prevent augmented model error and redirect to classic loss
    if unsup_factor!=0 and model.is_augmenting() and augment>0:

        # Supervised loss - Cross-entropy
        model.augment(mode=False)
        sup_logits = model(xs)
        model.augment(mode=True)

        log_sup = F.log_softmax(sup_logits, dim=1)
        sup_loss = F.nll_loss(log_sup, ys)
        # sup_loss = F.cross_entropy(log_sup, ys)

        if augment>1:
            # Unsupervised loss - Cross-Entropy
            xs_a=torch.cat([xs for _ in range(augment)], dim=0) # (B,C,H,W)=>(B*augment,C,H,W)
            ys_a=torch.cat([ys for _ in range(augment)], dim=0)
            aug_logits=model(xs_a) # (B*augment,nb_class)
            
            w_loss=model['data_aug'].loss_weight() #(B*augment) if Dataug
            
            log_aug = F.log_softmax(aug_logits, dim=1)
            aug_loss = F.nll_loss(log_aug, ys_a , reduction='none')
            # aug_loss = F.cross_entropy(log_aug, ys_a , reduction='none')
            aug_loss = (aug_loss * w_loss).mean()

            #KL divergence loss (w/ logits) - Prediction/Distribution similarity
            sup_logits_a=torch.cat([sup_logits for _ in range(augment)], dim=0)
            log_sup_a=torch.cat([log_sup for _ in range(augment)], dim=0)

            kl_loss = (F.softmax(sup_logits_a, dim=1)*(log_sup_a-log_aug)).sum(dim=-1)
            kl_loss = (w_loss * kl_loss).mean()
        else:  
            # Unsupervised loss - Cross-Entropy
            aug_logits = model(xs)
            w_loss = model['data_aug'].loss_weight() #Weight loss
            
            log_aug = F.log_softmax(aug_logits, dim=1)
            aug_loss = F.nll_loss(log_aug, ys , reduction='none')
            # aug_loss = F.cross_entropy(log_aug, ys , reduction='none')
            aug_loss = (aug_loss * w_loss).mean()

            #KL divergence loss (w/ logits) - Prediction/Distribution similarity
            kl_loss = (F.softmax(sup_logits, dim=1)*(log_sup-log_aug)).sum(dim=-1)
            kl_loss = (w_loss * kl_loss).mean()

        loss = sup_loss + unsup_factor * (aug_loss + kl_loss)

    else: #Supervised loss - Cross-Entropy
        sup_logits = model(xs)
        loss = F.cross_entropy(sup_logits, ys)
        # log_sup = F.log_softmax(sup_logits, dim=1)
        # loss = F.cross_entropy(log_sup, ys)

    return loss

def train_classic(model, opt_param, epochs=1, print_freq=1):
    """Classic training of a model.

        Args:
            model (nn.Module): Model to train.
            opt_param (dict): Dictionnary containing optimizers parameters.
            epochs (int): Number of epochs to perform. (default: 1)
            print_freq (int): Number of epoch between display of the state of training. If set to None, no display will be done. (default:1)

        Returns:
            (list) Logs of training. Each items is a dict containing results of an epoch.
    """
    device = next(model.parameters()).device

    #Optimizer
    #opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    optim = torch.optim.SGD(model.parameters(), 
        lr=opt_param['Inner']['lr'], 
        momentum=opt_param['Inner']['momentum'], 
        weight_decay=opt_param['Inner']['weight_decay'], 
        nesterov=opt_param['Inner']['nesterov']) #lr=1e-2 / momentum=0.9

    #Scheduler
    inner_scheduler=None
    if opt_param['Inner']['scheduler']=='cosine':
        inner_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs, eta_min=0.)
    elif opt_param['Inner']['scheduler']=='multiStep':
        #Multistep milestones inspired by AutoAugment
        inner_scheduler=torch.optim.lr_scheduler.MultiStepLR(optim, 
            milestones=[int(epochs/3), int(epochs*2/3), int(epochs*2.7/3)], 
            gamma=0.1)
    elif opt_param['Inner']['scheduler']=='exponential':
        #inner_scheduler=torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.1) #Wrong gamma
        inner_scheduler=torch.optim.lr_scheduler.LambdaLR(optim, lambda epoch: (1 - epoch / epochs) ** 0.9)
    elif opt_param['Inner']['scheduler'] is not None:
        raise ValueError("Lr scheduler unknown : %s"%opt_param['Inner']['scheduler'])

    # from warmup_scheduler import GradualWarmupScheduler
    # inner_scheduler=GradualWarmupScheduler(optim, multiplier=2, total_epoch=5, after_scheduler=inner_scheduler)

    #Training
    model.train()
    dl_val_it = iter(dl_val)
    log = []
    for epoch in range(epochs):
        #print_torch_mem("Start epoch")
        #print(optim.param_groups[0]['lr'])
        t0 = time.perf_counter()
        for i, (features, labels) in enumerate(dl_train):
            #viz_sample_data(imgs=features, labels=labels, fig_name='../samples/data_sample_epoch{}_noTF'.format(epoch))
            #print_torch_mem("Start iter")
            features,labels = features.to(device), labels.to(device)

            optim.zero_grad()
            logits = model.forward(features)
            pred = F.log_softmax(logits, dim=1)
            loss = F.cross_entropy(pred,labels)
            loss.backward()
            optim.step()

            # print_graph(loss, '../samples/torchvision_WRN') #to visualize computational graph
            # sys.exit()

        if inner_scheduler is not None:
            inner_scheduler.step()
            # print(optim.param_groups[0]['lr'])

        #### Tests ####
        tf = time.perf_counter()

        val_loss = compute_vaLoss(model=model, dl_it=dl_val_it, dl=dl_val)
        accuracy, f1 =test(model)
        model.train()

        #### Log ####
        data={
            "epoch": epoch,
            "train_loss": loss.item(),
            "val_loss": val_loss.item(),
            "acc": accuracy,
            "f1": f1.tolist(),
            "time": tf - t0,

            "param": None,
        }
        log.append(data)
        #### Print ####
        if(print_freq and epoch%print_freq==0):
            print('-'*9)
            print('Epoch : %d/%d'%(epoch,epochs))
            print('Time : %.00f'%(tf - t0))
            print('Train loss :',loss.item(), '/ val loss', val_loss.item())
            print('Accuracy max:', max([x["acc"] for x in log]))
            print('F1 :', ["{0:0.4f}".format(i) for i in f1])

    return log

def run_dist_dataugV3(model, opt_param, epochs=1, inner_it=1, dataug_epoch_start=0, unsup_loss=1, augment_loss=1, hp_opt=False, print_freq=1, save_sample_freq=None):
    """Training of an augmented model with higher.

            This function is intended to be used with Augmented_model containing an Higher_model (see dataug.py).
            Ex : Augmented_model(Data_augV5(...), Higher_model(model))

            Training loss can either be computed directly from augmented inputs (unsup_loss=0).
            However, it is recommended to use the mixed loss computation, which combine original and augmented inputs to compute the loss (unsup_loss>0).

        Args:
            model (nn.Module): Augmented model to train.
            opt_param (dict): Dictionnary containing optimizers parameters.
            epochs (int): Number of epochs to perform. (default: 1)
            inner_it (int): Number of inner iteration before a meta-step. 0 inner iteration means there's no meta-step. (default: 1)
            dataug_epoch_start (int): Epoch when to start data augmentation. (default: 0)
            unsup_loss (float): Proportion of the unsup_loss loss added to the supervised loss. If set to 0, the loss is only computed on augmented inputs. (default: 1)
            augment_loss (int): Number of augmented example for each sample in loss computation. (Default : 1)
            hp_opt (bool): Wether to learn inner optimizer parameters. (default: False)
            print_freq (int): Number of epoch between display of the state of training. If set to None, no display will be done. (default:1)
            save_sample_freq (int): Number of epochs between saves of samples of data. If set to None, no sample will be saved. (default: None)

        Returns:
            (list) Logs of training. Each items is a dict containing results of an epoch.
    """
    device = next(model.parameters()).device
    log = []
    # kl_log={"prob":[], "mag":[]}
    dl_val_it = iter(dl_val)

    high_grad_track = True
    if inner_it == 0: #No HP optimization
        high_grad_track=False
    if dataug_epoch_start!=0: #Augmentation de donnee differee
        model.augment(mode=False)
        high_grad_track = False

    ## Optimizers ##
    #Inner Opt
    inner_opt = torch.optim.SGD(model['model']['original'].parameters(), 
        lr=opt_param['Inner']['lr'],   
        momentum=opt_param['Inner']['momentum'], 
        weight_decay=opt_param['Inner']['weight_decay'], 
        nesterov=opt_param['Inner']['nesterov']) #lr=1e-2 / momentum=0.9

    diffopt = model['model'].get_diffopt(
        inner_opt, 
        grad_callback=(lambda grads: clip_norm(grads, max_norm=max_grad)),
        track_higher_grads=high_grad_track)

    #Scheduler
    inner_scheduler=None
    if opt_param['Inner']['scheduler']=='cosine':
        inner_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(inner_opt, T_max=epochs, eta_min=0.)
    elif opt_param['Inner']['scheduler']=='multiStep':
        #Multistep milestones inspired by AutoAugment
        inner_scheduler=torch.optim.lr_scheduler.MultiStepLR(inner_opt, 
            milestones=[int(epochs/3), int(epochs*2/3), int(epochs*2.7/3)], 
            gamma=0.1)
    elif opt_param['Inner']['scheduler']=='exponential':
        #inner_scheduler=torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.1) #Wrong gamma
        inner_scheduler=torch.optim.lr_scheduler.LambdaLR(inner_opt, lambda epoch: (1 - epoch / epochs) ** 0.9)
    elif not(opt_param['Inner']['scheduler'] is None or opt_param['Inner']['scheduler']==''):
        raise ValueError("Lr scheduler unknown : %s"%opt_param['Inner']['scheduler'])

    #Warmup
    if opt_param['Inner']['warmup']['multiplier']>=1:
        from warmup_scheduler import GradualWarmupScheduler
        inner_scheduler=GradualWarmupScheduler(inner_opt, 
            multiplier=opt_param['Inner']['warmup']['multiplier'], 
            total_epoch=opt_param['Inner']['warmup']['epochs'], 
            after_scheduler=inner_scheduler)

    #Meta Opt
    hyper_param = list(model['data_aug'].parameters())
    if hp_opt : #(deprecated)
        for param_group in diffopt.param_groups: 
            # print(param_group)
            for param in hp_opt:
                param_group[param]=torch.tensor(param_group[param]).to(device).requires_grad_()
                hyper_param += [param_group[param]]
    meta_opt = torch.optim.Adam(hyper_param, lr=opt_param['Meta']['lr']) 

    #Meta-Scheduler (deprecated)
    meta_scheduler=None
    if opt_param['Meta']['scheduler']=='multiStep':
        meta_scheduler=torch.optim.lr_scheduler.MultiStepLR(meta_opt, 
            milestones=[int(epochs/3), int(epochs*2/3)],# int(epochs*2.7/3)], 
            gamma=3.16)
    elif opt_param['Meta']['scheduler'] is not None:
        raise ValueError("Lr scheduler unknown : %s"%opt_param['Meta']['scheduler'])

    model.train()
    meta_opt.zero_grad()

    for epoch in range(1, epochs+1):
        t0 = time.perf_counter()
        val_loss=None
       
        #Cross-Validation
        #dl_train, dl_val = cvs.next_split()
        #dl_val_it = iter(dl_val)

        for i, (xs, ys) in enumerate(dl_train):
            xs, ys = xs.to(device), ys.to(device)

            if(unsup_loss==0):
            #Methode uniforme
                logits = model(xs)  # modified `params` can also be passed as a kwarg
                loss = F.cross_entropy(F.log_softmax(logits, dim=1), ys, reduction='none')  # no need to call loss.backwards()

                if model._data_augmentation: #Weight loss
                    w_loss = model['data_aug'].loss_weight()#.to(device)
                    loss = loss * w_loss
                loss = loss.mean()
            
            else:
            #Methode mixed
                loss = mixed_loss(xs, ys, model, unsup_factor=unsup_loss, augment=augment_loss)

            # print_graph(loss, '../samples/pytorch_WRN') #to visualize computational graph
            # sys.exit()

            # t = time.process_time()
            diffopt.step(loss)#(opt.zero_grad, loss.backward, opt.step)
            # print(len(model['model']['functional']._fast_params),"step", time.process_time()-t)

            if(high_grad_track and i>0 and i%inner_it==0 and epoch>=opt_param['Meta']['epoch_start']): #Perform Meta step
                val_loss = compute_vaLoss(model=model, dl_it=dl_val_it, dl=dl_val) + model['data_aug'].reg_loss(opt_param['Meta']['reg_factor'])
                model.train()
                #print_graph(val_loss) #to visualize computational graph
                val_loss.backward()

                torch.nn.utils.clip_grad_norm_(model['data_aug'].parameters(), max_norm=max_grad, norm_type=2) #Prevent exploding grad with RNN

                # print("Grad mix",model['data_aug']["temp"].grad)
                # prv_param=model['data_aug']._params
                meta_opt.step()
                # kl_log["prob"].append(F.kl_div(prv_param["prob"],model['data_aug']["prob"], reduction='batchmean').item())
                # kl_log["mag"].append(F.kl_div(prv_param["mag"],model['data_aug']["mag"], reduction='batchmean').item())

                #Adjust Hyper-parameters
                model['data_aug'].adjust_param()
                if hp_opt: 
                    for param_group in diffopt.param_groups: 
                        for param in hp_opt:
                            param_group[param].data = param_group[param].data.clamp(min=1e-5)

                #Reset gradients
                diffopt.detach_()
                model['model'].detach_()
                meta_opt.zero_grad()

            elif not high_grad_track or epoch<opt_param['Meta']['epoch_start']:
                diffopt.detach_()
                model['model'].detach_()
                meta_opt.zero_grad()
                
        tf = time.perf_counter()

        #Schedulers
        if inner_scheduler is not None:
            inner_scheduler.step()
            #Transfer inner_opt lr to diffopt
            for diff_param_group in diffopt.param_groups:
                for param_group in inner_opt.param_groups:
                    diff_param_group['lr'] = param_group['lr']
        if meta_scheduler is not None:
            meta_scheduler.step()
        
        # if epoch<epochs/3:
        #     model['data_aug']['temp'].data=torch.tensor(0.5, device=device)
        # elif epoch>epochs/3 and epoch<(epochs*2/3):
        #     model['data_aug']['temp'].data=torch.tensor(0.75, device=device)
        # elif epoch>(epochs*2/3):
        #     model['data_aug']['temp'].data=torch.tensor(1.0, device=device)
        # model['data_aug']['temp'].data=torch.tensor(1./3+2/3*(epoch/epochs), device=device)
        # print('Temp',model['data_aug']['temp'])

        if (save_sample_freq and epoch%save_sample_freq==0): #Data sample saving
                try:
                    viz_sample_data(imgs=xs, labels=ys, fig_name='../samples/data_sample_epoch{}_noTF'.format(epoch))
                    model.train()
                    viz_sample_data(imgs=model['data_aug'](xs), labels=ys, fig_name='../samples/data_sample_epoch{}'.format(epoch), weight_labels=model['data_aug'].loss_weight())
                    model.eval()
                except:
                    print("Couldn't save samples epoch %d : %s"%(epoch, str(sys.exc_info()[1])))
                    pass

        if(not val_loss): #Compute val loss for logs
            val_loss = compute_vaLoss(model=model, dl_it=dl_val_it, dl=dl_val)

        # Test model
        accuracy, f1 =test(model)
        model.train()

        #### Log ####
        param = [{'p': p.item(), 'm':model['data_aug']['mag'].item()} for p in model['data_aug']['prob']] if model['data_aug']._shared_mag else [{'p': p.item(), 'm': m.item()} for p, m in zip(model['data_aug']['prob'], model['data_aug']['mag'])]
        data={
            "epoch": epoch,
            "train_loss": loss.item(),
            "val_loss": val_loss.item(),
            "acc": accuracy,
            "f1": f1.tolist(),
            "time": tf - t0,

            "param": param,
        }
        if not model['data_aug']._fixed_temp: data["temp"]=model['data_aug']['temp'].item()
        if hp_opt : data["opt_param"]=[{'lr': p_grp['lr'], 'momentum': p_grp['momentum']} for p_grp in diffopt.param_groups]
        log.append(data)
        #############
        #### Print ####
        if(print_freq and epoch%print_freq==0):
            print('-'*9)
            print('Epoch : %d/%d'%(epoch,epochs))
            print('Time : %.00f'%(tf - t0))
            print('Train loss :',loss.item(), '/ val loss', val_loss.item())
            print('Accuracy max:', max([x["acc"] for x in log]))
            print('F1 :', ["{0:0.4f}".format(i) for i in f1])
            print('Data Augmention : {} (Epoch {})'.format(model._data_augmentation, dataug_epoch_start))
            if not model['data_aug']._fixed_prob: print('TF Proba :', ["{0:0.4f}".format(p) for p in model['data_aug']['prob']])
            #print('proba grad',model['data_aug']['prob'].grad)
            if not model['data_aug']._fixed_mag: 
                if model['data_aug']._shared_mag:
                    print('TF Mag :', "{0:0.4f}".format(model['data_aug']['mag']))
                else:
                    print('TF Mag :', ["{0:0.4f}".format(m) for m in model['data_aug']['mag']])
            #print('Mag grad',model['data_aug']['mag'].grad)
            if not model['data_aug']._fixed_temp: print('Temp:', model['data_aug']['temp'].item())
            #print('Reg loss:', model['data_aug'].reg_loss().item())
            # if len(kl_log["prob"])!=0:
            #     print("KL prob : mean %f, std %f, max %f, min %f"%(np.mean(kl_log["prob"]), np.std(kl_log["prob"]), max(kl_log["prob"]), min(kl_log["prob"])))
            #     print("KL mag : mean %f, std %f, max %f, min %f"%(np.mean(kl_log["mag"]), np.std(kl_log["mag"]), max(kl_log["mag"]), min(kl_log["mag"])))
            #     kl_log={"prob":[], "mag":[]}
                
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
                    grad_callback=(lambda grads: clip_norm(grads, max_norm=max_grad)),
                    track_higher_grads=high_grad_track)

    aug_acc, aug_f1 = test(model, augment=augment_loss)

    return log, aug_acc

#OLD
# def run_simple_smartaug(model, opt_param, epochs=1, inner_it=1, print_freq=1, unsup_loss=1):
#     """Simple training of an augmented model with higher.

#             This function is intended to be used with Augmented_model containing an Higher_model (see dataug.py).
#             Ex : Augmented_model(Data_augV5(...), Higher_model(model))

#             Training loss can either be computed directly from augmented inputs (unsup_loss=0).
#             However, it is recommended to use the mixed loss computation, which combine original and augmented inputs to compute the loss (unsup_loss>0).

#             Does not support LR scheduler.

#         Args:
#             model (nn.Module): Augmented model to train.
#             opt_param (dict): Dictionnary containing optimizers parameters.
#             epochs (int): Number of epochs to perform. (default: 1)
#             inner_it (int): Number of inner iteration before a meta-step. 0 inner iteration means there's no meta-step. (default: 1)
#             print_freq (int): Number of epoch between display of the state of training. If set to None, no display will be done. (default:1)
#             unsup_loss (float): Proportion of the unsup_loss loss added to the supervised loss. If set to 0, the loss is only computed on augmented inputs. (default: 1)
            
#         Returns:
#             (dict) A dictionary containing a whole state of the trained network.
#     """
#     device = next(model.parameters()).device

#     ## Optimizers ##
#     hyper_param = list(model['data_aug'].parameters())
#     model.start_bilevel_opt(inner_it=inner_it, hp_list=hyper_param, opt_param=opt_param, dl_val=dl_val)

#     model.train()

#     for epoch in range(1, epochs+1):
#         t0 = time.process_time()
       
#         for i, (xs, ys) in enumerate(dl_train):
#             xs, ys = xs.to(device), ys.to(device)
            
#             #Methode mixed
#             loss = mixed_loss(xs, ys, model, unsup_factor=unsup_loss)

#             model.step(loss) #(opt.zero_grad, loss.backward, opt.step) + automatic meta-optimisation
                
#         tf = time.process_time()

#         #### Print ####
#         if(print_freq and epoch%print_freq==0):
#             print('-'*9)
#             print('Epoch : %d/%d'%(epoch,epochs))
#             print('Time : %.00f'%(tf - t0))
#             print('Train loss :',loss.item(), '/ val loss', model.val_loss().item())
#             if not model['data_aug']._fixed_prob: print('TF Proba :', model['data_aug']['prob'].data)
#             if not model['data_aug']._fixed_mag: print('TF Mag :', model['data_aug']['mag'].data)
#             if not model['data_aug']._fixed_temp: print('Temp:', model['data_aug']['temp'].item())
#         #############

#     return model['model'].state_dict()