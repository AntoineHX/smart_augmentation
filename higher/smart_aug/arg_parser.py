import argparse

#Argparse
parser = argparse.ArgumentParser(description='Run smart augmentation')
parser.add_argument('-dv','--device', default='cuda', dest='device',
                    help='Device : cpu / cuda')
parser.add_argument('-dt','--dtype', default='FP32', dest='dtype',
                    help='Data type (Default: Float32)')

parser.add_argument('-m','--model', default='resnet18', dest='model',
                    help='Network')
parser.add_argument('-pt','--pretrained', default='', dest='pretrained',
                    help='Use pretrained weight if possible')

parser.add_argument('-ep','--epochs', type=int, default=10, dest='epochs',
                    help='epoch')
# parser.add_argument('-ot', '--optimizer', default='SGD', dest='opt_type',
#                     help='Model optimizer')
parser.add_argument('-lr', type=float, default=1e-1, dest='lr',
                    help='Model learning rate')
parser.add_argument('-mo', '--momentum', type=float, default=0.9, dest='momentum',
                    help='Momentum')
parser.add_argument('-dc', '--decay', type=float, default=0.0005, dest='decay',
                    help='Weight decay')
parser.add_argument('-ns','--nesterov', type=bool, default=False, dest='nesterov',
                    help='Nesterov momentum ?')
parser.add_argument('-sc', '--scheduler', default='cosine', dest='scheduler',
                    help='Model learning rate scheduler')
parser.add_argument('-wu', '--warmup', type=float, default=0, dest='warmup',
                    help='Warmup multiplier') 
                  

parser.add_argument('-a','--augment', type=bool, default=False, dest='augment',
                    help='Data augmentation ?')
parser.add_argument('-N', type=int, default=1,
                    help='Combination of TF')
parser.add_argument('-K', type=int, default=0,
                    help='Number inner iteration')
parser.add_argument('-al','--augment_loss', type=int, default=1, dest='augment_loss',
                    help='Number of augmented example for each sample in loss computation.')
parser.add_argument('-t', '--temp', type=float, default=0.5, dest='temp',
                    help='Probability distribution temperature')
parser.add_argument('-tfc','--tf_config', default='../config/invScale_wide_tf_config.json', dest='tf_config',
                    help='TF config')
parser.add_argument('-ls', '--learn_seq', type=bool, default=False, dest='learn_seq',
                    help='Learn order of application of TF (DataugV7-8) ?')
parser.add_argument('-fm', '--fixed_mag', type=bool, default=False, dest='fixed_mag',
                    help='Fixed magnitude when learning data augmentation ?')
parser.add_argument('-sm', '--shared_mag', type=bool, default=False, dest='shared_mag',
                    help='Shared magnitude when learning data augmentation ?')

# parser.add_argument('-mot', '--metaoptimizer', default='Adam', dest='meta_opt_type',
#                     help='Meta optimizer (Augmentations)')
parser.add_argument('-mlr', type=float, default=1e-2, dest='mlr',
                    help='Meta learning rate (Augmentations)')
parser.add_argument('-ms', type=int, default=0, dest='meta_epoch_start',
                    help='Epoch at which start meta learning')
parser.add_argument('-mr', type=float, default=0.001, dest='mag_reg',
                    help='Augmentation magnitudes regulation factor')

parser.add_argument('-rf','--res_folder', default='../res/', dest='res_folder',
                    help='Results folder')
parser.add_argument('-pf','--postfix', default='', dest='postfix',
                    help='Res postfix')

parser.add_argument('-dr','--dataroot', default='~/scratch/data', dest='dataroot',
                    help='Datasets folder')
parser.add_argument('-ds','--dataset', default='CIFAR10', dest='dataset',
                    help='Dataset')
parser.add_argument('-bs','--batch_size', type=int, default=256, dest='batch_size',
                    help='Batch size') #256 (WRN) / 512
parser.add_argument('-w','--workers', type=int, default=6, dest='workers',
                    help='Numer of workers (Nb CPU cores).')