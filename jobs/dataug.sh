#!/bin/bash
#SBATCH --gres=gpu:v100l:1    # https://docs.computecanada.ca/wiki/Using_GPUs_with_Slurm
#SBATCH --cpus-per-task=6 #6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M  # Memory proportional to CPUs: 32000 Cedar, 64000 Graham.
#SBATCH --account=def-mpederso
#SBATCH --time=10:00:00
#SBATCH --job-name=Dataug-ResNet18
# --exclusive #In case of Out of Memory ?
#SBATCH --output=log/%x-%j.out
#SBATCH --mail-user=harle.collette.antoine@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# Setup
source ~/dataug/bin/activate

#Execute
# echo $(pwd) = /home/antoh/projects/def-mpederso/antoh/smart_augmentation/jobs
cd ../higher/smart_aug/

python test_dataug.py \
    -a true \
    -m resnet18 \
    -ep 50 \
    -K 1 \
    -N 3 \
    -tfc ../config/invScale_wide_tf_config.json \
    -ds CIFAR10 \
    -al 1 \
    -pf __test_noCopy