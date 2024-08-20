#!/bin/bash
#SBATCH --gres=gpu:v100l:1    # https://docs.computecanada.ca/wiki/Using_GPUs_with_Slurm
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --account=def-mpederso
#SBATCH --time=300:00:00
#SBATCH --job-name=imageNet-RandAug-ResNet18
#SBATCH --output=log/%x-%j.out
#SBATCH --mail-user=harle.collette.antoine@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#ImageNet Extract
echo "Extracting ImageNet..."
mkdir $SLURM_TMPDIR/data
time tar -xf ~/projects/def-mpederso/dataset/imagenet/imagenet.tar -C $SLURM_TMPDIR/data

echo "Executing code..."
# Setup
source ~/dataug/bin/activate

#Execute
# echo $(pwd) = /home/antoh/projects/def-mpederso/antoh/smart_augmentation/jobs
cd ../higher/smart_aug/

time python test_dataug.py \
    -a true \
    -m resnet18 \
    -ep 40 \
    -K 0 \
    -N 3 \
    -tfc ../config/invScale_wide_tf_config.json \
    -dr $SLURM_TMPDIR/data \
    -ds ImageNet \
    -al 10 \
    -pt true \
    -pf __pretrained_AL10