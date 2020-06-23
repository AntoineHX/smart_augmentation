#!/bin/bash
#SBATCH --gres=gpu:1 #gpu:v100l:1    # https://docs.computecanada.ca/wiki/Using_GPUs_with_Slurm
#SBATCH --cpus-per-task=6 #6 (V100:8)  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M  #32000M       # Memory proportional to CPUs: 32000 Cedar, 64000 Graham.
#SBATCH --account=def-mpederso
#SBATCH --time=8:00:00
#SBATCH --job-name=Classic-WRN50_2-TinyImageNet
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
    -m wide_resnet50_2 \
    -ep 40 \
    -pt true \
    -pf __pretrained