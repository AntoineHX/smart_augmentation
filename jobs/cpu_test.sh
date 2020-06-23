#!/bin/bash
#SBATCH --cpus-per-task=6 #6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M  #32000M       # Memory proportional to CPUs: 32000 Cedar, 64000 Graham.
#SBATCH --account=def-mpederso
#SBATCH --time=01:00:00
#SBATCH --job-name=test_dataug
#SBATCH --output=log/%x-%j.out


# Setup
source ~/dataug/bin/activate

#Execute
# echo $(pwd) = /home/antoh/projects/def-mpederso/antoh/smart_augmentation/jobs
cd ../higher/smart_aug/

python test_dataug.py \
    -dv 'cpu' \
    -a true \
    -m LeNet \
    -ep 2 \
    -K 1 \
    -N 3 \
    -tfc ../config/wide_geom_tf_config.json \
    -ms 1 \
    -ls true \
    -pf __test