#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --account=def-mpederso
#SBATCH --time=00:03:00
#SBATCH --job-name=setup
#SBATCH --output=log/%x-%j.out

# Setup
#sudo rm -rf ~/dataug/
module load python/3.7.4
virtualenv --no-download ~/dataug
source ~/dataug/bin/activate
pip install --no-index --upgrade pip

pip install --no-index torch torchvision #torchviz

pip install kornia # --no-deps ?

git clone https://github.com/facebookresearch/higher.git
cd higher
pip install .
cd ..

pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
pip install efficientnet_pytorch

pip install --no-index matplotlib scipy
#