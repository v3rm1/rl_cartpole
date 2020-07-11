#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=dqn_cartpole
#SBATCH --mem=7000
module load Python/3.6.4-foss-2018a
module load CUDA/9.1.85
module load Boost/1.66.0-foss-2018a-Python-3.6.4
module load TensorFlow/2.1.0-fosscuda-2019b-Python-3.7.4
pip install pycuda --user
pip install keras --user
pip install matplotlib --user
pip install gym --user
pip install pandas --user
python3 ./dqn/dqn_cartpole.py
