#!/bin/bash
#PBS -l walltime=21:0:0
#PBS -l select=1:ncpus=127:mem=256gb
#PBS -N custom_op
#PBS -o logs/
#PBS -e logs/

# Navigate to the directory where the SIF file is located
cd $PBS_O_WORKDIR
cd ../

# Run the Apptainer container and execute the subsequent commands
apptainer exec apptainer.sif /bin/bash -c "
  source /root/.bashrc
  cd src
  python3 custom_electron_op.py > ../logs/custom_electron_op.log 2>&1
"