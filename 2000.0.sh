#! /bin/bash


######## login
#SBATCH --job-name=im_0
#SBATCH --output=./job-outs/Econ_ClimateChange_mute.py/cearth_35_tauc_6603/mercury_100000_15_0_0_20.0 400.0 2000.0.out
#SBATCH --error=./job-outs/Econ_ClimateChange_mute.py/cearth_35_tauc_6603/mercury_100000_15_0_0_20.0 400.0 2000.0.err


#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=3
#SBATCH --mem=5G
#SBATCH --time=7-00:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0

echo "$SLURM_JOB_NAME"

echo "Program starts $(date)"

python3 /home/bcheng4/QuantClimateChange/Econ_ClimateChange_mute.py --maxiter 100000  --simutime 15 --cearth 35 --tauc 6603 --maxiter 100000 --fraction 0.05 --alphamute 0 --betamute 0  --Xmaxarr 20.0 400.0 2000.0

echo "Program ends $(date)"

