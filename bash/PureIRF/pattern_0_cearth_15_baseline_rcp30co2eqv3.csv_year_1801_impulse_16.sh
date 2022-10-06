#! /bin/bash


######## login
#SBATCH --job-name=im_168
#SBATCH --output=./job-outs/PureIRF/pattern_0_cearth_15_baseline_rcp30co2eqv3.csv_year_1801/mercury_impulse_16.out
#SBATCH --error=./job-outs/PureIRF/pattern_0_cearth_15_baseline_rcp30co2eqv3.csv_year_1801/mercury_impulse_16.err


#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=7-00:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0

echo "$SLURM_JOB_NAME"

echo "Program starts $(date)"

python3 /home/bcheng4/QuantClimateChange/Model_Erik_PureIRF_graphsize.py --pattern 0 --cearth 15 --impulse 16 --baseline rcp30co2eqv3.csv --year 1801

echo "Program ends $(date)"

