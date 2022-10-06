#! /bin/bash


######## login
#SBATCH --job-name=im_157
#SBATCH --output=./job-outs/PureIRF/pattern_2_cearth_15_baseline_rcp45co2eqv3.csv_year_1801/mercury_impulse_15.out
#SBATCH --error=./job-outs/PureIRF/pattern_2_cearth_15_baseline_rcp45co2eqv3.csv_year_1801/mercury_impulse_15.err


#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=7-00:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0

echo "$SLURM_JOB_NAME"

echo "Program starts $(date)"

python3 /home/bcheng4/QuantClimateChange/Model_Erik_PureIRF_graphsize.py --pattern 2 --cearth 15 --impulse 15 --baseline rcp45co2eqv3.csv --year 1801

echo "Program ends $(date)"

