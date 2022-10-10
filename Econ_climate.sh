#! /bin/bash

action_name="EconClimate"
# python_name="Model_Erik_PureIRF_graphsize.py"
python_name="Econ_ClimateChange.py"

ceartharray=(0.3916)
taucarray=(30)
maxiter=5000
fraction=0.05

count=0
for cearth in ${ceartharray[@]}; do
    for tauc in ${taucarray[@]}; do
        mkdir -p ./job-outs/${action_name}/cearth_${cearth}_tauc_${tauc}/

        if [ -f ./bash/${action_name}/cearth_${cearth}_tauc_${tauc}.sh ]; then
            rm ./bash/${action_name}/cearth_${cearth}_tauc_${tauc}.sh
        fi

        mkdir -p ./bash/${action_name}/

        touch ./bash/${action_name}/cearth_${cearth}_tauc_${tauc}.sh

        tee -a ./bash/${action_name}/cearth_${cearth}_tauc_${tauc}.sh <<EOF
#! /bin/bash


######## login
#SBATCH --job-name=im_$count
#SBATCH --output=./job-outs/${action_name}/cearth_${cearth}_tauc_${tauc}/mercury2.out
#SBATCH --error=./job-outs/${action_name}/cearth_${cearth}_tauc_${tauc}/mercury2.err


#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=7-00:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0

echo "\$SLURM_JOB_NAME"

echo "Program starts \$(date)"

python3 /home/bcheng4/QuantClimateChange/$python_name --cearth ${cearth} --tauc ${tauc} --maxiter ${maxiter} --fraction ${fraction}

echo "Program ends \$(date)"

EOF

        sbatch ./bash/${action_name}/cearth_${cearth}_tauc_${tauc}.sh
        count=$(($count + 1))
    done
done
