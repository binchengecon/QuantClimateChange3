#! /bin/bash

python_name="HJB_suri_maxarray.py"
# python_name="HJB_suri_mutefrac.py"
# python_name="HJB_suri_muteboth.py"
action_name=$python_name
# ceartharray=(35 10)
# taucarray=(1886 6603)
ceartharray=(35)
taucarray=(6603)
maxiter=100000
fraction=0.05
simutime=15

hXarr=(0.2 4.0 10)
Xminarr=(0.00000001 0.0 0.00000001)
Xmaxarr=(5 400 400)

count=0
for cearth in ${ceartharray[@]}; do
    for tauc in ${taucarray[@]}; do
        mkdir -p ./job-outs/${action_name}/cearth_${cearth}_tauc_${tauc}/

        if [ -f ./bash/${action_name}/cearth_${cearth}_tauc_${tauc}_${simutime}_${Xmaxarr[0]}_${Xmaxarr[1]}_${Xmaxarr[2]}.sh ]; then
            rm ./bash/${action_name}/cearth_${cearth}_tauc_${tauc}_${simutime}_${Xmaxarr[0]}_${Xmaxarr[1]}_${Xmaxarr[2]}.sh
        fi

        mkdir -p ./bash/${action_name}/

        touch ./bash/${action_name}/cearth_${cearth}_tauc_${tauc}_${simutime}_${Xmaxarr[0]}_${Xmaxarr[1]}_${Xmaxarr[2]}.sh

        tee -a ./bash/${action_name}/cearth_${cearth}_tauc_${tauc}_${simutime}_${Xmaxarr[0]}_${Xmaxarr[1]}_${Xmaxarr[2]}.sh <<EOF
#! /bin/bash


######## login
#SBATCH --job-name=im_$count
#SBATCH --output=./job-outs/${action_name}/cearth_${cearth}_tauc_${tauc}/mercury_${maxiter}_${simutime}_${Xmaxarr[0]}_${Xmaxarr[1]}_${Xmaxarr[2]}.out
#SBATCH --error=./job-outs/${action_name}/cearth_${cearth}_tauc_${tauc}/mercury_${maxiter}_${simutime}_${Xmaxarr[0]}_${Xmaxarr[1]}_${Xmaxarr[2]}.err


#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=3
#SBATCH --mem=5G
#SBATCH --time=7-00:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0

echo "\$SLURM_JOB_NAME"

echo "Program starts \$(date)"

python3 /home/bcheng4/QuantClimateChange/$python_name --maxiter ${maxiter}  --simutime ${simutime} --Xmaxarr ${Xmaxarr[@]} --hXarr ${hXarr[@]}

echo "Program ends \$(date)"

EOF

        sbatch ./bash/${action_name}/cearth_${cearth}_tauc_${tauc}_${simutime}_${Xmaxarr[0]}_${Xmaxarr[1]}_${Xmaxarr[2]}.sh
        count=$(($count + 1))
    done
done
