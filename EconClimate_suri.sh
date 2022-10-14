#! /bin/bash

python_name="HJB_suri_maxarray.py"
# python_name="HJB_suri_mutefrac.py"
# python_name="HJB_suri_muteboth.py"
action_name=$python_name
# ceartharray=(35 10)
# taucarray=(1886 6603)
ceartharray=(35)
taucarray=(6603)
maxiter=10
fraction=0.1
epsilon=0.1
simutime=600

hXarr=(0.2 4.0 40)
Xminarr=(0.00000001 0.0 0.00000001)
Xmaxarr=(10 400 2000)

count=0
for cearth in ${ceartharray[@]}; do
    for tauc in ${taucarray[@]}; do
        mkdir -p ./job-outs/${action_name}/cearth_${cearth}_tauc_${tauc}/

        if [ -f ./bash/${action_name}/cearth_${cearth}_tauc_${tauc}_${simutime}_${Xmaxarr[0]}_${Xmaxarr[1]}_${Xmaxarr[2]}_${hXarr[0]}_${hXarr[1]}_${hXarr[2]}_${fraction}_${epsilon}.sh ]; then
            rm ./bash/${action_name}/cearth_${cearth}_tauc_${tauc}_${simutime}_${Xmaxarr[0]}_${Xmaxarr[1]}_${Xmaxarr[2]}_${hXarr[0]}_${hXarr[1]}_${hXarr[2]}_${fraction}_${epsilon}.sh
        fi

        mkdir -p ./bash/${action_name}/

        touch ./bash/${action_name}/cearth_${cearth}_tauc_${tauc}_${simutime}_${Xmaxarr[0]}_${Xmaxarr[1]}_${Xmaxarr[2]}_${hXarr[0]}_${hXarr[1]}_${hXarr[2]}_${fraction}_${epsilon}.sh

        tee -a ./bash/${action_name}/cearth_${cearth}_tauc_${tauc}_${simutime}_${Xmaxarr[0]}_${Xmaxarr[1]}_${Xmaxarr[2]}_${hXarr[0]}_${hXarr[1]}_${hXarr[2]}_${fraction}_${epsilon}.sh <<EOF
#! /bin/bash


######## login
#SBATCH --job-name=im_$count
#SBATCH --output=./job-outs/${action_name}/cearth_${cearth}_tauc_${tauc}/mercury_${maxiter}_${simutime}_${Xmaxarr[0]}_${Xmaxarr[1]}_${Xmaxarr[2]}_${hXarr[0]}_${hXarr[1]}_${hXarr[2]}_${fraction}_${epsilon}.out
#SBATCH --error=./job-outs/${action_name}/cearth_${cearth}_tauc_${tauc}/mercury_${maxiter}_${simutime}_${Xmaxarr[0]}_${Xmaxarr[1]}_${Xmaxarr[2]}_${hXarr[0]}_${hXarr[1]}_${hXarr[2]}_${fraction}_${epsilon}.err


#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=3
#SBATCH --mem=5G
#SBATCH --time=7-00:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0

echo "\$SLURM_JOB_NAME"

echo "Program starts \$(date)"

python3 /home/bcheng4/QuantClimateChange/$python_name --maxiter ${maxiter}  --simutime ${simutime} --Xmaxarr ${Xmaxarr[@]} --hXarr ${hXarr[@]} --fraction ${fraction} --epsilon ${epsilon}

echo "Program ends \$(date)"

EOF

        sbatch ./bash/${action_name}/cearth_${cearth}_tauc_${tauc}_${simutime}_${Xmaxarr[0]}_${Xmaxarr[1]}_${Xmaxarr[2]}_${hXarr[0]}_${hXarr[1]}_${hXarr[2]}_${fraction}_${epsilon}.sh
        count=$(($count + 1))
    done
done
