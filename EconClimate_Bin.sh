#! /bin/bash

python_name="HJB_Bin.py"
action_name=$python_name
ceartharray=(35)
taucarray=(6603)
maxiterarr=(5000 30000 60000)
fractionarr=(0.05 0.1)
epsilonarr=(0.05 0.1)
simutime=600
count=0

declare -A hXarr1=([0]=0.2 [1]=4.0 [2]=40.0)
declare -A hXarr2=([0]=0.1 [1]=2.0 [2]=20.0)
hXarrays=(hXarr1 hXarr2)

declare -A Xminarr1=([0]=0.0 [1]=200.0 [2]=0.0)
Xminarrays=(Xminarr1)

declare -A Xmaxarr1=([0]=10.0 [1]=400.0 [2]=2000.0)
declare -A Xmaxarr2=([0]=15.0 [1]=400.0 [2]=2000.0)
declare -A Xmaxarr3=([0]=17.5 [1]=400.0 [2]=2000.0)
declare -A Xmaxarr4=([0]=20.0 [1]=400.0 [2]=2000.0)

Xmaxarrays=(Xmaxarr1 Xmaxarr2 Xmaxarr3 Xmaxarr4)

for hXarri in "${hXarrays[@]}"; do
    for Xmaxarri in "${Xmaxarrays[@]}"; do
        for Xminarri in "${Xminarrays[@]}"; do

            declare -n hXtemp="$hXarri"
            declare -n Xmaxtemp="$Xmaxarri"
            declare -n Xmintemp="$Xminarri"

            for fraction in ${fractionarr[@]}; do
                epsilon=$fraction
                for maxiter in ${maxiterarr[@]}; do

                    for cearth in ${ceartharray[@]}; do
                        for tauc in ${taucarray[@]}; do

                            # echo "${hXtemp[@]}"
                            # echo "${Xmaxtemp[@]}"
                            # echo "${Xmintemp[@]}"
                            mkdir -p ./job-outs/${action_name}/cearth_${cearth}_tauc_${tauc}/

                            if
                                [ -f ./bash/${action_name}/cearth_${cearth}_tauc_${tauc}_${simutime}_${Xmintemp[0]},${Xmaxtemp[0]},${hXtemp[0]}_${Xmintemp[1]},${Xmaxtemp[1]},${hXtemp[1]}_${Xmintemp[2]},${Xmaxtemp[2]},${hXtemp[2]}_${fraction}_${epsilon}.sh ]
                            then
                                rm ./bash/${action_name}/cearth_${cearth}_tauc_${tauc}_${simutime}_${Xmintemp[0]},${Xmaxtemp[0]},${hXtemp[0]}_${Xmintemp[1]},${Xmaxtemp[1]},${hXtemp[1]}_${Xmintemp[2]},${Xmaxtemp[2]},${hXtemp[2]}_${fraction}_${epsilon}.sh
                            fi

                            mkdir -p ./bash/${action_name}/

                            touch ./bash/${action_name}/cearth_${cearth}_tauc_${tauc}_${simutime}_${Xmintemp[0]},${Xmaxtemp[0]},${hXtemp[0]}_${Xmintemp[1]},${Xmaxtemp[1]},${hXtemp[1]}_${Xmintemp[2]},${Xmaxtemp[2]},${hXtemp[2]}_${fraction}_${epsilon}.sh

                            tee -a ./bash/${action_name}/cearth_${cearth}_tauc_${tauc}_${simutime}_${Xmintemp[0]},${Xmaxtemp[0]},${hXtemp[0]}_${Xmintemp[1]},${Xmaxtemp[1]},${hXtemp[1]}_${Xmintemp[2]},${Xmaxtemp[2]},${hXtemp[2]}_${fraction}_${epsilon}.sh <<EOF
#! /bin/bash


######## login
#SBATCH --job-name=im_$count
#SBATCH --output=./job-outs/${action_name}/cearth_${cearth}_tauc_${tauc}/mercury_${maxiter}_${simutime}_${Xmintemp[0]},${Xmaxtemp[0]},${hXtemp[0]}_${Xmintemp[1]},${Xmaxtemp[1]},${hXtemp[1]}_${Xmintemp[2]},${Xmaxtemp[2]},${hXtemp[2]}_${fraction}_${epsilon}.out
#SBATCH --error=./job-outs/${action_name}/cearth_${cearth}_tauc_${tauc}/mercury_${maxiter}_${simutime}_${Xmintemp[0]},${Xmaxtemp[0]},${hXtemp[0]}_${Xmintemp[1]},${Xmaxtemp[1]},${hXtemp[1]}_${Xmintemp[2]},${Xmaxtemp[2]},${hXtemp[2]}_${fraction}_${epsilon}.err


#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=3
#SBATCH --mem=5G
#SBATCH --time=7-00:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0

echo "\$SLURM_JOB_NAME"

echo "Program starts \$(date)"

python3 /home/bcheng4/QuantClimateChange/$python_name --maxiter ${maxiter} --cearth ${cearth} --tauc ${tauc}  --simutime ${simutime} --Xmaxarr ${Xmaxtemp[@]} --Xminarr ${Xmintemp[@]} --hXarr ${hXtemp[@]} --fraction ${fraction} --epsilon ${epsilon} --filename ${python_name}

echo "Program ends \$(date)"

EOF

                            sbatch ./bash/${action_name}/cearth_${cearth}_tauc_${tauc}_${simutime}_${Xmintemp[0]},${Xmaxtemp[0]},${hXtemp[0]}_${Xmintemp[1]},${Xmaxtemp[1]},${hXtemp[1]}_${Xmintemp[2]},${Xmaxtemp[2]},${hXtemp[2]}_${fraction}_${epsilon}.sh
                            count=$(($count + 1))
                        done
                    done

                done

            done
        done
    done
done


