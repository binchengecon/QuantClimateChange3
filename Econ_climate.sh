#! /bin/bash

# action_name="EconClimate"
# # python_name="Model_Erik_PureIRF_graphsize.py"
# python_name="Econ_ClimateChange.py"

# ceartharray=(0.3916 15 63)
# taucarray=(30 6603)
# maxiter=50000
# fraction=0.05

# hXarr=(0.2 4.0 40.0)
# Xminarr=(0.00000001 0.0 0.00000001)
# Xmaxarr=(9.00 4.0 0.0 3.0)

# count=0
# for cearth in ${ceartharray[@]}; do
#     for tauc in ${taucarray[@]}; do
#         mkdir -p ./job-outs/${action_name}/cearth_${cearth}_tauc_${tauc}/

#         if [ -f ./bash/${action_name}/cearth_${cearth}_tauc_${tauc}.sh ]; then
#             rm ./bash/${action_name}/cearth_${cearth}_tauc_${tauc}.sh
#         fi

#         mkdir -p ./bash/${action_name}/

#         touch ./bash/${action_name}/cearth_${cearth}_tauc_${tauc}.sh

#         tee -a ./bash/${action_name}/cearth_${cearth}_tauc_${tauc}.sh <<EOF
# #! /bin/bash

# ######## login
# #SBATCH --job-name=im_$count
# #SBATCH --output=./job-outs/${action_name}/cearth_${cearth}_tauc_${tauc}/mercury_${maxiter}.out
# #SBATCH --error=./job-outs/${action_name}/cearth_${cearth}_tauc_${tauc}/mercury_${maxiter}.err

# #SBATCH --account=pi-lhansen
# #SBATCH --partition=standard
# #SBATCH --cpus-per-task=1
# #SBATCH --mem=1G
# #SBATCH --time=7-00:00:00

# ####### load modules
# module load python/booth/3.8/3.8.5  gcc/9.2.0

# echo "\$SLURM_JOB_NAME"

# echo "Program starts \$(date)"

# python3 /home/bcheng4/QuantClimateChange/$python_name --cearth ${cearth} --tauc ${tauc} --maxiter ${maxiter} --fraction ${fraction}

# echo "Program ends \$(date)"

# EOF

#         sbatch ./bash/${action_name}/cearth_${cearth}_tauc_${tauc}.sh
#         count=$(($count + 1))
#     done
# done

# action_name="EconClimate_Old"
# # python_name="Model_Erik_PureIRF_graphsize.py"
# python_name="Econ_ClimateChange_old.py"

# ceartharray=(35 10)
# taucarray=(1886 6603)
# maxiter=50000
# fraction=0.05

# hXarr=(0.2 4.0 40.0)
# Xminarr=(0.00000001 0.0 0.00000001)
# Xmaxarr=(9.00 4.0 0.0 3.0)

# count=0
# for cearth in ${ceartharray[@]}; do
#     for tauc in ${taucarray[@]}; do
#         mkdir -p ./job-outs/${action_name}/cearth_${cearth}_tauc_${tauc}/

#         if [ -f ./bash/${action_name}/cearth_${cearth}_tauc_${tauc}.sh ]; then
#             rm ./bash/${action_name}/cearth_${cearth}_tauc_${tauc}.sh
#         fi

#         mkdir -p ./bash/${action_name}/

#         touch ./bash/${action_name}/cearth_${cearth}_tauc_${tauc}.sh

#         tee -a ./bash/${action_name}/cearth_${cearth}_tauc_${tauc}.sh <<EOF
# #! /bin/bash

# ######## login
# #SBATCH --job-name=im_$count
# #SBATCH --output=./job-outs/${action_name}/cearth_${cearth}_tauc_${tauc}/mercury_${maxiter}.out
# #SBATCH --error=./job-outs/${action_name}/cearth_${cearth}_tauc_${tauc}/mercury_${maxiter}.err

# #SBATCH --account=pi-lhansen
# #SBATCH --partition=standard
# #SBATCH --cpus-per-task=3
# #SBATCH --mem=5G
# #SBATCH --time=7-00:00:00

# ####### load modules
# module load python/booth/3.8/3.8.5  gcc/9.2.0

# echo "\$SLURM_JOB_NAME"

# echo "Program starts \$(date)"

# python3 /home/bcheng4/QuantClimateChange/$python_name --cearth ${cearth} --tauc ${tauc} --maxiter ${maxiter} --fraction ${fraction}

# echo "Program ends \$(date)"

# EOF

#         sbatch ./bash/${action_name}/cearth_${cearth}_tauc_${tauc}.sh
#         count=$(($count + 1))
#     done
# done

# action_name="EconClimate_Suri_mutealpha"
# action_name="EconClimate_Suri_mutealpha"
# python_name="Model_Erik_PureIRF_graphsize.py"
python_name="HJB_suri_mutealpha.py"
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

hXarr=(0.2 4.0 40.0)
Xminarr=(0.00000001 0.0 0.00000001)
Xmaxarr=(9.00 4.0 0.0 3.0)

count=0
for cearth in ${ceartharray[@]}; do
    for tauc in ${taucarray[@]}; do
        mkdir -p ./job-outs/${action_name}/cearth_${cearth}_tauc_${tauc}_${simutime}/

        if [ -f ./bash/${action_name}/cearth_${cearth}_tauc_${tauc}_${simutime}.sh ]; then
            rm ./bash/${action_name}/cearth_${cearth}_tauc_${tauc}_${simutime}.sh
        fi

        mkdir -p ./bash/${action_name}/

        touch ./bash/${action_name}/cearth_${cearth}_tauc_${tauc}_${simutime}.sh

        tee -a ./bash/${action_name}/cearth_${cearth}_tauc_${tauc}_${simutime}.sh <<EOF
#! /bin/bash


######## login
#SBATCH --job-name=im_$count
#SBATCH --output=./job-outs/${action_name}/cearth_${cearth}_tauc_${tauc}/mercury_${maxiter}_${simutime}.out
#SBATCH --error=./job-outs/${action_name}/cearth_${cearth}_tauc_${tauc}/mercury_${maxiter}_${simutime}.err


#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=3
#SBATCH --mem=5G
#SBATCH --time=7-00:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0

echo "\$SLURM_JOB_NAME"

echo "Program starts \$(date)"

python3 /home/bcheng4/QuantClimateChange/$python_name --maxiter ${maxiter}  --simutime ${simutime}

echo "Program ends \$(date)"

EOF

        sbatch ./bash/${action_name}/cearth_${cearth}_tauc_${tauc}_${simutime}.sh
        count=$(($count + 1))
    done
done
