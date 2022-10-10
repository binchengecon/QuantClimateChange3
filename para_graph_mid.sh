#! /bin/bash

action_name="PureIRF"
server_name="graph"
# python_name="Model_Erik_PureIRF_graphsize.py"
python_name="Model_Erik_PureIRF_graphsize2.py"


# ceartharray=(0.3725 0.3916 15)
ceartharray=(0.3916 15)
impulsearray=($(seq 0 50))
# impulsearray=($(seq 61 100 ))
# declare -a baselinearray=("carbonvoid.csv" "rcp85co2eqv3.csv" "rcp60co2eqv3.csv" "rcp45co2eqv3.csv" "rcp30co2eqv3.csv" "rcp00co2eqv3.csv")
declare -a baselinearray=("carbonvoid.csv" "rcp00co2eqv3.csv" "rcp30co2eqv3.csv" "rcp45co2eqv3.csv" "rcp60co2eqv3.csv")
# declare -a baselinearray=("carbonvoid.csv" "rcp85co2eqv3.csv" "rcp60co2eqv3.csv" "rcp45co2eqv3.csv" "rcp30co2eqv3.csv" "rcp00co2eqv3.csv")
# declare -a baselinearray=("carbonvoid.csv" "rcp85co2eqv3.csv")
# declare -a baselinearray=("rcp60co2eqv3.csv" "rcp45co2eqv3.csv")
# declare -a baselinearray=("rcp30co2eqv3.csv" "rcp00co2eqv3.csv")
# yeararray=(1801 1900 2000 2010 2020 2030)
tem_ylim_lower_array=(-11 -8 -5 -2.5 -1.5 -0.0025 -0.0025 -0.0025 -0.0025 -0.0025) 
tem_ylim_upper_array=(3 2 1.5 0.8 0.3 0.0125 0.0125 0.015 0.015 0.015)
#cearth part       #0.3916 0.3916 0.3916 0.3916 0.3916 15 15 15 15 15 
#baseline part     # "carbonvoid.csv" "rcp00co2eqv3.csv" "rcp30co2eqv3.csv" "rcp45co2eqv3.csv" "rcp60co2eqv3.csv" #"carbonvoid.csv" "rcp00co2eqv3.csv" "rcp30co2eqv3.csv" "rcp45co2eqv3.csv" "rcp60co2eqv3.csv"
yeararray=(2010)
pattern=1
count=0


for impulse in ${impulsearray[@]}
do
    ylimcount=0
    for cearth in ${ceartharray[@]}
    do
        for baseline in ${baselinearray[@]}
        do
            for year in ${yeararray[@]}
            do
                
                mkdir -p ./job-outs/${action_name}/pattern_${pattern}_cearth_${cearth}_baseline_${baseline}_year_${year}/
                
                if [ -f ./bash/${action_name}/pattern_${pattern}_cearth_${cearth}_baseline_${baseline}_year_${year}_impulse_${impulse}.sh ]
                then
                    rm ./bash/${action_name}/pattern_${pattern}_cearth_${cearth}_baseline_${baseline}_year_${year}_impulse_${impulse}.sh
                fi
                
                mkdir -p ./bash/${action_name}/
                
                touch ./bash/${action_name}/pattern_${pattern}_cearth_${cearth}_baseline_${baseline}_year_${year}_impulse_${impulse}.sh
                
                        tee -a ./bash/${action_name}/pattern_${pattern}_cearth_${cearth}_baseline_${baseline}_year_${year}_impulse_${impulse}.sh << EOF
#! /bin/bash


######## login
#SBATCH --job-name=im_$count
#SBATCH --output=./job-outs/${action_name}/pattern_${pattern}_cearth_${cearth}_baseline_${baseline}_year_${year}/mercury_impulse_${impulse}.out
#SBATCH --error=./job-outs/${action_name}/pattern_${pattern}_cearth_${cearth}_baseline_${baseline}_year_${year}/mercury_impulse_${impulse}.err


#SBATCH --time=36:00:00
#SBATCH --partition=broadwl
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=20
#SBATCH --mem-per-cpu=2000

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0

echo "\$SLURM_JOB_NAME"

echo "Program starts \$(date)"

python3 /home/bcheng4/QuantClimateChange/$python_name --pattern ${pattern} --cearth ${cearth} --impulse ${impulse} --baseline ${baseline} --year ${year} --tem_ylim_lower ${tem_ylim_lower_array[$ylimcount]} --tem_ylim_upper ${tem_ylim_upper_array[$ylimcount]}

echo "Program ends \$(date)"

EOF
                
                sbatch ./bash/${action_name}/pattern_${pattern}_cearth_${cearth}_baseline_${baseline}_year_${year}_impulse_${impulse}.sh
                count=$(($count+1))
            done
            ylimcount=$(($ylimcount+1))
        done
    done
done

