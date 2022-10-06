#! /bin/bash

action_name="PureIRF"
server_name="graph"
python_name="Model_Erik_PureIRF_graphsize.py"


declare -a titles=('Impulse Response of Temperature Anomaly' 'Impulse Response of Carbon Concentration Dynamics')
declare -a ylabels=('Temperature (K)' 'Carbon (ppm)' 'Emission (Gtc)' 'Degree (Celsius)')
ceartharray=(0.3725 0.3916 15)
impulsearray=($(seq 0 100 ))
declare -a baselinearray = ("carbonvoid.csv" "rcp85co2eqv3.csv" "rcp60co2eqv3.csv" "rcp45co2eqv3.csv" "rcp30co2eqv3.csv" "rcp00co2eqv3.csv")
yeararray = (1801 1900 2000 2010 2020 2030)
pattern=0
count=0


for title in ${titles[@]}
do
    for ylabel in ${ylabels[@]}
    do
        for cearth in ${ceartharray[@]}
        do
            for impulse in ${impulsearray[@]}
            do
                for baseline in ${baselinearray[@]}
                do
                    for year in ${yeararray[@]}
                    do
                        
                        mkdir -p ./job-outs/${action_name}/pattern_${pattern}_title_${title}_ylabel_${ylabel}_cearth_${cearth}_baseline_${baseline}_year_${year}/
                        
                        if [ -f ./bash/${action_name}/pattern_${pattern}_title_${title}_ylabel_${ylabel}_cearth_${cearth}_baseline_${baseline}_year_${year}_impulse_${impulse}.sh ]
                        then
                            rm ./bash/${action_name}/pattern_${pattern}_title_${title}_ylabel_${ylabel}_cearth_${cearth}_baseline_${baseline}_year_${year}_impulse_${impulse}.sh
                        fi
                        
                        mkdir -p ./bash/${action_name}/
                        
                        touch ./bash/${action_name}/pattern_${pattern}_title_${title}_ylabel_${ylabel}_cearth_${cearth}_baseline_${baseline}_year_${year}_impulse_${impulse}.sh
                        
                        tee -a ./bash/${action_name}/pattern_${pattern}_title_${title}_ylabel_${ylabel}_cearth_${cearth}_baseline_${baseline}_year_${year}_impulse_${impulse}.sh << EOF
#! /bin/bash


######## login
#SBATCH --job-name=impulse${actiontime}_$count
#SBATCH --output=./job-outs/${action_name}/pattern_${pattern}_title_${title}_ylabel_${ylabel}_cearth_${cearth}_baseline_${baseline}_year_${year}/mercury_impulse_${impulse}.out
#SBATCH --error=./job-outs/${action_name}/pattern_${pattern}_title_${title}_ylabel_${ylabel}_cearth_${cearth}_baseline_${baseline}_year_${year}/mercury_impulse_${impulse}.err


#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=3
#SBATCH --mem=16G
#SBATCH --time=7-00:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0

echo "\$SLURM_JOB_NAME"

echo "Program starts \$(date)"

python3 /home/bcheng4/QuantClimateChange/$python_name --pattern ${pattern} --title ${title} --ylabel ${ylabel} --cearth ${cearth} --impulse ${impulse} --baseline ${baseline} --year ${year}

echo "Program ends \$(date)"

EOF
                        
                        sbatch ./bash/${action_name}/pattern_${pattern}_title_${title}_ylabel_${ylabel}_cearth_${cearth}_baseline_${baseline}_year_${year}_impulse_${impulse}.sh
                        count=$(($count+1))
                    done
                done
            done
        done
    done
done


action_name="PureIRF"
server_name="graph"
python_name="Model_Erik_PureIRF_graphsize.py"


declare -a titles=('Impulse Response of Temperature Anomaly per Gtc' 'Impulse Response of Carbon Concentration Dynamics per Gtc')
declare -a ylabels=('Temperature (K)' 'Carbon (ppm)' 'Emission (Gtc)' 'Degree (Celsius)')
ceartharray=(0.3725 0.3916 15)
impulsearray=($(seq 0 100 ))
declare -a baselinearray = ("carbonvoid.csv" "rcp85co2eqv3.csv" "rcp60co2eqv3.csv" "rcp45co2eqv3.csv" "rcp30co2eqv3.csv" "rcp00co2eqv3.csv")
yeararray = (1801 1900 2000 2010 2020 2030)
patternarray=(0 1 2)
count=0
pattern=1

for title in ${titles[@]}
do
    for ylabel in ${ylabels[@]}
    do
        for cearth in ${ceartharray[@]}
        do
            for impulse in ${impulsearray[@]}
            do
                for baseline in ${baselinearray[@]}
                do
                    for year in ${yeararray[@]}
                    do
                        
                        mkdir -p ./job-outs/${action_name}/pattern_${pattern}_title_${title}_ylabel_${ylabel}_cearth_${cearth}_baseline_${baseline}_year_${year}/
                        
                        if [ -f ./bash/${action_name}/pattern_${pattern}_title_${title}_ylabel_${ylabel}_cearth_${cearth}_baseline_${baseline}_year_${year}_impulse_${impulse}.sh ]
                        then
                            rm ./bash/${action_name}/pattern_${pattern}_title_${title}_ylabel_${ylabel}_cearth_${cearth}_baseline_${baseline}_year_${year}_impulse_${impulse}.sh
                        fi
                        
                        mkdir -p ./bash/${action_name}/
                        
                        touch ./bash/${action_name}/pattern_${pattern}_title_${title}_ylabel_${ylabel}_cearth_${cearth}_baseline_${baseline}_year_${year}_impulse_${impulse}.sh
                        
                        tee -a ./bash/${action_name}/pattern_${pattern}_title_${title}_ylabel_${ylabel}_cearth_${cearth}_baseline_${baseline}_year_${year}_impulse_${impulse}.sh << EOF
#! /bin/bash


######## login
#SBATCH --job-name=impulse${actiontime}_$count
#SBATCH --output=./job-outs/${action_name}/pattern_${pattern}_title_${title}_ylabel_${ylabel}_cearth_${cearth}_baseline_${baseline}_year_${year}/mercury_impulse_${impulse}.out
#SBATCH --error=./job-outs/${action_name}/pattern_${pattern}_title_${title}_ylabel_${ylabel}_cearth_${cearth}_baseline_${baseline}_year_${year}/mercury_impulse_${impulse}.err


#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=3
#SBATCH --mem=16G
#SBATCH --time=7-00:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0

echo "\$SLURM_JOB_NAME"

echo "Program starts \$(date)"

python3 /home/bcheng4/QuantClimateChange/$python_name --pattern ${pattern} --title ${title} --ylabel ${ylabel} --cearth ${cearth} --impulse ${impulse} --baseline ${baseline} --year ${year}

echo "Program ends \$(date)"

EOF
                        
                        sbatch ./bash/${action_name}/pattern_${pattern}_title_${title}_ylabel_${ylabel}_cearth_${cearth}_baseline_${baseline}_year_${year}_impulse_${impulse}.sh
                        count=$(($count+1))
                    done
                done
            done
        done
    done
done


action_name="PureIRF"
server_name="graph"
python_name="Model_Erik_PureIRF_graphsize.py"


declare -a titles=('Impulse Response of Temperature Anomaly per Ttc' 'Impulse Response of Carbon Concentration Dynamics per Ttc')
declare -a ylabels=('Temperature (K)' 'Carbon (ppm)' 'Emission (Gtc)' 'Degree (Celsius)')
ceartharray=(0.3725 0.3916 15)
impulsearray=($(seq 0 100 ))
declare -a baselinearray = ("carbonvoid.csv" "rcp85co2eqv3.csv" "rcp60co2eqv3.csv" "rcp45co2eqv3.csv" "rcp30co2eqv3.csv" "rcp00co2eqv3.csv")
yeararray = (1801 1900 2000 2010 2020 2030)
patternarray=(0 1 2)
count=0
pattern=2

for title in ${titles[@]}
do
    for ylabel in ${ylabels[@]}
    do
        for cearth in ${ceartharray[@]}
        do
            for impulse in ${impulsearray[@]}
            do
                for baseline in ${baselinearray[@]}
                do
                    for year in ${yeararray[@]}
                    do
                        
                        mkdir -p ./job-outs/${action_name}/pattern_${pattern}_title_${title}_ylabel_${ylabel}_cearth_${cearth}_baseline_${baseline}_year_${year}/
                        
                        if [ -f ./bash/${action_name}/pattern_${pattern}_title_${title}_ylabel_${ylabel}_cearth_${cearth}_baseline_${baseline}_year_${year}_impulse_${impulse}.sh ]
                        then
                            rm ./bash/${action_name}/pattern_${pattern}_title_${title}_ylabel_${ylabel}_cearth_${cearth}_baseline_${baseline}_year_${year}_impulse_${impulse}.sh
                        fi
                        
                        mkdir -p ./bash/${action_name}/
                        
                        touch ./bash/${action_name}/pattern_${pattern}_title_${title}_ylabel_${ylabel}_cearth_${cearth}_baseline_${baseline}_year_${year}_impulse_${impulse}.sh
                        
                        tee -a ./bash/${action_name}/pattern_${pattern}_title_${title}_ylabel_${ylabel}_cearth_${cearth}_baseline_${baseline}_year_${year}_impulse_${impulse}.sh << EOF
#! /bin/bash


######## login
#SBATCH --job-name=impulse${actiontime}_$count
#SBATCH --output=./job-outs/${action_name}/pattern_${pattern}_title_${title}_ylabel_${ylabel}_cearth_${cearth}_baseline_${baseline}_year_${year}/mercury_impulse_${impulse}.out
#SBATCH --error=./job-outs/${action_name}/pattern_${pattern}_title_${title}_ylabel_${ylabel}_cearth_${cearth}_baseline_${baseline}_year_${year}/mercury_impulse_${impulse}.err


#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=3
#SBATCH --mem=16G
#SBATCH --time=7-00:00:00

####### load modules
module load python/booth/3.8/3.8.5  gcc/9.2.0

echo "\$SLURM_JOB_NAME"

echo "Program starts \$(date)"

python3 /home/bcheng4/QuantClimateChange/$python_name --pattern ${pattern} --title ${title} --ylabel ${ylabel} --cearth ${cearth} --impulse ${impulse} --baseline ${baseline} --year ${year}

echo "Program ends \$(date)"

EOF
                        
                        sbatch ./bash/${action_name}/pattern_${pattern}_title_${title}_ylabel_${ylabel}_cearth_${cearth}_baseline_${baseline}_year_${year}_impulse_${impulse}.sh
                        count=$(($count+1))
                    done
                done
            done
        done
    done
done
