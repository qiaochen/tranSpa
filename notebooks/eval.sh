#!/bin/bash
declare -a dataset_names=("starmap_AllenVISp" "osmFISH_AllenVISp" "merfish_moffit" "seqFISH_SingleCell")
declare -a method_names=("tangram" "stPlus" "spaGE" "transImpClsSpa" "transImp" "transImpCls" "transImpSpa")
for data_name in ${dataset_names[@]}
do
    for method_name in ${method_names[@]}
    do
        echo "$data_name, $method_name"
        python efficiency_benchmark.py --dataset_name $data_name --method_name $method_name 
    done
done