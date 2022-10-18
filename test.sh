#! /bin/bash

declare -A arr1=([0]=bar [1]=aaa [2]=aaaa)
declare -A arr2=([0]=baz [1]=bbb [2]=bbbb)
arrays=(arr1 arr2)


for idx in "${arrays[@]}"; do
    declare -n temp="$idx"
    echo "${temp[@]}"
done