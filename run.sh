#!/bin/bash

########################################
############# CSCI 2951-O ##############
########################################
E_BADARGS=65
# if [ $# -ne 1 ]
# then
# 	echo "Usage: `basename $0` <input>"
# 	exit $E_BADARGS
# fi

input=$1
shift 1

# Pass remaining args to binary
filename=$(basename -- "$input")
filename="${filename%.*}"
# flamegraph -o flamegraphs/"$filename".svg -- ./target/release/project1 -f "$input" "$@"
./target/release/project1 -f "$input" "$@"
