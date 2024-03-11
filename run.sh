#!/bin/bash

########################################
############# CSCI 2951-O ##############
########################################
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    echo -e "Usage: $(basename "$0") <input> [<binary args...>]"
    echo -e "For flamegraphs, set the environment variable \`FLAMEGRAPH=1\`."
    exit 1
fi

input=$1
shift 1

# Pass remaining args to binary
filename=$(basename -- "$input")
filename="${filename%.*}"

if [[ -n $FLAMEGRAPH ]]; then
    echo -e "Running flamegraph on execution..."
    flamegraph -o flamegraphs/"$filename".svg -- ./target/release/sat-solver -f "$input" "$@"
else 
    ./target/release/sat-solver -f "$input" "$@"
fi
