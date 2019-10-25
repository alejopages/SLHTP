#!/bin/bash

if [[ $# -ne 2 ]];
then
	echo Transfers tifs from cyverse dir to ./
	exit
fi;

analyses_dir=$1
local=$2
basepath=""

ils -r $analyses_dir | while read line;
do
    # Store basepath for joining with filename later
    if [[ "${line:0:2}" = "C-" ]];
    then
        basepath=${line#"C-"}
    fi

    if [[ "${line: -4}" = ".tif" ]];
    then
        iget -v $basepath/$line $local
    fi
done
