#!/bin/bash

for number in `seq 0 30`;
do

    num=$(( 3917377 + $number ))
    jobid="$num"
    jobid+='.blue101'
    qdel $jobid
done
