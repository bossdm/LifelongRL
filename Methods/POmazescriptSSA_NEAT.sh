#!/bin/bash

while getopts ":s:v:m:i:f:r:t:w:p:" opt; do
  case $opt in
    s) export STOPTIME="$OPTARG"
	echo STOPTIME $STOPTIME
    ;;

    
    m) export METHOD="$OPTARG" 
	echo METHOD $METHOD
	
    ;;
    f) export FILENAME="$OPTARG" 
	echo FILENAME $FILENAME
    ;;
    t) export WALLTIME="$OPTARG" 
	echo WALLTIME $WALLTIME
    ;;
    w) export WORKING_MEMORY="$OPTARG"
	echo WORKING_MEMORY $WORKING_MEMORY
    ;;
    p) export PROGRAM_CELLS="$OPTARG"
	echo PROGRAM_CELLS $PROGRAM_CELLS
    ;;
     r) export RUN="$OPTARG" 
    echo RUN $RUN
    ;;
    i) export INTERNAL="$OPTARG"
	echo INTERNAL $INTERNAL
    ;;
    n) export NETWORK_FF="$OPTARG" 
	echo NETWORK_FF=$NETWORK_FF ;;

    
    ?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done
sleep 2

	

				for a in "random"; do
				for q in "False"; do
				for n in "False"; do
				for p in 50 100; 
				
				do
					for w in 50 100;
					do
					for r in `seq 1 15`;
					do
					export RUN=$r
					echo RUN $RUN
					export ACTION=$a 
					echo ACTION=$ACTION
					export PROBABILISTIC=$q
					echo PROBABILISTIC=$PROBABILISTIC
					export NETWORK_FF=$n 
					echo NETWORK_FF=$NETWORK_FF
					export PROGRAM_CELLS=$p
					echo PROGRAM_CELLS=$PROGRAM_CELLS
					export WORKING_MEMORY=$w
					echo WORKING_MEMORY $WORKING_MEMORY
					qsub POmazeCommandsSSA_NEAT.sh -l walltime=$WALLTIME,nodes=1 -V 
					sleep 0.05
				done 
				done
				done
				done
				done
				done
	     		
	






