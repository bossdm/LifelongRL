#!/bin/bash

export HOME=/home/db2c15


while getopts ":s:v:m:i:f:r:t:w:p:a:n:" opt; do
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
	echo NETWORK_FF=$NETWORK_FF
    ;;
    a) export ACTION="$OPTARG"
       echo ACTION=$ACTION
    ;;

    ?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done
  			
              		
			if [ "$METHOD" == "LSTMQ" ] ;
			then
				echo LSTM
				echo RUN $RUN
				qsub ${HOME}/POmazeCommandsSSA_NEAT.sh -l walltime=$WALLTIME,nodes=1 -V 
				sleep 0.05	
				
			else
				echo RUN $RUN
				export PROGRAM_CELLS=$p
				echo PROGRAM_CELLS=$PROGRAM_CELLS
				export WORKING_MEMORY=$w
				echo WORKING_MEMORY $WORKING_MEMORY
				qsub ${HOME}/POmazeCommandsSSA_NEAT.sh -l walltime=$WALLTIME,nodes=1 -V 
				sleep 0.05

	     		
				
			fi
			






