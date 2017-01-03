#!/bin/bash
source paths.config
source vars.config
source bench.sh
source config.sh

####################################
# splash benches are too tiny
parsec_splash()
{
	#build
	#build_eraser
	#basictools
	bench=parsec
	run
	#bench=splash
 	#run
}

can()
{
	setup
	
	check=""
	experiment=datarace
	inputsize = default
	echo "NO INSTRUMENTATION EXPERIMENTS"	
	#WITHOUT PIN
    mode=nopin
    #parsec_splash
    
	#ONLY PIN
    #mode=onlypin
    #parsec_splash
    echo "ONLY INSTRUMENTATION EXPERIMENTS"
	#ONLY INSTRUMENTATION
    mode=onlyinst
    #parsec_splash

    #echo "ONLY EXPERIMENTS"
    #EVENTLIST + ERASER on CPU
    #mode=check
    #check=cpu
    #parsec_splash
    
    echo "ONLY EVENTLIST EXPERIMENTS"
    mode=onlyevent
    #parsec_splash
    
    echo "GOLDILOCKS ON GPU EXPERIMENTS"
    #CHECK MODE GPU - GOLDI
    mode=check
    atomicity=no
    check=gpu
    memory=1
    parsec_splash

    echo "INLINED ERASER EXPERIMENTS"
    #Inlined Eraser
    #caution: ram overflow, possible memory leakage (from Pin itself?)
    mode=check
    check=eraser
    #parsec_splash
    
    #check mode
    mode=check
    check=djitplus
    #parsec_splash
    
    echo "INLINED FASTTRACK EXPERIMENTS"
    #check mode
    mode=check
    check=fasttrack
    #parsec_splash
}

####################################

run()
{
	createConfigFile
	
	if [ -z "$check" ] || [ "$check" = "gpu" ] || [ "$check" = "cpu" ]
	then
		build
	elif [ "$check" = "eraser" ]
	then
		build_eraser
	elif [ "$check" = "djitplus" ]
	then
		build_djitplus
	elif [ "$check" = "fasttrack" ]
	then
		build_fasttrack
	fi
    
    prefix=$experiment"_"$bench"_"
    
    if [ "$bench" = "parsec" ]
    then
	#inputs
	runForInput $inputsize
    fi
	
    if [ "$bench" = "splash" ]
    then
	#inputs
	runForInput default
    fi	
}

execute()
{
	for ((i=0; i < EXP_LOOP; i++)) 
	do
		resultfile=$experiments/$prefix$app$input$suffix"_run"$i".txt"
		
		if [ -z $stdin_file ]
		then
			if [ $mode = "nopin" ]
			then
			(time $run_exec $run_args) 2>&1 | tee $resultfile
			else
			(time $pin_exec $pin_args -t $pin_tool $tool_args -- $run_exec $run_args) 2>&1 | tee $resultfile 
			fi
		else
			if [ $mode = "nopin" ]
			then
			(time $run_exec $run_args < $stdin_file) 2>&1 | tee $resultfile
			else
			(time $pin_exec $pin_args -t $pin_tool $tool_args -- $run_exec $run_args < $stdin_file) 2>&1 | tee $resultfile 
			fi
		fi	
	done
}

#####################################
