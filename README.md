# LifelongRL

Code related to the following works including and related to my PhD thesis on reinforcement learning in long-term environments:

    [1] Bossens, D.M. (2020). Reinforcement learning with limited prior knowledge in long-term environments, PhD thesis (to be submitted after minor corrections).
    [2] Bossens, D. M., Townsend, N. C., & Sobey, A. J. (2019). Learning to learn with active adaptive perception. Neural Networks, 115, 30–49. https://doi.org/10.1016/J.NEUNET.2019.03.006
    [3] Bossens, D. M. & Sobey, A. J. (2020). Efficiently utilising a limited number of policies to solve a large number of tasks. Neural Networks (under review).

The original code was written in python 2.7 but has since been modified to work on python 3.7. Integer division and tuples inside function arguments caused some problems but they should be resolved now; using python 2.7 is also an option. Also, some parts of the code were removed either because they were experimental, not used, or private. If something is not working, please report the bug. The examples reported below are tested though.


-----------------------------------------------------------------------------------------

In the main directory, the there are two main scripts which I used:

# POmaze.py, 

This script runs the non-episodic partially observable maze setup used in [1,2].

To run this script, do:

    python POmaze.py [main options: -r ${run_number} -m ${method_type}]


For example, the command:

    python POmaze.py -r 0 -m SSA_gradientQsequence -v True

will run the SMP-DRQN method on maze 0 and puts visualisation on.

Note that the maze numbering is cyclic, such that -r 30 will run a different replicate of the same type of maze, and that maze 0-9 is Easy, 10-19 is Medium, and 20-29 is Difficult





# MultiExperiment.py

This script runs the lifelong learning experiments in [1,3]. This involves task sequences of up to 18 tasks.

To run this script, do:

    python MultiExperiment.py [main options: -r ${run_number} -m ${method_type} -P ${number_of_policies} -x ${experiment_type}]


For example, the command:

    python MultiExperiment.py -r 0 -m  TaskDrift_DRQN -P 4 -x lifelongx18t  -v True

will run learning with multiple policies, using 4 DRQN policies and adaptive policy selection, on the 0'th task sequence with 18 unique tasks.


Although currently, there are tasks located in the Tasks directory, you can also generate your own tasks using any of the experiment types in{"create_lifelong_tasks","create_lifelongx_tasks","create_lifelongx_tasks_uniform","create_lifelongx_test_tasks"}. For example,

    python MultiExperiment.py -x create_lifelongx_tasks 


This will generate files with pattern "lifelongx*tasks" and "lifelongx*feature_weights" in the main source directory. 

















