# LifelongRL

Code related to the following works including and related to my PhD thesis on reinforcement learning in long-term/lifetime environments:

    [1] Bossens, D. M. (2020). Reinforcement learning with limited prior knowledge in long-term environments. PhD Thesis (University of Southampton). Retrieved from http://eprints.soton.ac.uk/id/eprint/442596
    [2] Bossens, D. M., Townsend, N. C., & Sobey, A. J. (2019). Learning to learn with active adaptive perception. Neural Networks, 115, 30–49. https://doi.org/10.1016/J.NEUNET.2019.03.006
    [3] Bossens, D. M., & Sobey, A. J. (2021). Lifetime policy reuse and the importance of task capacity. ArXiv Preprint ArXiv:2106.01741, 1–27. Retrieved from http://arxiv.org/abs/2106.01741

This includes:
1. Active adaptive perception [1,2]: the "SSA_gradientQsequence" in POmaze.py (partially observable mazes)
2. Lifetime policy reuse [1,3]: the TaskDrift_* methods in MultiExperiment.py (18-task POcman POMDP sequence) and lifelong_cartpole_agent.py (27-task Cartpole MDP sequence)
3. Lifelong SSA [1]: "SingleSMP_DRQN_lifetimeTaskspecific" and "SingleSMP_DRQN_lifetimeTaskspecificRelative" in MultiExperiment.py (18-task POcman POMDP sequence)



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

This script runs the lifelong learning experiments in [1,3]. This involves task sequences of up to 18 POMDP tasks.

To run this script, do:

    python MultiExperiment.py [main options: -r ${run_number} -m ${method_type} -P ${number_of_policies} -x ${experiment_type}]


For example, the command:

    python MultiExperiment.py -r 0 -m  TaskDrift_DRQN -P 4 -x lifelongx18t  -v True

will run learning with multiple policies according to lifetime policy reuse, using 4 DRQN policies and adaptive policy selection, on the 0'th task sequence with 18 unique tasks. A variety of other methods can be found in POcmanUtils.py:

    -"Unadaptive" prefix (instead of TaskDrift) means lifetime policy reuse without adaptive policy selection
    -no prefix means a single-policy method
    -"1-to-1" means the task-specific learner, where each policy is specialised on a specific task
    -"Selective" means DRQN with selective experience replay (see Isele, D., & Cosgun, A. (2018). Selective experience replay for lifelong learning. 32nd AAAI  Conference on Artificial Intelligence, AAAI 2018, 3302–3309. Retrieved from http://arxiv.org/abs/1802.10269)
    -"EWC" is Elastic Weight Consolidation (Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., & Rusu, A. A. (2017). Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences of the United States of America, 114(13), 3521–3526. https://doi.org/10.1073/pnas.1611835114 ). The implementation could use some further verification, so it is not recommended to use it at the moment.


Although currently, there are tasks located in the Tasks directory, you can also generate your own tasks using any of the experiment types in{"create_lifelong_tasks","create_lifelongx_tasks","create_lifelongx_tasks_uniform","create_lifelongx_test_tasks"}. For example,

    python MultiExperiment.py -x create_lifelongx_tasks 


This will generate files with pattern "lifelongx*tasks" and "lifelongx*feature_weights" in the main source directory. 



# lifelong_cartpole_agent.py

this runs the 27-task sequence of MDPs found in reference [3]. the interface is comparable to MultiExperiment, except that you have to choose between

-x single (one task run for its full length, useful for tuning or obtaining an estimate of optimal performance on the task) and 
-x lifelong (the main scenario involving 27-task sequences with randomly ordered task-blocks).














