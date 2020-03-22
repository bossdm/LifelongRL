from deep_q_rl.run_nature import AgentNetworkParams

defaultParams = AgentNetworkParams.INPUT


class DQ_RL(object):

    def __init__(self):
        if parameters.nn_file is None:
            network = q_network.DeepQLearner(defaults.RESIZED_WIDTH,
                                             defaults.RESIZED_HEIGHT,
                                             num_actions,
                                             parameters.phi_length,
                                             parameters.discount,
                                             parameters.learning_rate,
                                             parameters.rms_decay,
                                             parameters.rms_epsilon,
                                             parameters.momentum,
                                             parameters.clip_delta,
                                             parameters.freeze_interval,
                                             parameters.batch_size,
                                             parameters.network_type,
                                             parameters.update_rule,
                                             parameters.batch_accumulator,
                                             rng)
        else:
            handle = open(parameters.nn_file, 'r')
            network = cPickle.load(handle)

        agent = ale_agent.NeuralAgent(network,
                                      parameters.epsilon_start,
                                      parameters.epsilon_min,
                                      parameters.epsilon_decay,
                                      parameters.replay_memory_size,
                                      parameters.experiment_prefix,
                                      parameters.replay_start_size,
                                      parameters.update_frequency,
                                      rng)
