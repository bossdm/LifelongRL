for i in range(num_episodes):

    # The Q-Network
    while j < max_epLength:
        j += 1
        # Choose an action by greedily (with e chance of random action) from the Q-network
        if np.random.rand(1) < e or total_steps < pre_train_steps:
            state1 = sess.run(mainQN.rnn_state, \
                              feed_dict={mainQN.scalarInput: [s / 255.0], mainQN.trainLength: 1, mainQN.state_in: state,
                                         mainQN.batch_size: 1})
            a = np.random.randint(0, 4)
        else:
            a, state1 = sess.run([mainQN.predict, mainQN.rnn_state], \
                                 feed_dict={mainQN.scalarInput: [s / 255.0], mainQN.trainLength: 1,
                                            mainQN.state_in: state, mainQN.batch_size: 1})
            a = a[0]
        s1P, r, d = env.step(a)
        s1 = processState(s1P)
        total_steps += 1
        episodeBuffer.append(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))
        if total_steps > pre_train_steps:
            if e > endE:
                e -= stepDrop

            if total_steps % (update_freq * 1000) == 0:
                print "Target network updated."
                updateTarget(targetOps, sess)

            if total_steps % (update_freq) == 0:
                # Reset the recurrent layer's hidden state
                state_train = (np.zeros([batch_size, h_size]), np.zeros([batch_size, h_size]))

                trainBatch = myBuffer.sample(batch_size, trace_length)  # Get a random batch of experiences.
                # Below we perform the Double-DQN update to the target Q-values
                Q1 = sess.run(mainQN.predict, feed_dict={ \
                    mainQN.scalarInput: np.vstack(trainBatch[:, 3] / 255.0), \
                    mainQN.trainLength: trace_length, mainQN.state_in: state_train, mainQN.batch_size: batch_size})
                Q2 = sess.run(targetQN.Qout, feed_dict={ \
                    targetQN.scalarInput: np.vstack(trainBatch[:, 3] / 255.0), \
                    targetQN.trainLength: trace_length, targetQN.state_in: state_train,
                    targetQN.batch_size: batch_size})
                end_multiplier = -(trainBatch[:, 4] - 1)
                doubleQ = Q2[range(batch_size * trace_length), Q1]
                targetQ = trainBatch[:, 2] + (y * doubleQ * end_multiplier)
                # Update the network with our target values.
                sess.run(mainQN.updateModel, \
                         feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 0] / 255.0), mainQN.targetQ: targetQ, \
                                    mainQN.actions: trainBatch[:, 1], mainQN.trainLength: trace_length, \
                                    mainQN.state_in: state_train, mainQN.batch_size: batch_size})
        rAll += r
        s = s1
        sP = s1P
        state = state1
        if d == True:
            break