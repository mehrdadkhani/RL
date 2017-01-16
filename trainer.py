from rlcommon import *
from ddpg import *
from replay_buffer import ReplayBuffer


def main():
    np.set_printoptions(threshold=np.inf)
    # Initialize Redis for IPC
    redis_server = redis.StrictRedis(host='localhost', port=6379, db=0)
    redis_pubsub = redis_server.pubsub()
    redis_cmd = redis_server.pubsub()
    redis_pubsub.subscribe('Experience')
    redis_cmd.subscribe('cmd')

    # Initialize Monitor
    # mon = Popen('python3 parameter_monitor.py', stdin=PIPE, shell=True)

    # Initialize Replay Buffer
    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)

    # Initialize Tensorflow (Actor and Critic)

    tf_sess = tf.Session()

    actor = ActorNetwork(tf_sess, state_dim, action_dim, action_bound, ACTOR_LEARNING_RATE, TAU)

    critic = CriticNetwork(tf_sess, state_dim, action_dim, CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars())

    summary_ops, summary_vars = build_summaries()

    tf_sess.run(tf.initialize_all_variables())
    writer = tf.train.SummaryWriter(SUMMARY_DIR, tf_sess.graph)

    # save neural net parameters
    saver = tf.train.Saver()

    nn_model = None
    if nn_model is not None:  # nn_model is the path to file
        saver.restore(sess, nn_model)
        print("Model restored.")

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize Variables
    td_loss_sum = 0
    td_loss = 0
    iteration_counter = 0

    avg_time_interval = MinimumTrainerUpdateInterval

    last_dump_ts = time()
    time_stamp_rec = time()

    cmd = "echo start > super_ep.txt"
    Popen(cmd, shell=True).wait()

    ep = 0

    #IsTraining = False
    nomore_flag = True
    exploration_flag = False
    countt = 0
    IsTraining = True

    # saver.restore(tf_sess, '2017')
    # main loop of actor
    while True:
        # Start time
        start_time = time()

        # Update Replay Buffer
        HasNewExperience = False
        dumpstr = ''
        while True:
            msg = redis_pubsub.get_message()
            if msg:
                if msg['type'] == 'message':
                    HasNewExperience = True
                    new_experience = pickle.loads(msg['data'])
                    # Add new experience to replay buffer
                    replay_buffer.add(
                        np.reshape(new_experience['state'], (state_dim,)),
                        np.reshape(new_experience['action'], (action_dim,)),
                        new_experience['reward'],
                        False,
                        np.reshape(new_experience['next_state'], (state_dim,)))

                    dumpstr = "%(td_loss)7.2f %(qlen)f %(dp)f %(throughput)f " % {"td_loss": td_loss, "qlen": new_experience['queue_delay'], "dp": new_experience['dropprob'], "throughput": new_experience['throughput']}
                    cmd = "echo '%s' >> log.txt" % dumpstr
                    Popen(cmd, shell=True).wait()


                    #mon.stdin.write('%f,%f,%f,%f\n' % (new_experience['queue_delay'], new_experience['dropprob'], new_experience['throughput'],new_experience['power']))
            else:
                break

        if HasNewExperience == False:
            sleep(0.01)  # 10ms
            continue

        # Update Critic and Actor
        if replay_buffer.size() > MINIBATCH_SIZE and IsTraining:
            iteration_counter += 1
            s_batch, a_batch, r_batch, t_batch, s2_batch = \
                replay_buffer.sample_batch(MINIBATCH_SIZE)
            #TEST
            # s_batch = s_batch - 0.3
            # s2_batch -= 0.3


            #print "begin: ", a_batch
            # a_outs = actor.predict(s_batch)
            # grads = critic.action_gradients(s_batch, a_outs)

            #cmd = "echo '%s' >> critic_grad.txt" % grads
            #Popen(cmd, shell=True).wait()

            # cmd = "echo '%s' >> s_batch.txt" % str(s_batch)
            # Popen(cmd, shell=True).wait()
            # actor_grads = actor.output_gradient(s_batch, grads[0])

            # layer1_norm = np.linalg.norm(actor_grads[0])
            # layer2_norm = np.linalg.norm(actor_grads[2])
            # layer3_norm = np.linalg.norm(actor_grads[4])
            # layer1b_norm = np.linalg.norm(actor_grads[1])
            # layer2b_norm = np.linalg.norm(actor_grads[3])
            # layer3b_norm = np.linalg.norm(actor_grads[5])
            # #print(layer1_norm+layer2_norm+layer3_norm)
            # actor_w = actor.output_weights()
            # w1 = np.linalg.norm(actor_w[0])
            # w1b = np.linalg.norm(actor_w[1])
            # w2 = np.linalg.norm(actor_w[2])
            # w2b = np.linalg.norm(actor_w[3])
            # w3 = np.linalg.norm(actor_w[4])
            # w3b = np.asarray(actor_w[5])
            # gradstr = "%(norm_1)f %(norm_2)f %(norm_3)f %(normb_1)f %(normb_2)f %(normb_3)f %(w1)f %(w1b)f %(w2)f %(w2b)f %(w3)f %(w3b)f" % {
            #    "norm_1": layer1_norm, "norm_2": layer2_norm, "norm_3": layer3_norm, "normb_1": layer1b_norm,
            #     "normb_2": layer2b_norm, "normb_3": layer3b_norm, "w1": w1, "w1b": w1b, "w2": w2, "w2b": w2b,
            #     "w3": w3, "w3b": w3b}
            #cmd = "echo '%s' >> actor_grad.txt" % np.average(actor_grads[0])
            #Popen(cmd, shell=True).wait()
            #actor.train(s_batch, grads[0])

            # calculate targets
            target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))

            y_i = []
            for k in xrange(MINIBATCH_SIZE):
                if t_batch[k]:
                    y_i.append(r_batch[k])
                else:
                    y_i.append(r_batch[k] + GAMMA * target_q[k])

            td_loss, _ = critic.train(s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))
            td_loss_sum += td_loss

            a_outs = actor.predict(s_batch)
            #a_outs = (a_outs - np.min(a_outs))/(np.max(a_outs)-np.min(a_outs))*action_bound
            grads = critic.action_gradients(s_batch, a_outs)
            # print a_outs
            #if nomore_flag == False:
            # if (np.average(new_experience['action'])) < 0 and nomore_flag == False:
            #     grads[0] = 10000.0 * np.random.uniform(0,1,size=np.shape(grads[0]))
            #     print "reverse1"
            #
            # if (np.average(new_experience['action'])) > action_bound and nomore_flag == False:
            #     grads[0] = -10000.0 * np.random.uniform(0, 1, size=np.shape(grads[0]))
            #     print "reverse2"
            # if exploration_flag == True:
            #     grads[0] = 10000.0 * np.random.uniform(-1,1,size=np.shape(grads[0]))
            #
            # if (np.average(new_experience['action'])) > (action_bound - 0.1):
            #     grads[0] = - 100.0 * np.absolute(grads[0])
            #     print "reverse2"
            #   #      countt += 1
            #         #grads[0] = np.random.uniform(0,1,size=np.shape(grads[0]))
            #     print "reversed1"
            # if (np.average(new_experience['action'])) > (action_bound - 0.1):
            #     grads[0] = -100.0 * np.absolute(grads[0])
            #     ##  countt += 1
            #     #grads[0] = - np.random.uniform(0,1,size=np.shape(grads[0]))
            #     print "reversed2"
            #if countt > 100:
             #   nomore_flag = True
              #  time_stamp_rec = time()
            #if time() - time_stamp_rec > 200 and nomore_flag == False:
            #    countt =0
            #    nomore_flag = False

            actor.train(s_batch, grads[0])

            actor.update_target_network()
            critic.update_target_network()

        # Send new actor neural network to executor
        # TODO

        actor_nn_params = tf_sess.run(actor.network_params)
        redis_server.publish('Actor', pickle.dumps(actor_nn_params))

        if time() - last_dump_ts > dump_time_interval:
            last_dump_ts = time()
            if iteration_counter == 0:
                iteration_counter = -1
            print(
            'TRAINER ', ep, ' : Iteration Counter: ', iteration_counter, ' TdLoss: ', td_loss_sum / iteration_counter,'rewards: ', new_experience['reward'])#, 'qbatch: ', np.average(s_batch[:,0::3]),'pbatch: ', np.average(s_batch[:,1::3]),'thbatch: ', np.average(s_batch[:,2::3]))

            #me:
            # actor_weights = actor.get_w()
            # print actor_weights
            # stract = ""
            # for i in xrange(len(actor_weights[3])):
            #     stract+= str(actor_weights[3][i]) + " "
            # cmd = "echo '%s' >> actor_w.txt" % stract
            # Popen(cmd, shell=True).wait()

            ep += 1
            iteration_counter = 0
            td_loss_sum = 0

        # Handle Control Command

        while True:
            msg = redis_cmd.get_message()
            if msg:
                if msg['type'] == 'message':
                    cmd = pickle.loads(msg['data'])
                    if cmd['cmd'] == 'load':
                        print('Load Model: ', cmd['name'])
                        saver.restore(tf_sess, cmd['name'])
                        pass

                    if cmd['cmd'] == 'store':
                        print('Store Model: ', cmd['name'])
                        saver.save(tf_sess, cmd['name'])
                        pass

                    if cmd['cmd'] == 'stop_training':
                        print('Stop Training')
                        IsTraining = False
                        pass

                    if cmd['cmd'] == 'resume_training':
                        print('Resume Training')
                        IsTraining = True
                        pass
                    if cmd['cmd'] == 'nomore':
                        print('NOMORE')
                        nomore_flag = True
                        pass

                    if cmd['cmd'] == 'more':
                        print('MORE')
                        nomore_flag = False
                        pass
                    if cmd['cmd'] == 'exp':
                        print('EXP:')
                        exploration_flag = not exploration_flag
                        print exploration_flag
                        pass
                    

            else:
                break

        # Time Management
        time_passed = time() - start_time
        if time_passed < MinimumTrainerUpdateInterval:
            sleep(MinimumTrainerUpdateInterval - time_passed)

        time_passed = time() - start_time
        avg_time_interval = avg_time_interval * 0.75 + time_passed * 0.25


if __name__ == '__main__':
    main()








