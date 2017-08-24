import tensorflow as tf
import numpy as np
import gym

policy_network_verbose = False
update_gradients_verbose = False

learning_rate = 0.0001

n_episode = 500
gamma = 0.95

observation_placeholder = tf.placeholder(tf.float32, shape=[None, 4])
W1 = tf.get_variable('W1', shape=[4, 64])
b1 = tf.get_variable('b1', shape=[64])
W2 = tf.get_variable('W2', shape=[64, 64])
b2 = tf.get_variable('b2', shape=[64])
W3 = tf.get_variable('W3', shape=[64, 2])
b3 = tf.get_variable('b3', shape=[2])
hidden1 = tf.nn.relu(tf.matmul(observation_placeholder, W1) + b1)
hidden2 = tf.nn.relu(tf.matmul(hidden1, W2) + b2)
scores = tf.nn.relu(tf.matmul(hidden2, W3) + b3)
probs = tf.nn.softmax(scores)

learning_rate_placeholder = tf.placeholder(tf.float32)
discount_placeholder = tf.placeholder(tf.float32)
gain_placeholder = tf.placeholder(tf.float32)
action_placeholder = tf.placeholder(tf.float32, shape=[None, 2])
log_probs = tf.log(probs) * action_placeholder
grads = tf.gradients(log_probs, [W1, b1, W2, b2, W3, b3])
update_W1 = W1.assign_add(learning_rate_placeholder * discount_placeholder * gain_placeholder * grads[0])
update_b1 = b1.assign_add(learning_rate_placeholder * discount_placeholder * gain_placeholder * grads[1])
update_W2 = W2.assign_add(learning_rate_placeholder * discount_placeholder * gain_placeholder * grads[2])
update_b2 = b2.assign_add(learning_rate_placeholder * discount_placeholder * gain_placeholder * grads[3])
update_W3 = W3.assign_add(learning_rate_placeholder * discount_placeholder * gain_placeholder * grads[4])
update_b3 = b3.assign_add(learning_rate_placeholder * discount_placeholder * gain_placeholder * grads[5])

def policy_network(observation, session, verbose=False):
    probabilities = session.run(probs, feed_dict={observation_placeholder: [observation]})
    if verbose:
        print(probabilities)
    return np.argmax(probabilities)

def update_gradients(learning_rate, discount, gain, observation, action, session, verbose=False):
    if action == 0:
        mask = [[1, 0]]
    else:
        mask = [[0, 1]]
    result = session.run([probs, log_probs, grads, update_W1, update_b1, update_W2, update_b2, update_W3, update_b3], feed_dict={observation_placeholder: [observation], learning_rate_placeholder: learning_rate, discount_placeholder: discount, gain_placeholder: gain, action_placeholder: mask})
    if verbose:
        print(result[2]) # print the gradients w.r.t. weights and biases

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    env = gym.make('CartPole-v1')

    # Train
    print('Training...')
    for episode in range(n_episode):
        observation = env.reset()
        reward = 0
        done = False
        total_reward = 0.0
        memory = []
        while not done:
            action = policy_network(observation, sess, verbose=policy_network_verbose)
            next_observation, next_reward, next_done, _ = env.step(action)
            if next_done:
                next_reward = -100
            memory.append((observation, action, next_reward))
            total_reward += next_reward
            observation, reward, done = next_observation, next_reward, next_done
        print('Episode {0} Reward: {1}'.format(episode, total_reward))
        discount = 1.0
        gain = 0
        for observation, action, reward in memory:
            gain += discount * reward
            discount *= gamma
        discount = 1.0
        for t in range(len(memory)):
            observation, action, reward = memory[t]
            update_gradients(learning_rate=learning_rate, discount=discount, gain=gain, observation=observation, action=action, session=sess, verbose=update_gradients_verbose)
            gain -= discount * reward
            discount *= gamma
    print('Training done!')
    print()

    # Test
    input('Press Enter to start testing!')
    print('Testing...')
    for episode in range(5):
        observation = env.reset()
        reward = 0
        done = False
        total_reward = 0.0
        while not done:
            env.render()
            action = policy_network(observation, sess)
            next_observation, next_reward, next_done, _ = env.step(action)
            total_reward += next_reward
            observation, reward, done = next_observation, next_reward, next_done
        print('Reward:', total_reward)

    env.close()