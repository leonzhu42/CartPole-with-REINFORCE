import random
import tensorflow as tf
import numpy as np
import gym

policy_network_verbose = False
update_gradients_verbose = False

policy_learning_rate = 0.0001
baseline_learning_rate = 0.0001

n_episode = 1000
gamma = 0.99

observation_placeholder = tf.placeholder(tf.float32, shape=[None, 4])
W1_policy = tf.get_variable('W1_policy', shape=[4, 64])
b1_policy = tf.get_variable('b1_policy', shape=[64])
W2_policy = tf.get_variable('W2_policy', shape=[64, 64])
b2_policy = tf.get_variable('b2_policy', shape=[64])
W3_policy = tf.get_variable('W3_policy', shape=[64, 2])
b3_policy = tf.get_variable('b3_policy', shape=[2])
hidden1_policy = tf.nn.relu(tf.matmul(observation_placeholder, W1_policy) + b1_policy)
hidden2_policy = tf.nn.relu(tf.matmul(hidden1_policy, W2_policy) + b2_policy)
scores = tf.nn.relu(tf.matmul(hidden2_policy, W3_policy) + b3_policy)
probs = tf.nn.softmax(scores)

discount_placeholder = tf.placeholder(tf.float32)
gain_placeholder = tf.placeholder(tf.float32)
action_placeholder = tf.placeholder(tf.float32, shape=[None, 2])
log_probs = tf.log(probs) * action_placeholder
grads_policy = tf.gradients(log_probs, [W1_policy, b1_policy, W2_policy, b2_policy, W3_policy, b3_policy])
update_W1_policy = W1_policy.assign_add(policy_learning_rate * discount_placeholder * gain_placeholder * grads_policy[0])
update_b1_policy = b1_policy.assign_add(policy_learning_rate * discount_placeholder * gain_placeholder * grads_policy[1])
update_W2_policy = W2_policy.assign_add(policy_learning_rate * discount_placeholder * gain_placeholder * grads_policy[2])
update_b2_policy = b2_policy.assign_add(policy_learning_rate * discount_placeholder * gain_placeholder * grads_policy[3])
update_W3_policy = W3_policy.assign_add(policy_learning_rate * discount_placeholder * gain_placeholder * grads_policy[4])
update_b3_policy = b3_policy.assign_add(policy_learning_rate * discount_placeholder * gain_placeholder * grads_policy[5])

W1_baseline = tf.get_variable('W1_baseline', shape=[4, 64])
b1_baseline = tf.get_variable('b1_baseline', shape=[64])
W2_baseline = tf.get_variable('W2_baseline', shape=[64, 64])
b2_baseline = tf.get_variable('b2_baseline', shape=[64])
W3_baseline = tf.get_variable('W3_baseline', shape=[64, 1])
b3_baseline = tf.get_variable('b3_baseline', shape=[1])
hidden1_baseline = tf.nn.relu(tf.matmul(observation_placeholder, W1_baseline) + b1_baseline)
hidden2_baseline = tf.nn.relu(tf.matmul(hidden1_baseline, W2_baseline) + b2_baseline)
values = tf.matmul(hidden2_baseline, W3_baseline) + b3_baseline

grads_baseline = tf.gradients(values, [W1_baseline, b1_baseline, W2_baseline, b2_baseline, W3_baseline, b3_baseline])
update_W1_baseline = W1_baseline.assign_add(baseline_learning_rate * gain_placeholder * grads_baseline[0])
update_b1_baseline = b1_baseline.assign_add(baseline_learning_rate * gain_placeholder * grads_baseline[1])
update_W2_baseline = W2_baseline.assign_add(baseline_learning_rate * gain_placeholder * grads_baseline[2])
update_b2_baseline = b2_baseline.assign_add(baseline_learning_rate * gain_placeholder * grads_baseline[3])
update_W3_baseline = W3_baseline.assign_add(baseline_learning_rate * gain_placeholder * grads_baseline[4])
update_b3_baseline = b3_baseline.assign_add(baseline_learning_rate * gain_placeholder * grads_baseline[5])

def policy_network(observation, session, verbose=False):
    probabilities = session.run(probs, feed_dict={observation_placeholder: [observation]})
    if verbose:
        print(probabilities)
    if random.random() <= probabilities[0][0]:
        return 0
    else:
        return 1

def update_policy_gradients(discount, gain, observation, action, session, verbose=False):
    if action == 0:
        mask = [[1, 0]]
    else:
        mask = [[0, 1]]
    result = session.run([probs, log_probs, grads_policy, update_W1_policy, update_b1_policy, update_W2_policy, update_b2_policy, update_W3_policy, update_b3_policy], feed_dict={observation_placeholder: [observation], discount_placeholder: discount, gain_placeholder: gain, action_placeholder: mask})
    if verbose:
        print(result[2]) # print the gradients w.r.t. weights and biases

def update_baseline_gradients(gain, observation, session, verbose=False):
    result = session.run([grads_baseline, update_W1_baseline, update_b1_baseline, update_W2_baseline, update_b2_baseline, update_W3_baseline, update_b3_baseline], feed_dict={observation_placeholder: [observation], gain_placeholder: gain})
    if verbose:
        print(result[0])

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
            gain_with_baseline = gain - sess.run(values, feed_dict={observation_placeholder: [observation]})[0][0]
            update_policy_gradients(discount=discount, gain=gain_with_baseline, observation=observation, action=action, session=sess, verbose=update_gradients_verbose)
            update_baseline_gradients(gain=gain_with_baseline, observation=observation, session=sess, verbose=update_gradients_verbose)
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