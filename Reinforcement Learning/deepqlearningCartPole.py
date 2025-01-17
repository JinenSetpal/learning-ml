import gym
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import namedtuple, deque


# noinspection PyShadowingNames
class DQNA:
    def __init__(self, env, discount_factor=0.95,
                 epsilon_greedy=1.0, epsilon_min=0.01,
                 epsilon_decay=0.995, learning_rate=1e-3, max_memory_size=2000):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_greedy
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.action_size = env.action_space.n
        self.memory = deque(maxlen=max_memory_size)
        self.state_size = env.observation_space.shape[0]
        self._build_nn_model()

    def _build_nn_model(self, n_layers=3):
        self.model = tf.keras.Sequential()
        for n in range(n_layers - 1):
            self.model.add(tf.keras.layers.Dense(units=32, activation=tf.keras.activations.relu))
            self.model.add(tf.keras.layers.Dense(units=32, activation=tf.keras.activations.relu))
        self.model.add(tf.keras.layers.Dense(units=self.action_size))

        self.model.build(input_shape=(None, self.state_size))
        self.model.summary()
        self.model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.lr))

    def remember(self, transition):
        self.memory.append(transition)

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state)[0]
        return np.argmax(q_values)

    def _learn(self, batch_samples):
        batch_states, batch_targets = [], []
        for transition in batch_samples:
            s, a, r, next_s, done = transition
            target = r if done else r + self.gamma * np.max(self.model.predict(next_s)[0])
            target_all = self.model.predict(next_s)[0]
            target_all[a] = target
            batch_states.append(s.flatten())
            batch_targets.append(target_all)
            self._adjust_epsilon()

        return self.model.fit(x=np.array(batch_states),
                              y=np.array(batch_targets),
                              epochs=1,
                              verbose=0)

    def _adjust_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def replay(self, batch_size):
        samples = random.sample(self.memory, batch_size)
        history = self._learn(samples)
        return history.history['loss'][0]


# noinspection PyShadowingNames
def plot_learning_history(history):
    fig = plt.figure(1, figsize=(14, 5))
    ax = fig.add_subplot(2, 1, 1)
    episodes = np.arange(len(history[0])) + 1
    plt.plot(episodes, history[0], lw=4, marker='o', markersize=10)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.xlabel('Episodes', size=20)
    plt.ylabel('# Total Rewards', size=20)
    plt.show()


np.random.seed(1)
tf.random.set_seed(1)

transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
episodes = 200
batch_size = 32
init_replay_memory_size = 500

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    agent = DQNA(env)
    state = env.reset()
    state = np.reshape(state, [1, agent.state_size])

    for i in range(init_replay_memory_size):
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        # noinspection PyRedeclaration
        next_state = np.reshape(state, [1, agent.state_size])
        agent.remember(transition(state, action, reward, next_state, done))
        if done:
            state = env.reset()
            state = np.reshape(state, [1, agent.state_size])
        else:
            state = next_state

    total_rewards, losses = [], []
    for e in range(episodes):
        # noinspection PyRedeclaration
        state = env.reset()
        if e % 10 == 0:
            env.render()
        state = np.reshape(state, [1, agent.state_size])
        for i in range(500):
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            # noinspection PyRedeclaration
            next_state = np.reshape(state, [1, agent.state_size])
            agent.remember(transition(state, action, reward, next_state, done))
            state = next_state
            if e % 10 == 0:
                env.render()
            if done:
                total_rewards.append(i)
                print('Episode %d/%d, Total reward: %d' % (e, episodes, i))
                break
            loss = agent.replay(batch_size)
            losses.append(loss)

    plot_learning_history(total_rewards)
