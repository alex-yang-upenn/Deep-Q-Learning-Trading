from keras.models import load_model
from keras.models import Sequential
from keras.layers import Input, Dense, Activation
from keras.optimizers import Adam

import numpy as np
try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm

from collections import deque
import heapq
import math
import random

from Plotting import formatPrice, PlotBehavior


class Agent:
    def __init__(self, batchsize, X_train=None, X_eval=None, is_eval=False, model_name="", **kwargs):
        """
        Sets parameters for a Deep Q-Learning model or loads an existing one from a file

        Args:
            X_train (float array): Training dataset. This should represent sequential daily prices.
            X_eval (float array): Evaluation dataset. This should represent sequential daily prices.
            batchsize (int): Number of samples to use for experience replay

        """
        self.X_train = X_train
        self.X_eval = X_eval
        self.batchsize = batchsize

        # Dimension of the input state. Number of previous days used for prediction.
        self.state_size = kwargs.get("state_size", 1)
        # Dimension of the input state. Number of previous days used for prediction.
        self.action_size = kwargs.get("action_size", 3)
        
        self.gamma = kwargs.get("gamma", 0.95)
        self.epsilon = kwargs.get("epsilon", 1.0)
        self.epsilon_min = kwargs.get("epsilon_min", 0.05)
        self.epsilon_decay = kwargs.get("epsilon_decay", 0.999)

        self.memory = deque(maxlen=len(X_train) if X_train is not None else 2000)
        self.inventoryHeap = []

        self.is_eval = is_eval
        self.model_name = model_name
        self.model = load_model(model_name) if is_eval else self._model()

    def set_eval(self, is_eval):
        self.is_eval = is_eval

    def set_training_dataset(self, X_train):
        self.X_train = X_train

    def set_eval_dataset(self, X_eval):
        self.X_eval = X_eval

    def _model(self):
        """
        Defines architecture and and compile a new Deep Q-Learning model.
        - Input layer: 32 units, ReLU activation
        - Hidden layer 1: 16 units, ReLU activation
        - Hidden layer 2: 8 units, ReLU activation
        - Output layer: Dimension equal to action_size, Linear activation
        Uses Mean Squared Error (MSE) loss function and Adam optimizer with a learning rate of 5e-4.

        Returns:
            keras.models.Sequential: Compiled Neural Network
        """
        model = Sequential()
        model.add(Input(shape=(self.state_size,)))
        model.add(Dense(units=32, name="layer1"))
        model.add(Activation("relu"))
        model.add(Dense(units=16, name="layer2"))
        model.add(Activation("relu"))
        model.add(Dense(units=8, name="layer3"))
        model.add(Activation("relu"))
        model.add(Dense(units=self.action_size, name="output"))
        model.add(Activation("linear"))

        model.compile(
            loss="mse",
            optimizer=Adam(learning_rate=5e-4)
        )

        return model
    
    def getState(self, end_index):
        """
        Get the price difference between adjacent days within a set window (length = self.state_size), ending at index t, 
        from a sequential list of values. The differences are sigmoid normalized.

        If t < self.state_size, the window is left-padded with the initial value.

        Args:
            end_index (int): The ending index for the state window.

        Returns:
            np.array: A 2D numpy array of shape (1, self.state_size) containing the normalized differences
                      for the last n - 1 days.
        """
        if self.is_eval:
            dataset = self.X_eval
        else:
            dataset = self.X_train

        start_index = end_index - self.state_size
        block = dataset[start_index:end_index + 1] if start_index >= 0 else -start_index * [dataset[0]] + dataset[0:end_index + 1] 
        res = []
        for i in range(self.state_size):
            res.append(Sigmoid(block[i + 1] - block[i]))
        return np.array([res])

    def act(self, state):
        """
        Determines the agent's action based on the current state.

        Uses an epsilon-greedy policy:
        - Probability (1 - ε): Choose the action with the highest Q-value.
        - Probability (ε): Choose a random action.

        During training, ε typically starts high and decreases over time,
        promoting exploration initially and exploitation later.

        During evaluation, a lower ε (e.g. 0.05) is used to reduce randomness
        while still allowing for some exploration, which helps prevent overfitting.

        Args:
            state (float array): The window of price differences used to determine the action.

        Returns:
            int: The chosen action index.
        """
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        options = self.model.predict(state, verbose=0)
        return np.argmax(options[0])

    def expReplay(self):
        """
        Perform experience replay to train the Deep Q-Network.

        This method implements the core learning algorithm for Deep Q-Learning:
        1. Sample a mini-batch of experiences from memory.
        2. Compute target Q-values for each sample using the Bellman equation.
            - For non-terminal states, the target Q-value is: Q(s,a) = r + γ * max(Q(s',a')) where r is the reward, γ is the discount 
            factor, and s' is the next state.
            - For terminal states, the target Q-value is simply the reward.
        3. Update the Q-network to better approximate these target values.
            - The Q-network is updated using the mean squared error between its current predictions and the computed target Q-values.
        4. Decay the exploration rate (ε).
            - ε (exploration rate) is decayed after each replay to gradually reduce random actions as the agent learns.
        
        Side effects:
        - Updates the weights of self.model (the Q-network).
        - Reduces self.epsilon if it's above self.epsilon_min.

        Note:
        Has no effect if self.memory is not sufficiently filled
        """
        l = len(self.memory)

        if l < self.batchsize:
            return

        mini_batch = random.sample(self.memory, self.batchsize)
        
        states = np.array([experience[0][0] for experience in mini_batch])
        actions = np.array([experience[1] for experience in mini_batch])
        rewards = np.array([experience[2] for experience in mini_batch])
        next_states = np.array([experience[3][0] for experience in mini_batch])
        dones = np.array([experience[4] for experience in mini_batch]).astype(int)

        # Compute target Q-values
        target_qs = rewards + self.gamma * np.max(self.model.predict(next_states, verbose=0), axis=1) * (1 - dones)
        
        # Get current Q-values and update with target Q-values
        current_qs = self.model.predict(states, verbose=0)
        current_qs[np.arange(self.batchsize), actions] = target_qs

        # Train the model
        self.model.fit(states, current_qs, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    

    def train(self, num_episodes, replay_steps, save_steps, logging_steps, checkpoints_folder, model_name, plot=True):
        """
        Train the deep Q-learning agent on the provided dataset.

        Iterates through the training dataset for the set number of episodes. 
        and timesteps, performing actions, calculating rewards, and updating the Q-network.

        Args:
            num_episodes (int): The number of episodes to train for.
            replay_steps (int): The number of steps between each experience replay.
            save_steps (int): The model saves a checkpoint every save_steps episodes.
            logging_steps (int): The model logs results to termianl every logging_steps episodes.
            checkpoints_folder (str): The folder path to save model checkpoints.
            model_name (str): The name to use when saving the final trained model.
            plot (bool, optional): Whether to plot the agent's behavior after each episode. Defaults to True.

        Raises:
            Exception: If the training dataset (X_train) is not set.

        Returns:
            None
        """
        
        if self.is_eval:
            print("Warning: Agent is in evaluation mode. Setting to training mode")
            self.is_eval = False

        if not self.X_train:
            raise Exception("Training dataset not set")

        # Training Loop
        for e in range(1, num_episodes + 1):
            print("episode: " + str(e) + "/" + str(num_episodes))

            curr_state = self.getState(0)
            self.inventoryHeap.clear()

            total_profit = 0
            states_sell = []
            states_buy = []

            # Iterating through dataset
            for t in tqdm(range(len(self.X_train) - 1)):
                action = self.act(curr_state)
                next_state = self.getState(t + 1)
                reward = 0

                if action == 0:  # Agent recommended Sit
                    reward = 0
                elif action == 1:  # Agent recommended Buy
                    heapq.heappush(self.inventoryHeap, self.X_train[t])
                    reward = 0
                    states_buy.append(t)
                elif action == 2 and len(self.inventoryHeap) > 0:  # Agent recommended Sell
                    bought_price = heapq.heappop(self.inventoryHeap)
                    reward = self.X_train[t] - bought_price
                    total_profit += self.X_train[t] - bought_price
                    states_sell.append(t)

                done = (t == (len(self.X_train) - 2))
                self.memory.append((curr_state, action, reward, next_state, done))

                curr_state = next_state
                
                if t % replay_steps == 0:
                    self.expReplay()

            if plot:
                if e % logging_steps == 0:
                    print("Epsilon: " + str(self.epsilon))
                    print("--------------------------------")
                    print("Total Profit: " + formatPrice(total_profit))
                    print("--------------------------------")
                    PlotBehavior(dataset=self.X_train, states_buy=states_buy, states_sell=states_sell, profit=total_profit)

            if e % save_steps == 0:
                self.model.save(checkpoints_folder + "/model_episode" + str(e) + ".keras")
        
        self.model.save(model_name)

    def eval(self, verbose=True):
        if not self.is_eval:
            print("Warning: Agent is in training mode. Setting to evaluation mode")
            self.is_eval = True

        if not self.X_eval:
            raise Exception("Evaluation dataset not set")

        self.epsilon = 0.05

        curr_state = self.getState(0)
        self.inventoryHeap.clear()
        
        total_profit = 0
        states_sell_eval = []
        states_buy_eval = []      

        for t in tqdm(range(len(self.X_eval) - 1)):
            action = self.act(curr_state)
            next_state = self.getState(t + 1)
            reward = 0

            if action == 0:
                reward = 0
            elif action == 1:
                heapq.heappush(self.inventoryHeap, self.X_eval[t])
                reward = 0
                states_buy_eval.append(t)
                if verbose:
                    print("Buy: " + formatPrice(self.X_eval[t]))
            elif action == 2 and len(self.inventoryHeap) > 0:
                bought_price = heapq.heappop(self.inventoryHeap)
                total_profit += self.X_eval[t] - bought_price
                states_sell_eval.append(t)
                if verbose:
                    print("Sell: " + formatPrice(self.X_eval[t]) + " | profit: " + formatPrice(self.X_eval[t] - bought_price))

            done = (t == len(self.X_eval) - 2)
            self.memory.append((curr_state, action, reward, next_state, done))
            
            curr_state = next_state
            
        print("------------------------------------------")
        print("Total Profit: " + formatPrice(total_profit))
        print("------------------------------------------")
        PlotBehavior(self.X_eval, states_buy_eval, states_sell_eval, total_profit)


def Sigmoid(x):
    """
    Sigmoid activation function.

    For Deep Q-Learning, this function is used to normalize price differences, mapping them to a range
    between 0 and 1. This helps in creating more stable and consistent input features for the Q-network.

    Args:
        x: The input value, typically a price difference.

    Returns:
        The sigmoid of x, a value between 0 and 1.
    """
    return 1 / (1 + math.exp(-x))
