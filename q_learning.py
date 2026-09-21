import random
import numpy as np


class QLearningAgent:

    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        # States: LOW, MEDIUM, HIGH
        self.states = ["LOW", "MEDIUM", "HIGH"]

        # Actions: signal duration
        self.actions = [10, 20, 30]

        # Q-table
        self.q_table = np.zeros((len(self.states), len(self.actions)))

    def get_state_index(self, state):
        return self.states.index(state)

    def choose_action(self, state):

        state_index = self.get_state_index(state)

        # Exploration
        if random.random() < self.epsilon:
            return random.choice(self.actions)

        # Exploitation
        best_action_index = np.argmin(self.q_table[state_index])

        return self.actions[best_action_index]

    def update(self, state, action, cost, next_state):

        state_index = self.get_state_index(state)
        next_state_index = self.get_state_index(next_state)
        action_index = self.actions.index(action)

        current_q = self.q_table[state_index][action_index]

        future_q = np.min(self.q_table[next_state_index])

        new_q = current_q + self.learning_rate * (
            cost
            + self.discount_factor * future_q
            - current_q
        )

        self.q_table[state_index][action_index] = new_q