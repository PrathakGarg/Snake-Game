import random
from collections import deque

import torch
import numpy as np

from game import SnakeGameAI, Direction, Point
from model import LinearQNet, QTrainer
from helper import plot

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = LinearQNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y + 20)
        point_d = Point(head.x, head.y - 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            ((dir_u and game.is_collision(point_u)) or
             (dir_d and game.is_collision(point_d)) or
             (dir_l and game.is_collision(point_l)) or
             (dir_r and game.is_collision(point_r))),

            ((dir_u and game.is_collision(point_r)) or
             (dir_r and game.is_collision(point_d)) or
             (dir_d and game.is_collision(point_l)) or
             (dir_l and game.is_collision(point_u))),

            ((dir_u and game.is_collision(point_l)) or
             (dir_l and game.is_collision(point_d)) or
             (dir_d and game.is_collision(point_r)) or
             (dir_r and game.is_collision(point_u))),

            dir_l, dir_r, dir_u, dir_d,

            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            final_move[random.randint(0, 2)] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            final_move[torch.argmax(prediction).item()] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean = []
    total_score = 0
    highscore = 0

    agent = Agent()
    game = SnakeGameAI()

    while True:
        cur_state = agent.get_state(game)
        move = agent.get_action(cur_state)

        reward, done, score = game.play_step(move)
        new_state = agent.get_state(game)

        agent.train_short_memory(cur_state, move, reward, new_state, done)

        agent.remember(cur_state, move, reward, new_state, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > highscore:
                highscore = score
                agent.model.save()

            print('Game ', agent.n_games, "\nScore: ", score, "\nBest: ", highscore)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean.append(mean_score)
            plot(plot_scores, plot_mean)


if __name__ == '__main__':
    train()
