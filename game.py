import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple, deque
from pygame.locals import OPENGL, DOUBLEBUF
from OpenGL.GL import *

pygame.init()
font = pygame.font.Font('arial.ttf', 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 200


class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT

        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iter = 0
        self.loop_check = deque([Point(self.w / 2, self.h / 2)], maxlen=4)
        self.loops = 0

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iter += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            # if pygame.mouse.get_pressed()[0]:
            #     # print("henlo")
            #     global SPEED
            #     self.display = pygame.display.set_mode((self.w, self.h), flags=pygame.HIDDEN)
            #     SPEED = 2000
            # if pygame.mouse.get_pressed()[1]:
            #     self.display = pygame.display.set_mode((self.w, self.h), flags=OPENGL | DOUBLEBUF)


        # 2. move
        self._move(action)  # update the head
        self.snake.insert(0, self.head)

        # 3. check if game over
        game_over = False
        reward = 0
        if self.is_collision() or self.frame_iter > 100 * len(self.snake):
            game_over = True
            reward = -20
            return reward, game_over, self.score

        # Detect Loop
        if self.frame_iter >= 1:
            if self.head == self.loop_check[0]:
                # print("loop detected")
                self.loops += 1
                if self.loops <= 25:
                    reward -= 2
            self.loop_check.append(Point(self.head.x, self.head.y))

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 150*np.log(self.score+1)
            # reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # [straight, right, left]
        clockwise_dir = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        dir_ind = clockwise_dir.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clockwise_dir[dir_ind]
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = clockwise_dir[(dir_ind + 1) % 4]
        else:
            new_dir = clockwise_dir[(dir_ind - 1) % 4]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y

        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
