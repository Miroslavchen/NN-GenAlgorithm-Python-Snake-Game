import pygame, sys, random
import numpy as np

class game:
    def __init__(self, render_mode=None, squere=5):
        #Ініціалізація базових перемінних
        self.render_mode = render_mode

        self.size_x = 21
        self.size_y = 21
        self.frame_size_x = self.size_x * 10
        self.frame_size_y = self.size_y * 10

        self.squere = squere

        #Створення квадрата взаємностей
        self.dict_len = {}

        for i in range(22):
            if i % 2 != 0 and i != 0 and i != 1:
                self.dict_len[i] = -10 * ((i - 1) // 2)

        self.metrix = []

        y = 10

        for i in range(1, squere+1):
            x = self.dict_len[squere]
            for j in range(1, squere+1):
                self.metrix.append([x, y])
                x += 10
            y -= 10


        if self.render_mode != "rgb_array":
            #Інізіалізація гри
            self.check_errors = pygame.init()
            if self.check_errors[1] > 0:
                print(f'[!] Had {self.check_errors[1]} errors when initialising game, exiting...')
                sys.exit(-1)
            else:
                print('[+] Game successfully initialised')
            pygame.display.set_caption('Snake Eater')
            self.game_window = pygame.display.set_mode((self.frame_size_x, self.frame_size_y))
            self.fps_controller = pygame.time.Clock()

            #Кольори
            self.black = pygame.Color(0, 0, 0)
            self.white = pygame.Color(254, 254, 254)
            self.red = pygame.Color(255, 0, 0)
            self.green = pygame.Color(0, 255, 0)
            self.blue = pygame.Color(0, 0, 255)
        
    def __game_over(self):
        if self.render_mode != "rgb_array":
            my_font = pygame.font.SysFont('times new roman', 90)
            game_over_surface = my_font.render('YOU DIED', True, self.red)
            game_over_rect = game_over_surface.get_rect()
            game_over_rect.midtop = (self.frame_size_x / 2, self.frame_size_y / 4)
            self.game_window.fill(self.black)
            self.game_window.blit(game_over_surface, game_over_rect)
            self.__show_score(0, self.red, 'times', 20)
            pygame.display.flip()
            pygame.quit()
            sys.exit()
        elif self.render_mode == "rgb_array":
            pass

    def __show_score(self, choice, color, font, size):
        if self.render_mode != "rgb_array":
            score_font = pygame.font.SysFont(font, size)
            score_surface = score_font.render('Score : ' + str(self.score), True, color)
            score_rect = score_surface.get_rect()
            if choice == 1:
                score_rect.midtop = (self.frame_size_x / 10, 15)
            else:
                score_rect.midtop = (self.frame_size_x / 2, self.frame_size_y / 1.25)
            self.game_window.blit(score_surface, score_rect)
        elif self.render_mode == "rgb_array":
            pass
    
    def is_true(self):
        self.food_pos = [random.randrange(1, (self.frame_size_x // 10)) * 10,
                         random.randrange(1, (self.frame_size_y // 10)) * 10]
        for body in self.snake_body:
            if self.food_pos[0] == body[0] and self.food_pos[1] == body[1]:
                self.is_true()
        return self.food_pos

    def reset(self):
        self.score = 0

        self.snake_pos = [100, 50]
        self.snake_body = [[100, 50], [100 - 10, 50], [100 - (2 * 10), 50]]
        self.snake_on_eat = False

        self.food_pos = [random.randrange(1, (self.size_x)) * 10, random.randrange(1, (self.frame_size_y // 10)) * 10]
        self.food_spawn = True

        self.direction = 'RIGHT'
        self.change_to = self.direction

        self.observation = []
        for i in range(self.squere):
            self.observation.append([0 for i in range(self.squere)])

        self.reward = 0

        return self.observation

    def step(self, action, difficulty=500):
        self.snake_on_eat = False

        screen = []

        for i in range(self.squere ** 2):
            if self.snake_body[0][0] - self.metrix[i][0] == self.food_pos[0] and self.snake_body[0][1] - \
                    self.metrix[i][1] == self.food_pos[1]:
                screen.append(1)
                # if self.render_mode != "rgb_array":
                #     print("I`m here")
            elif self.metrix[i][0]==0 and self.metrix[i][1]==0:
                screen.append(7)
            elif self.snake_body[0][0] - self.metrix[i][0] <= 0 or self.snake_body[0][1] - self.metrix[i][1] <= 0 or \
                    self.snake_body[0][0] - self.metrix[i][0] >= self.frame_size_x or self.snake_body[0][1] - \
                    self.metrix[i][1] >= self.frame_size_y or \
                    self.snake_body[0][0] - self.metrix[i][0] == self.snake_body[0][0] and self.snake_body[0][1] - \
                    self.metrix[i][1] == self.snake_body[0][1] or self.snake_body[0][0] - self.metrix[i][0] == \
                    self.snake_body[0][0] - 10 and self.snake_body[0][1] - self.metrix[i][1] == self.snake_body[0][
                1] or self.snake_body[0][0] - self.metrix[i][0] == self.snake_body[0][0] + 10 and \
                    self.snake_body[0][1] - self.metrix[i][1] == self.snake_body[0][1] or self.snake_body[0][0] - \
                    self.metrix[i][0] == self.snake_body[0][0] and self.snake_body[0][1] - self.metrix[i][1] == \
                    self.snake_body[0][1] - 10 or self.snake_body[0][0] - self.metrix[i][0] == self.snake_body[0][
                0] and self.snake_body[0][1] - self.metrix[i][1] == self.snake_body[0][1] + 10:
                screen.append(-1)
            else:
                screen.append(0)

        screen = np.array(screen)

        direct = action

        if self.render_mode != "rgb_array":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == ord('k'):
                        return [screen, -1.0, True]

                    elif event.key == ord("w"):
                        direct = 0
                    elif event.key == ord("s"):
                        direct = 1
                    elif event.key == ord("a"):
                        direct = 2
                    elif event.key == ord("d"):
                        direct = 3
        if direct == 0 and self.change_to != 'DOWN':
            self.change_to = 'UP'

        if direct == 1 and self.change_to != 'UP':
            self.change_to = 'DOWN'

        if direct == 2 and self.change_to != 'RIGHT':
            self.change_to = 'LEFT'

        if direct == 3 and self.change_to != 'LEFT':
            self.change_to = 'RIGHT'

        if self.change_to == 'UP':
            self.direction = 'UP'
        if self.change_to == 'DOWN':
            self.direction = 'DOWN'
        if self.change_to == 'LEFT':
            self.direction = 'LEFT'
        if self.change_to == 'RIGHT':
            self.direction = 'RIGHT'

        if self.direction == 'UP':
            self.snake_pos[1] -= 10
        if self.direction == 'DOWN':
            self.snake_pos[1] += 10
        if self.direction == 'LEFT':
            self.snake_pos[0] -= 10
        if self.direction == 'RIGHT':
            self.snake_pos[0] += 10

        self.snake_body.insert(0, list(self.snake_pos))

        if not self.food_spawn:
            self.is_true()
        self.food_spawn = True
        
        if self.render_mode != "rgb_array":
            self.game_window.fill(self.black)
            for pos in self.snake_body:
                pygame.draw.rect(self.game_window, self.red, pygame.Rect(pos[0], pos[1], 10, 10))

            pygame.draw.rect(self.game_window, self.white, pygame.Rect(self.food_pos[0], self.food_pos[1], 10, 10))

            pygame.display.update()

            self.fps_controller.tick(difficulty)

        if self.snake_pos[0] < 0 or self.snake_pos[0] > self.frame_size_x - 10:
            if self.render_mode != "rgb_array":
                print(f'Він отримав: {self.score}\nВін врізався в стінку!')
            self.score = 0
            return [screen, -1.0 + 1.0*self.score, True]

        if self.snake_pos[1] < 0 or self.snake_pos[1] > self.frame_size_y - 10:
            if self.render_mode != "rgb_array":
                print(f'Він отримав: {self.score}\nВін врізався в стінку!!')
            self.score = 0
            return [screen, -1.0 + 1.0*self.score, True]

        for block in self.snake_body[1:]:
            if self.snake_pos[0] == block[0] and self.snake_pos[1] == block[1]:
                if self.render_mode != "rgb_array":
                    print(f'Він отримав: {self.score}\nВін сам себе вбив!')
                self.score = 0
                return [screen, -1.0 + 1.0*self.score, True]



        if self.snake_pos[0] == self.food_pos[0] and self.snake_pos[1] == self.food_pos[1]:
            self.score += 1
            self.snake_on_eat = True
            self.food_spawn = False
            return [screen, 1.0*self.score, False]
        else:
            self.snake_body.pop()
            if self.render_mode != "rgb_array":
                print(screen)
                #(42-(abs((self.snake_pos[0]//10)-(self.food_pos[0]//10)))+(abs((self.snake_pos[1]//10)-(self.food_pos[1]//10))))
            return [screen, -0.000000001, False]
