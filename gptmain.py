import time
import random
import matplotlib.pyplot as plt 
import numpy as np

from deap import base, creator, tools
import algelitism
from neuralnetwork import NNetwork
from Game.Snake import game

# Настройки игры
insquere = 5
env = game(render_mode="rgb_array", squere=insquere)

# Настройки нейронной сети
NEURONS_IN_LAYERS = [insquere**2, 24, 6, 3, 2, 8, 64, 4] 
network = NNetwork(*NEURONS_IN_LAYERS)

LENGTH_CHROM = NNetwork.getTotalWeights(*NEURONS_IN_LAYERS)
LOW, UP, ETA = -1.0, 1.0, 2

# Настройки генетического алгоритма
POPULATION_SIZE = 200
P_CROSSOVER = 0.8
P_MUTATION = 0.3  # Увеличение вероятности мутации для большего разнообразия
MAX_GENERATIONS = 300  # Увеличение количества поколений для лучшего обучения
HALL_OF_FAME_SIZE = 2  # Увеличение размера зала славы

hof = tools.HallOfFame(HALL_OF_FAME_SIZE)
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("randomWeight", random.uniform, LOW, UP)
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.randomWeight, LENGTH_CHROM)
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

population = toolbox.populationCreator(n=POPULATION_SIZE)

def getScore(individual):
    """Вычисляет общую награду для данного индивида (хромосомы)."""
    network.set_weights(individual)
    observation = env.reset()
    action_counter = 0
    total_reward = 0
    done = False

    distance = lambda a, b: np.sqrt(np.sum((np.array(a) - np.array(b))**2))  # Преобразование списков в массивы
    prev_dist = distance(env.snake_pos, env.food_pos)
    
    while not done and action_counter < 200:
        action_counter += 1
        ac = network.predict(observation)
        action = int(np.argmax(ac))
        observation, reward, done = env.step(action)
        
        current_dist = distance(env.snake_pos, env.food_pos)
        if current_dist < prev_dist:
            reward += 0.00001  # Поощрение за приближение к пище
        prev_dist = current_dist
        
        total_reward += reward

    return total_reward,

toolbox.register("evaluate", getScore)
toolbox.register("select", tools.selTournament, tournsize=90)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=LOW, up=UP, eta=ETA)
toolbox.register("mutate", tools.mutPolynomialBounded, low=LOW, up=UP, eta=ETA, indpb=1.0/LENGTH_CHROM)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("max", np.max)
stats.register("avg", np.mean)

# Запуск генетического алгоритма с элементом элитизма
population, logbook = algelitism.eaSimpleElitism(
    population, toolbox,
    cxpb=P_CROSSOVER,
    mutpb=P_MUTATION,
    ngen=MAX_GENERATIONS,
    halloffame=hof,
    stats=stats,
    verbose=True
)

max_fitness_values, mean_fitness_values = logbook.select("max", "avg")

# Визуализация результатов
plt.plot(max_fitness_values, color='red')
plt.plot(mean_fitness_values, color='green')
plt.xlabel('Поколение')
plt.ylabel('Макс/средняя приспособленность')
plt.title('Зависимость максимальной и средней приспособленности от поколения')
plt.show()

# Использование лучшей найденной хромосомы для игры
hromo = hof.items[0]
print(hromo)
network.set_weights(hromo)

env = game(render_mode='human', squere=insquere)
observation = env.reset()
action = int(np.argmax(network.predict(observation)))
count = 1

# Запуск игры
while True:
    while True:
        observation, reward, done = env.step(action, difficulty=6)
        if done:
            break
        time.sleep(0.03)
        action = int(np.argmax(network.predict(observation)))
    env.reset()
    print("Итерация:", count)
    count += 1
