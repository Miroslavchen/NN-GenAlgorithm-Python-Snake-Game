import time
import random
import matplotlib.pyplot as plt 
import numpy as np

from deap import base, creator, tools
import algelitism
from neuralnetwork import NNetwork
from Game.Snake import game

# Налаштування гри
insquere = 5
env = game(render_mode="rgb_array", squere=insquere)

# Налаштування нейронної мережі
NEURONS_IN_LAYERS = [insquere**2, 24, 6, 4]
network = NNetwork(*NEURONS_IN_LAYERS)

LENGTH_CHROM = NNetwork.getTotalWeights(*NEURONS_IN_LAYERS)
LOW, UP, ETA = -1.0, 1.0, 90

# Налаштування генетичного алгоритму
POPULATION_SIZE = 200
P_CROSSOVER = 0.8
P_MUTATION = 0.65  # Збільшення ймовірності мутації для більшої різноманітності
MAX_GENERATIONS = 150  # Збільшення кількості поколінь для кращого навчання
HALL_OF_FAME_SIZE = 30  # Збільшення розміру зали слави

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
    """Обчислює загальну винагороду для даної особини (хромосоми)."""
    network.set_weights(individual)
    observation = env.reset()
    action_counter = 0
    total_reward = 0
    done = False

    distance = lambda a, b: np.sqrt(np.sum((np.array(a) - np.array(b))**2))  # Конвертація списків в масиви
    prev_dist = distance(env.snake_pos, env.food_pos)
    
    while not done and action_counter < 200:
        action_counter += 1
        ac = network.predict(observation)
        action = int(np.argmax(ac))
        observation, reward, done = env.step(action)
        
        current_dist = distance(env.snake_pos, env.food_pos)
        if current_dist < prev_dist:
            reward += 0.00001  # Заохочення за наближення до їжі
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

# Запуск генетичного алгоритму з елементом елітизму
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

# Візуалізація результатів
plt.plot(max_fitness_values, color='red')
plt.plot(mean_fitness_values, color='green')
plt.xlabel('Покоління')
plt.ylabel('Макс/середня пристосованість')
plt.title('Залежність максимального та середнього пристосування від покоління')
plt.show()

# Використання найкращої знайденої хромосоми для гри
hromo = hof.items[0]
print(hromo)
network.set_weights(hromo)

env = game(render_mode='human', squere=insquere)
observation = env.reset()
action = int(np.argmax(network.predict(observation)))
count = 1

# Запуск гри
while True:
    while True:
        observation, reward, done = env.step(action, difficulty=6)
        if done:
            break
        time.sleep(0.03)
        action = int(np.argmax(network.predict(observation)))
    env.reset()
    print("Ітерація:", count)
    count += 1