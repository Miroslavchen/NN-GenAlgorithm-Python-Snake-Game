import time

from deap import base, algorithms
from deap import creator
from deap import tools

import algelitism
from neuralnetwork import NNetwork

import random
import matplotlib.pyplot as plt 
import numpy as np

from Game.Snake import game

hromo = None
if hromo == None:
    haveh = False
else:
    haveh = True

insquere = 17
env = game(render_mode="rgb_array", squere=insquere)

NEURONS_IN_LAYERS = [insquere**2, 24, 6, 4] 
network = NNetwork(*NEURONS_IN_LAYERS)

LENGTH_CHROM = NNetwork.getTotalWeights(*NEURONS_IN_LAYERS)
LOW = -1.0
UP = 1.0
ETA = 20

# Константи генетичного алгоритма
POPULATION_SIZE = 100   # Кількість індивидиумів в одній популяції
P_CROSSOVER = 0.8       # Ймовірність скрещення
P_MUTATION = 0.2         # Ймовірність мутації гена
MAX_GENERATIONS = 50   # Максимальна кількість поколінь
HALL_OF_FAME_SIZE = 5   # Зала слави

hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


toolbox = base.Toolbox()
toolbox.register("randomWeight", random.uniform, -1.0, 1.0)
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.randomWeight, LENGTH_CHROM)
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

population = toolbox.populationCreator(n=POPULATION_SIZE)

if not haveh:
    def getScore(individual):
        network.set_weights(individual)

        observation = env.reset()
        actionCounter = 0
        totalReward = 0

        done = False
        while not done and actionCounter < 200:
            actionCounter += 1
            ac = network.predict(observation)
            action = int(np.argmax(ac))
            observation, reward, done = env.step(action)
            totalReward += reward

        return totalReward,


    toolbox.register("evaluate", getScore)
    toolbox.register("select", tools.selTournament, tournsize=2)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=LOW, up=UP, eta=ETA)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=LOW, up=UP, eta=ETA, indpb=1.0/LENGTH_CHROM)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)


    #algelitism.eaSimpleElitism
    #algorithms.eaSimple
    population, logbook = algelitism.eaSimpleElitism(population, toolbox,
                                            cxpb=P_CROSSOVER,
                                            mutpb=P_MUTATION,
                                            ngen=MAX_GENERATIONS,
                                            halloffame=hof,
                                            stats=stats,
                                            verbose=True)

    maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")

    hromo = hof.items[0]
    print(hromo)

    plt.plot(maxFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Покоління')
    plt.ylabel('Макс/средня пристосування')
    plt.title('Залежність максимального та середнього пристосування від покоління')
    plt.show()

network.set_weights(hromo)

env = game(render_mode='human', squere=insquere)

observation = env.reset()
ac = network.predict(observation)
action = int(np.argmax(ac))
coun = 1

while True:
    while True:
        observation, reward, done = env.step(action, difficulty=6)

        if done:
            break

        time.sleep(0.03)
        ac = network.predict(observation)
        action = int(np.argmax(ac))
    env.reset()
    print("Its", coun)
    coun += 1