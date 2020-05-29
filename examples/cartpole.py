
import gym
import numpy as np

from ecosystem import Ecosystem


POPULATION = 30
GENERATIONS = 30

# Set up environment
env = gym.make('CartPole-v0')

# Set up ecosystem
ecosystem = Ecosystem()
ecosystem.create_initial_population(POPULATION, input_size=4, output_size=2)

# Start the evolution
for generation in range(GENERATIONS):
    genomes = ecosystem.get_population()
    for i, genome in enumerate(genomes):
        observation = env.reset()
        done = False
        while not done:
            env.render()
            action = np.array(genome(observation)).argmax()
            observation, reward, done, info = env.step(action)
            genome.fitness += reward
            print(f'{ecosystem}  Genome: {i + 1}  Fitness: {genome.fitness}', end='\r')
        print()
    ecosystem.next_generation()

env.close()
