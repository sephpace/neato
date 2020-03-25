
import gym
import numpy as np

from ecosystem import Ecosystem


GENERATIONS = 30

# Set up environment
env = gym.make('CartPole-v0')

# Set up ecosystem
ecosystem = Ecosystem()
ecosystem.create_initial_population(30, input_length=4, output_length=2)

# Start the evolution
for generation in range(GENERATIONS):
    genomes = ecosystem.get_population()
    for i in range(len(genomes)):
        genome = genomes[i]
        observation = env.reset()
        done = False
        while not done:
            env.render()
            action = np.array(genome(observation)).argmax()
            observation, reward, done, info = env.step(action)
            genome.set_fitness(genome.get_fitness() + reward)
            print(f'Genome: {i}', ecosystem)
    ecosystem.next_generation()
