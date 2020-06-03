
import gym

from ecosystem import Ecosystem
from genome import Genome


POPULATION = 100
GENERATIONS = 10


def evaluate_genome(genome, env, render=False):
    """
    Evaluates the given genome in the given environment.

    Args:
        genome (Genome): The genome.
        env (TimeLimit): The gym environment.
        render (bool):   Whether the environment should be rendered or not.
    """
    observation = env.reset()  # The initial input

    done = False
    while not done:
        # Render the environment
        if render:
            env.render()

        # Evaluate genome for given timestep
        action = genome(observation).argmax()               # Forward pass for genome
        observation, reward, done, info = env.step(action)  # Evaluate reward and get next input
        genome.fitness += reward                            # Update genome fitness


if __name__ == '__main__':
    # Set up environment
    env = gym.make('MountainCar-v0')

    # Set up ecosystem
    ecosystem = Ecosystem()
    genome = Genome(2, 3)
    ecosystem.create_initial_population(POPULATION, parent_genome=genome)

    # Start the evolution
    for generation in range(GENERATIONS):
        for genome in ecosystem.get_population():
            evaluate_genome(genome, env)

        # Show fittest genome for current generation
        genome = ecosystem.get_best_genome()
        evaluate_genome(genome, env, render=True)

        # Display ecosystem info
        print(ecosystem)

        # Kill less fit genomes and cross fitter genomes
        ecosystem.next_generation()

    # Close the environment
    env.close()

    # Uncomment the code below to save the best genome

    # best_genome = ecosystem.get_best_genome()
    # best_genome.save('cartpole.neat')
