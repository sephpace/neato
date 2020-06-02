
from copy import copy
import random
import unittest

import numpy as np

import activations
from ecosystem import Ecosystem, Species, EcosystemError
from genome import Genome, Node, Connection


class TestEcosystem(unittest.TestCase):

    def test_add_genome(self):
        e = Ecosystem()

        # Test to make sure it creates a new species if there aren't any
        msg = 'Failed to create new species with genome!'
        g = Genome(2, 2)
        g.add_connection(0, 2, weight=0.4)
        g.add_connection(0, 3, weight=0.4)
        g.add_connection(1, 2, weight=0.4)
        g.add_connection(1, 3, weight=0.4)
        self.assertEqual(len(e.species), 0, msg)
        e.add_genome(g)
        self.assertEqual(len(e.species), 1, msg)
        self.assertEqual(e.species[0].representative, g, msg)
        self.assertEqual(e.species[0][0], g, msg)
        self.assertEqual(len(e.get_population()), 1, msg)

        # Test to make sure it adds the genome to the same species if they are close
        msg = 'Failed to add genome to species correctly!'
        g2 = Genome(2, 2)
        g2.add_connection(0, 2, weight=0.3)
        g2.add_connection(0, 3, weight=0.4)
        g2.add_connection(1, 2, weight=0.4)
        g2.add_connection(1, 3, weight=0.4)
        e.add_genome(g2)
        self.assertEqual(len(e.species), 1, msg)
        self.assertEqual(e.species[0].representative, g, msg)
        self.assertEqual(e.species[0][1], g2, msg)
        self.assertEqual(len(e.get_population()), 2, msg)

        # Test to make sure a new species is created if the genome is distant from the current species
        msg = 'Failed to create new species with genome!'
        g3 = Genome(2, 2)
        g3.add_connection(0, 2, weight=0.3)
        g3.add_connection(0, 3, weight=0.4)
        g3.add_connection(1, 2, weight=0.4)
        g3.add_connection(1, 3, weight=0.4)
        g3.add_node(0)
        g3.add_node(1)
        g3.add_node(2)
        g3.add_node(3)
        e.add_genome(g3)
        self.assertEqual(len(e.species), 2, msg)
        self.assertEqual(e.species[1].representative, g3, msg)
        self.assertEqual(e.species[1][0], g3, msg)
        self.assertEqual(len(e.get_population()), 3, msg)

    def test_adjust_fitness(self):
        msg = 'Adjusted fitness incorrectly!'

        e = Ecosystem(threshold=0.5, disjoint_coefficient=1.0, excess_coefficient=1.0, weight_coefficient=0.4)

        # Create genomes
        g = Genome(2, 2)
        g2 = Genome(2, 2)
        g3 = Genome(2, 2)

        # Add connections and/or nodes
        g.add_connection(0, 2, weight=1.0)
        g.add_connection(0, 3, weight=1.0)
        g.add_connection(1, 2, weight=1.0)

        g2.add_connection(0, 2, weight=0.9)
        g2.add_connection(0, 3, weight=1.0)
        g2.add_connection(1, 2, weight=1.0)
        g2.add_connection(1, 3, weight=1.0)

        g3.add_connection(0, 2, weight=0.1)
        g3.add_connection(0, 3, weight=1.0)
        g3.add_connection(1, 2, weight=1.0)
        g3.add_connection(1, 3, weight=1.0)
        g3.add_node(0)
        g3.add_node(1)
        g3.add_node(2)
        g3.add_connection(0, 6, weight=1.0)
        g3.add_connection(1, 5, weight=1.0)

        g4 = copy(g3)
        g4.add_node(3)

        # Set genome fitness
        g.fitness = 10
        g2.fitness = 12
        g3.fitness = 22
        g4.fitness = 24

        # Test to make sure it doesn't change if there are no different genomes
        e.add_genome(g)
        e.adjust_fitness(g)
        self.assertEqual(g.fitness, 10, msg)

        e.add_genome(g2)
        e.adjust_fitness(g)
        e.adjust_fitness(g2)
        self.assertEqual(g.fitness, 10, msg)
        self.assertEqual(g2.fitness, 12, msg)

        # Test to make sure the fitness does change when there are different genomes
        e.add_genome(g3)
        e.add_genome(g4)
        e.adjust_fitness(g)
        e.adjust_fitness(g2)
        e.adjust_fitness(g3)
        e.adjust_fitness(g4)
        self.assertEqual(g.fitness, 5, msg)
        self.assertEqual(g2.fitness, 6, msg)
        self.assertEqual(g3.fitness, 11, msg)
        self.assertEqual(g4.fitness, 12, msg)

    def test_adjust_population_fitness(self):
        msg = 'Adjusted fitness incorrectly!'

        e = Ecosystem(threshold=0.5, disjoint_coefficient=1.0, excess_coefficient=1.0, weight_coefficient=0.4)

        # Create genomes
        g = Genome(2, 2)
        g2 = Genome(2, 2)
        g3 = Genome(2, 2)

        # Add connections and/or nodes
        g.add_connection(0, 2, weight=1.0)
        g.add_connection(0, 3, weight=1.0)
        g.add_connection(1, 2, weight=1.0)

        g2.add_connection(0, 2, weight=0.9)
        g2.add_connection(0, 3, weight=1.0)
        g2.add_connection(1, 2, weight=1.0)
        g2.add_connection(1, 3, weight=1.0)

        g3.add_connection(0, 2, weight=0.1)
        g3.add_connection(0, 3, weight=1.0)
        g3.add_connection(1, 2, weight=1.0)
        g3.add_connection(1, 3, weight=1.0)
        g3.add_node(0)
        g3.add_node(1)
        g3.add_node(2)
        g3.add_connection(0, 6, weight=1.0)
        g3.add_connection(1, 5, weight=1.0)

        g4 = copy(g3)
        g4.add_node(3)

        # Set genome fitness
        g.fitness = 10
        g2.fitness = 12
        g3.fitness = 22
        g4.fitness = 24

        # Test to make sure it doesn't change if there are no different genomes
        e.add_genome(g)
        e.adjust_population_fitness()
        self.assertEqual(g.fitness, 10, msg)

        e.add_genome(g2)
        e.adjust_population_fitness()
        self.assertEqual(g.fitness, 10, msg)
        self.assertEqual(g2.fitness, 12, msg)

        # Test to make sure the fitness does change when there are different genomes
        e.add_genome(g3)
        e.add_genome(g4)
        e.adjust_population_fitness()
        self.assertEqual(g.fitness, 5, msg)
        self.assertEqual(g2.fitness, 6, msg)
        self.assertEqual(g3.fitness, 11, msg)
        self.assertEqual(g4.fitness, 12, msg)

    def test_create_initial_population(self):
        e = Ecosystem()
        population_size = 20

        # Test to make sure it catches invalid parameters
        msg = 'Invalid parameters!'

        no_parent = False
        try:
            e.create_initial_population(population_size)
        except EcosystemError:
            no_parent = True
        self.assertTrue(no_parent, msg)

        no_input = False
        try:
            e.create_initial_population(population_size, output_size=3)
        except EcosystemError:
            no_input = True
        self.assertTrue(no_input, msg)

        no_output = False
        try:
            e.create_initial_population(population_size, input_size=3)
        except EcosystemError:
            no_output = True
        self.assertTrue(no_output, msg)

        # Test to make sure an inital population is created with input and output sizes
        msg = 'Failed to create initial population correctly!'
        e.create_initial_population(population_size, input_size=3, output_size=3)
        self.assertEqual(len(e.get_population()), population_size, msg)
        for g in e.get_population():
            input_count = 0
            output_count = 0
            for n in g.get_nodes():
                if n.type == 'input':
                    input_count += 1
                elif n.type == 'output':
                    output_count += 1
            self.assertEqual(input_count, 3, msg)
            self.assertEqual(output_count, 3, msg)

        # Test to make sure an initial population is created with a parent genome
        parent = Genome(2, 3)
        e.create_initial_population(15, parent_genome=parent, mutate=False)
        self.assertEqual(len(e.get_population()), 15, msg)
        for g in e.get_population():
            self.assertEqual(g, parent, msg)

        # Test to make sure mutations work
        msg = 'Failed to mutate initial population!'
        e.create_initial_population(100, parent_genome=parent)
        some_mutated = False
        for g in e.get_population():
            if g != parent:
                some_mutated = True
                break
        self.assertTrue(some_mutated, msg)

    def test_create_genome(self):
        e = Ecosystem()

        # Test to make sure it creates the genome correctly
        msg = 'Genome not created correctly!'
        e.create_genome(2, 2)
        self.assertEqual(len(e.species[0]), 1, msg)
        self.assertEqual(len(e.species[0][0].get_nodes()), 4, msg)
        e.create_genome(2, 3)
        self.assertEqual(len(e.species[0]), 2, msg)
        self.assertEqual(len(e.species[0][1].get_nodes()), 5)
        e.create_genome(45, 46)
        self.assertEqual(len(e.species[0]), 3, msg)
        self.assertEqual(len(e.species[0][2].get_nodes()), 91)

    def test_cross(self):
        e = Ecosystem()

        # Create genomes
        g = Genome(2, 2, ecosystem=e)
        g2 = Genome(2, 2, ecosystem=e)

        # Cross the genomes
        child = e.cross(g, g2)

        # Test child connections
        msg = 'Child connection doesn\'t exist within either parent!'
        for c in child.get_connections():
            self.assertTrue(g.get_connection(c.innovation_number) is not None or g2.get_connection(c.innovation_number) is not None, msg)

        # Test to make sure the child has the same amount of connections as the fitter parent
        msg = 'Child missing fitter parent connection(s)!'
        self.assertEqual(len(child.get_connections()), len(g.get_connections()), msg)

        # Test child nodes
        msg = 'Child node doesn\'t exist within either parent!'
        for n in child.get_nodes():
            self.assertTrue(g.get_node(n.id) is not None or g2.get_node(n.id) is not None, msg)

        # Test to make sure the child has the same amount of nodes as the fitter parent
        msg = 'Child is missing fitter parent node(s)!'
        self.assertEqual(len(child.get_nodes()), len(g.get_nodes()), msg)

        # Test preference for fit parents
        msg = 'Child connection preferred less fit parent!'
        for c in child.get_connections():
            in_both = g.get_connection(c.innovation_number) is not None and g2.get_connection(c.innovation_number) is not None
            in_fit_parent = g.get_connection(c.innovation_number) is not None and g2.get_connection(c.innovation_number) is None
            self.assertTrue(in_both or in_fit_parent, msg)

        # Add connections and nodes
        g.add_connection(0, 2)
        g.add_connection(0, 3)
        g.add_connection(1, 2)
        g.add_connection(1, 3)
        g.add_node(0)
        g.get_connections()[5].weight = 0.4

        g2.add_connection(0, 2)
        g2.add_connection(0, 3)
        g2.add_connection(1, 2)
        g2.add_connection(1, 3)
        g2.add_node(1)

        g.add_node(2)

        # Assign fitness to genomes
        g.fitness = 10
        g2.fitness = 5

        # Cross the genomes
        child = e.cross(g, g2)

        # Test child connections
        msg = 'Child connection doesn\'t exist within either parent!'
        for c in child.get_connections():
            self.assertTrue(g.get_connection(c.innovation_number) is not None or g2.get_connection(c.innovation_number) is not None, msg)

        # Test to make sure the child has the same amount of connections as the fitter parent
        msg = 'Child missing fitter parent connection(s)!'
        self.assertEqual(len(child.get_connections()), len(g.get_connections()), msg)

        # Test child nodes
        msg = 'Child node doesn\'t exist within either parent!'
        for n in child.get_nodes():
            self.assertTrue(g.get_node(n.id) is not None or g2.get_node(n.id) is not None, msg)

        # Test to make sure the child has the same amount of nodes as the fitter parent
        msg = 'Child is missing fitter parent node(s)!'
        self.assertEqual(len(child.get_nodes()), len(g.get_nodes()), msg)

        # Test preference for fit parents
        msg = 'Child connection preferred less fit parent!'
        for c in child.get_connections():
            in_both = g.get_connection(c.innovation_number) is not None and g2.get_connection(c.innovation_number) is not None
            in_fit_parent = g.get_connection(c.innovation_number) is not None and g2.get_connection(c.innovation_number) is None
            self.assertTrue(in_both or in_fit_parent, msg)

        # Swap the fitness and test again
        g.fitness = 5
        g2.fitness = 10

        # Cross the genomes
        child = e.cross(g, g2)

        # Test child connections
        msg = 'Child connection doesn\'t exist within either parent!'
        for c in child.get_connections():
            self.assertTrue(g.get_connection(c.innovation_number) is not None or g2.get_connection(c.innovation_number) is not None, msg)

        # Test to make sure the child has the same amount of connections as the fitter parent
        msg = 'Child missing fitter parent connection(s)!'
        self.assertEqual(len(child.get_connections()), len(g2.get_connections()), msg)

        # Test child nodes
        msg = 'Child node doesn\'t exist within either parent!'
        for n in child.get_nodes():
            self.assertTrue(g.get_node(n.id) is not None or g2.get_node(n.id) is not None, msg)

        # Test to make sure the child has the same amount of nodes as the fitter parent
        msg = 'Child is missing fitter parent node(s)!'
        self.assertEqual(len(child.get_nodes()), len(g2.get_nodes()), msg)

        # Test preference for fit parents
        msg = 'Child connection preferred less fit parent!'
        for c in child.get_connections():
            in_both = g.get_connection(c.innovation_number) is not None and g2.get_connection(c.innovation_number) is not None
            in_fit_parent = g.get_connection(c.innovation_number) is None and g2.get_connection(c.innovation_number) is not None
            self.assertTrue(in_both or in_fit_parent, msg)

    def test_get_best_genome(self):
        pop_size = 1000
        max_fitness = 100

        e = Ecosystem()
        e.create_initial_population(pop_size, input_size=2, output_size=2)
        pop = e.get_population()
        fitnesses = []
        for i in range(pop_size - 1):
            fitnesses.append(random.randrange(0, max_fitness - 1))
        fitnesses.insert(random.randint(0, pop_size - 1), max_fitness)
        for i in range(pop_size):
            pop[i].fitness = fitnesses[i]
        self.assertEqual(e.get_best_genome().fitness, max_fitness)

    def test_get_distance(self):
        error_margin = 0.000000000001

        e = Ecosystem(disjoint_coefficient=1.0, excess_coefficient=1.0, weight_coefficient=0.4)

        # Create genomes
        g = Genome(2, 2, ecosystem=e)
        g2 = Genome(2, 2, ecosystem=e)
        g3 = Genome(2, 2, ecosystem=e)

        # Add connections and/or nodes
        g.add_connection(0, 2, weight=1.0)
        g.add_connection(0, 3, weight=1.0)
        g.add_connection(1, 2, weight=1.0)

        g2.add_connection(0, 2, weight=0.9)
        g2.add_connection(0, 3, weight=1.0)
        g2.add_connection(1, 2, weight=1.0)
        g2.add_connection(1, 3, weight=1.0)

        g3.add_connection(0, 2, weight=0.1)
        g3.add_connection(0, 3, weight=1.0)
        g3.add_connection(1, 2, weight=1.0)
        g3.add_connection(1, 3, weight=1.0)
        g3.add_node(0)
        g3.add_node(1)
        g3.add_node(2)
        g3.add_connection(0, 6, weight=1.0)
        g3.add_connection(1, 5, weight=1.0)

        g.add_node(1)

        # Test distance results to make sure they are approximately correct
        msg = 'Invalid distance result!'
        self.assertAlmostEqual(e.get_distance(g, g2), 0.6133333333333334, msg=msg, delta=error_margin)
        self.assertAlmostEqual(e.get_distance(g2, g), 0.6133333333333334, msg=msg, delta=error_margin)
        self.assertAlmostEqual(e.get_distance(g, g3), 0.84, msg=msg, delta=error_margin)
        self.assertAlmostEqual(e.get_distance(g3, g), 0.84, msg=msg, delta=error_margin)
        self.assertAlmostEqual(e.get_distance(g2, g3), 0.7466666666666666, msg=msg, delta=error_margin)
        self.assertAlmostEqual(e.get_distance(g3, g2), 0.7466666666666666, msg=msg, delta=error_margin)

        # Test to make sure the same genome returns zero when tested against node_to_itself
        msg = 'Distance between same genomes is not zero!'
        self.assertEqual(e.get_distance(g, g), 0.0, msg)
        self.assertEqual(e.get_distance(g2, g2), 0.0, msg)
        self.assertEqual(e.get_distance(g3, g3), 0.0, msg)

    def test_get_innovation_number(self):
        e = Ecosystem()

        msg = 'Innovation number not unique!'

        # Make sure innovation numbers are unique
        inn_num = 0
        for i in range(10):
            for j in range(10, 20):
                self.assertEqual(e.get_innovation_number(i, j), inn_num, msg)
                inn_num += 1

        # Make sure it's the same twice
        inn_num = 0
        for i in range(10):
            for j in range(10, 20):
                self.assertEqual(e.get_innovation_number(i, j), inn_num, msg)
                inn_num += 1

    def test_kill(self):
        e = Ecosystem()

        # Test to make sure the genomes are successfully killed
        msg = 'Genome not killed!'
        for i in range(10):
            e.create_genome(2, 2)

        for g in e.get_population():
            e.kill(g)

        self.assertEqual(len(e.get_population()), 0, msg)

        # Test to make sure the species is removed if all its members are killed
        msg = 'Species not removed!'
        self.assertEqual(len(e.species), 0, msg)

    def test_kill_percentage(self):
        e = Ecosystem(disjoint_coefficient=1.0, excess_coefficient=1.0, weight_coefficient=0.4)

        # Create genomes
        g = Genome(2, 2, ecosystem=e)
        g2 = Genome(2, 2, ecosystem=e)
        g3 = Genome(2, 2, ecosystem=e)

        # Add connections and/or nodes
        g.add_connection(0, 2, weight=1.0)
        g.add_connection(0, 3, weight=1.0)
        g.add_connection(1, 2, weight=1.0)

        g2.add_connection(0, 2, weight=0.9)
        g2.add_connection(0, 3, weight=1.0)
        g2.add_connection(1, 2, weight=1.0)
        g2.add_connection(1, 3, weight=1.0)

        g3.add_connection(0, 2, weight=0.1)
        g3.add_connection(0, 3, weight=1.0)
        g3.add_connection(1, 2, weight=1.0)
        g3.add_connection(1, 3, weight=1.0)
        g3.add_node(0)
        g3.add_node(1)
        g3.add_node(2)
        g3.add_connection(0, 6, weight=1.0)
        g3.add_connection(1, 5, weight=1.0)

        g.add_node(1)

        # Test to make sure the correct amount are killed
        msg = 'Incorrect percentage killed!'
        for i in range(10):
            g_copy = g.copy()
            g2_copy = g2.copy()
            g3_copy = g3.copy()

            g_copy.fitness = random.randrange(0, 200)
            g2_copy.fitness = random.randrange(0, 200)
            g3_copy.fitness = random.randrange(0, 200)

            e.add_genome(g_copy)
            e.add_genome(g2_copy)
            e.add_genome(g3_copy)

        e.kill_percentage(50)
        for s in e.species:
            self.assertAlmostEqual(len(s), 5, msg=msg, delta=1)
        self.assertAlmostEqual(len(e.get_population()), 15, msg=msg, delta=1)

        for i in range(5):
            g_copy = g.copy()
            g2_copy = g2.copy()
            g3_copy = g3.copy()

            g_copy.fitness = random.randrange(0, 200)
            g2_copy.fitness = random.randrange(0, 200)
            g3_copy.fitness = random.randrange(0, 200)

            e.add_genome(g_copy)
            e.add_genome(g2_copy)
            e.add_genome(g3_copy)

        e.kill_percentage(25)
        for s in e.species:
            self.assertAlmostEqual(len(s), 8, msg=msg, delta=1)
        self.assertAlmostEqual(len(e.get_population()), 24, msg=msg, delta=1)

        for i in range(2):
            g_copy = g.copy()
            g2_copy = g2.copy()
            g3_copy = g3.copy()

            g_copy.fitness = random.randrange(0, 200)
            g2_copy.fitness = random.randrange(0, 200)
            g3_copy.fitness = random.randrange(0, 200)

            e.add_genome(g_copy)
            e.add_genome(g2_copy)
            e.add_genome(g3_copy)

        e.kill_percentage(75)
        for s in e.species:
            self.assertAlmostEqual(len(s), 3, msg=msg, delta=1)
        self.assertAlmostEqual(len(e.get_population()), 9, msg=msg, delta=1)

        e.kill_percentage(12)
        for s in e.species:
            self.assertAlmostEqual(len(s), 3, msg=msg, delta=1)
        self.assertAlmostEqual(len(e.get_population()), 9, msg=msg, delta=1)

        # Test with a larger population
        e.create_initial_population(100, parent_genome=g)
        e.add_genome(g2.copy())
        e.add_genome(g3.copy())

        e.kill_percentage(90)
        self.assertAlmostEqual(len(e.get_population()), 12, msg=msg, delta=3)

    def test_next_generation(self):
        e = Ecosystem()

        e.create_initial_population(100, input_size=2, output_size=2)

        for g in e.get_population():
            g.fitness = random.randrange(0, 200)

        for i in range(e.generation + 1, 5):
            e.next_generation()

            # Test to make sure the population size is the same from generation to generation
            msg = 'Population size different for next generation!'
            self.assertEqual(len(e.get_population()), 100, msg)

            # Test to make sure the generation number is incremented
            msg = 'Generation number not incremented!'
            self.assertEqual(e.generation, i, msg)


class TestSpecies(unittest.TestCase):
    def test_constructor(self):
        g = Genome(1, 1)
        s = Species(0, g)

        # Test to make sure attributes are set correctly.
        msg = 'Failed to assign species attributes correctly!'
        self.assertEqual(s.id, 0, msg)
        self.assertEqual(s.representative, g, msg)
        self.assertEqual(len(s), 1, msg)
        self.assertEqual(len(s), 1, msg)


class TestGenome(unittest.TestCase):

    def test_constructor(self):
        input_size = 2
        output_size = 2

        g = Genome(input_size, output_size)

        # Test if attributes are correct
        msg = 'Failed to assign genome attributes correctly!'
        self.assertEqual(len(g.get_nodes()), input_size + output_size, msg)
        self.assertEqual(len(g.get_connections()), 0)
        self.assertEqual(g.fitness, 0)

    def test__add_connection(self):
        g = Genome(2, 2)

        # Test connecting input and output nodes
        msg = 'Failed connection to output node!'
        g.add_connection(0, 2)
        self.assertEqual(len(g.get_connections()), 1, msg)
        self.assertEqual(g.get_connections()[0].in_node, 0, msg)
        self.assertEqual(g.get_connections()[0].out_node, 2, msg)

        g.add_connection(0, 3)
        self.assertEqual(len(g.get_connections()), 2, msg)
        self.assertEqual(g.get_connections()[1].in_node, 0, msg)
        self.assertEqual(g.get_connections()[1].out_node, 3, msg)

        g.add_connection(1, 2)
        self.assertEqual(len(g.get_connections()), 3, msg)
        self.assertEqual(g.get_connections()[2].in_node, 1, msg)
        self.assertEqual(g.get_connections()[2].out_node, 2, msg)

        g.add_connection(1, 3, 0.5)
        self.assertEqual(len(g.get_connections()), 4, msg)
        self.assertEqual(g.get_connections()[3].in_node, 1, msg)
        self.assertEqual(g.get_connections()[3].out_node, 3, msg)
        self.assertEqual(g.get_connections()[3].weight, 0.5, msg)

        # # Test connecting to hidden nodes
        msg = 'Failed connection to hidden node!'
        g.add_node(0)
        g.add_connection(1, 4, 0.7)
        self.assertEqual(len(g.get_connections()), 7, msg)
        self.assertEqual(g.get_connections()[6].in_node, 1, msg)
        self.assertEqual(g.get_connections()[6].out_node, 4, msg)
        self.assertEqual(g.get_connections()[6].weight, 0.7, msg)

        g.add_node(2)
        g.add_connection(4, 5, 0.9)
        self.assertEqual(len(g.get_connections()), 10, msg)
        self.assertEqual(g.get_connections()[9].in_node, 4, msg)
        self.assertEqual(g.get_connections()[9].out_node, 5, msg)
        self.assertEqual(g.get_connections()[9].weight, 0.9, msg)

        # Make sure duplicate connections aren't created
        msg = 'Duplicate connection added!'
        try:
            g.add_connection(0, 2)
        except AssertionError:
            pass
        self.assertEqual(len(g.get_connections()), 10, msg)

    def test_add_node(self):
        g = Genome(1, 1)

        # Test adding nodes
        msg = 'Node added incorrectly!'

        g.add_connection(0, 1)
        g.add_node(0)
        self.assertEqual(len(g.get_nodes()), 3, msg)
        self.assertEqual(len(g.get_connections()), 3, msg)
        self.assertFalse(g.get_connections()[0].expressed, msg)
        self.assertTrue(g.get_connections()[1].expressed, msg)
        self.assertTrue(g.get_connections()[2].expressed, msg)
        self.assertEqual(g.get_connections()[0].weight, g.get_connections()[1].weight, msg)
        self.assertEqual(g.get_connections()[2].weight, 1.0, msg)
        self.assertEqual(g.get_nodes()[2].activation, activations.modified_sigmoid, msg)

        g.add_node(1, activation=activations.absolute)
        self.assertEqual(len(g.get_nodes()), 4, msg)
        self.assertEqual(len(g.get_connections()), 5, msg)
        self.assertFalse(g.get_connections()[1].expressed, msg)
        self.assertTrue(g.get_connections()[3].expressed, msg)
        self.assertTrue(g.get_connections()[4].expressed, msg)
        self.assertEqual(g.get_connections()[1].weight, g.get_connections()[3].weight, msg)
        self.assertEqual(g.get_connections()[4].weight, 1.0, msg)
        self.assertEqual(g.get_nodes()[3].activation, activations.absolute, msg)

    def test_connections_at_max(self):
        g = Genome(2, 2)

        msg = 'Connections not at max and should be!'
        msg2 = 'Connections at max and shouldn\'t be!'

        # Test with no hidden nodes
        self.assertFalse(g.connections_at_max(), msg)
        g.add_connection(0, 2)
        g.add_connection(0, 3)
        g.add_connection(1, 2)
        self.assertFalse(g.connections_at_max(), msg)
        g.add_connection(1, 3)
        self.assertTrue(g.connections_at_max(), msg2)

        # Test with hidden nodes
        g.add_node(0)
        self.assertFalse(g.connections_at_max(), msg)
        g.add_connection(1, 4)
        g.add_connection(4, 3)
        self.assertTrue(g.connections_at_max(), msg2)
        g.add_node(6)
        self.assertFalse(g.connections_at_max(), msg)
        g.add_connection(0, 5)
        g.add_connection(5, 2)
        g.add_connection(5, 3)
        self.assertTrue(g.connections_at_max(), msg2)

    def test_copy(self):
        g = Genome(2, 2)

        # Test to make sure the copy is the same as the original
        msg = 'Copy is different from the original!'
        g.add_connection(0, 2)
        g.add_connection(0, 3)
        g.add_node(0)
        gc = g.copy()
        self.assertEqual(len(g.get_nodes()), len(gc.get_nodes()), msg)
        self.assertEqual(len(g.get_connections()), len(gc.get_connections()), msg)
        for i in range(len(g.get_connections())):
            self.assertEqual(g.get_connections()[i].weight, gc.get_connections()[i].weight, msg)

        # Test to make sure the copy doesn't change when the original does
        msg = 'Copy changes when original changes!'
        g.add_node(1)
        g.get_connections()[0].weight = 50
        self.assertNotEqual(len(g.get_nodes()), len(gc.get_nodes()), msg)
        self.assertNotEqual(len(g.get_connections()), len(gc.get_connections()), msg)
        self.assertNotEqual(g.get_connections()[0].weight, gc.get_connections()[0].weight, msg)

    def test_forward(self):
        error_margin = 0.000000000001

        g = Genome(2, 2)
        x = np.array([0.5, 0.5])

        msg = 'Invalid genome output!'

        # No hidden nodes, modified sigmoid activation
        g.add_connection(0, 2, weight=-0.7)
        g.add_connection(0, 3, weight=-0.1)
        g.add_connection(1, 2, weight=0.5)
        g.add_connection(1, 3, weight=0.9)
        y = g(x)
        self.assertAlmostEqual(y[0], 0.3798935676569099, msg=msg, delta=error_margin)
        self.assertAlmostEqual(y[1], 0.8765329524347759, msg=msg, delta=error_margin)

        # Different activation
        for n in g.get_nodes():
            if n.type != 'input':
                g.set_activation(n.id, activations.absolute)
        y = g(x)
        self.assertAlmostEqual(y[0], 0.1, msg=msg, delta=error_margin)
        self.assertAlmostEqual(y[1], 0.4, msg=msg, delta=error_margin)

        # With hidden nodes and different activations (sigmoid for the new ones)
        g.add_node(0, activation=activations.sigmoid)
        g.add_node(2, activation=activations.sigmoid)
        g.add_connection(4, 5, 0.5)
        y = g(x)
        self.assertAlmostEqual(y[0], 0.08953579350695234, msg=msg, delta=error_margin)
        self.assertAlmostEqual(y[1], 0.4, msg=msg, delta=error_margin)

        # Test with many hidden nodes in a line
        g2 = Genome(2, 2)
        g2.add_connection(0, 2, -0.3)
        g2.add_node(0)
        g2.add_node(2)
        g2.add_node(4)
        g2.add_node(6)
        g2.add_connection(1, 7, 0.7)
        g2.add_connection(4, 3, -0.2)
        y = g2(x)
        self.assertAlmostEqual(y[0], 0.18866305913528142, msg=msg, delta=error_margin)
        self.assertAlmostEqual(y[1], 0.2743863603871294, msg=msg, delta=error_margin)

        # Test with recursive loop
        g3 = Genome(1, 1)
        x = np.array([0.5])
        g3.add_connection(0, 1, 0.5)
        g3.add_node(0)
        g3.add_node(0)
        g3.add_node(1)
        g3.add_connection(3, 2, 0.5)
        g3.add_connection(4, 3, 0.5)
        y = g3(x)
        self.assertAlmostEqual(y[0], 0.9907948306148218, msg=msg, delta=error_margin)

    def test_get_node_inputs(self):
        g = Genome(2, 2)

        g.add_connection(0, 2)
        g.add_connection(0, 3)
        g.add_connection(1, 2)

        # Test to make sure it gets the inputs correctly
        msg = 'Node inputs are incorrect!'
        self.assertEqual(g.get_node_inputs(2), [0, 1], msg)
        self.assertEqual(g.get_node_inputs(3), [0], msg)

    def test_get_node_max_distance(self):
        g = Genome(2, 2)

        # Test to make sure empty genomes return the correct values for output nodes
        msg = 'Disconnected output node returned invalid value!'
        self.assertEqual(g.get_node_max_distance(g.get_node(2).id), -1, msg)
        self.assertEqual(g.get_node_max_distance(g.get_node(3).id), -1, msg)

        # Add nodes and connections
        g.add_connection(0, 2, weight=-0.7)
        g.add_connection(0, 3, weight=-0.1)
        g.add_connection(1, 2, weight=0.5)
        g.add_connection(1, 3, weight=0.9)
        g.add_node(0)
        g.add_node(2)
        g.add_connection(4, 5, 0.5)

        msg = 'Incorrect node max distance!'

        # Test the values of each node distance to make sure they are correct
        correct_distances = [0, 0, 3, 1, 1, 2]
        for node, distance in zip(g.get_nodes(), correct_distances):
            self.assertEqual(g.get_node_max_distance(node.id), distance, msg)

        # Add a node and test again
        g.add_node(8)
        correct_distances = [0, 0, 4, 1, 1, 3, 2]
        for node, distance in zip(g.get_nodes(), correct_distances):
            self.assertEqual(g.get_node_max_distance(node.id), distance, msg)

        # Add connection and test again
        g.add_connection(6, 3)
        correct_distances = [0, 0, 4, 3, 1, 3, 2]
        for node, distance in zip(g.get_nodes(), correct_distances):
            self.assertEqual(g.get_node_max_distance(node.id), distance, msg)

        # Test genome with connection loop
        msg = 'Genome failed to handle connection loop!'

        g2 = Genome(1, 1)
        g2.add_connection(0, 1)
        g2.add_node(0)
        g2.add_node(0)
        g2.add_node(1)
        g2.add_connection(3, 2)
        g2.add_connection(4, 3)
        correct_distances = [0, 4, 3, 3, 3]
        for node, distance in zip(g2.get_nodes(), correct_distances):
            self.assertEqual(g2.get_node_max_distance(node.id), distance, msg)

    def test_get_node_outputs(self):
        g = Genome(2, 2)

        g.add_connection(0, 2)
        g.add_connection(0, 3)
        g.add_connection(1, 2)

        # Test to make sure it gets the outputs correctly
        msg = 'Node outputs are incorrect!'
        self.assertEqual(g.get_node_outputs(0), [2, 3], msg)
        self.assertEqual(g.get_node_outputs(1), [2], msg)

    def test_mutate_add_connection(self):
        # Test adding a connection
        msg = 'Connection added incorrectly!'
        g = Genome(2, 2)

        for i in range(30):
            g = Genome(2, 2)
            g.mutate_add_connection()
            self.assertEqual(len(g.get_connections()), 1, msg)
            self.assertEqual(g.get_connections()[0].innovation_number, 0, msg)
            self.assertTrue(g.get_connections()[0].in_node in [n.id for n in g.get_nodes()], msg)
            self.assertTrue(g.get_connections()[0].out_node in [n.id for n in g.get_nodes()], msg)
            self.assertTrue(-1.0 <= g.get_connections()[0].weight <= 1.0, msg)
            self.assertTrue(g.get_connections()[0].expressed, msg)
            self.assertNotEqual(g.get_connections()[0].in_node, g.get_connections()[0].out_node)
            in_type = g.get_node(g.get_connections()[0].in_node).type
            out_type = g.get_node(g.get_connections()[0].out_node).type
            self.assertFalse(in_type == out_type != 'hidden')
            self.assertFalse((in_type == 'output' and out_type == 'input'))
            self.assertFalse((in_type == 'hidden' and out_type == 'input'))

        # Test to make sure connections are always added (unless at max)
        msg = 'Connection not added!'
        for i in range(2, 4):
            g.mutate_add_connection()
            self.assertEqual(len(g.get_connections()), i, msg)

        # Test to make sure it doesn't go above the maximum connections
        msg = 'Connections exceeded maximum amount!'
        g.mutate_add_connection()
        self.assertEqual(len(g.get_connections()), 4, msg)  # Shouldn't go past 4

    def test_mutate_add_node(self):
        g = Genome(1, 1)

        # Test to make sure you can't add a node without an existing connection
        msg = 'Node added without existing connection!'
        g.mutate_add_node()
        self.assertEqual(len(g.get_nodes()), 2, msg)

        # Test adding a node
        msg = 'Node added incorrectly!'
        g.mutate_add_connection()
        g.mutate_add_node()
        self.assertEqual(len(g.get_nodes()), 3, msg)
        self.assertEqual(g.get_nodes()[2].id, 2, msg)
        self.assertEqual(g.get_nodes()[2].type, 'hidden', msg)
        self.assertEqual(len(g.get_connections()), 3, msg)
        self.assertEqual(g.get_connections()[0].weight, g.get_connections()[1].weight, msg)
        self.assertEqual(g.get_connections()[2].weight, 1.0, msg)

        # Test to make sure you can't add a node without any expressed connections
        msg = "Node added to disabled connection!"
        for c in g.get_connections():
            c.disable()
        g.mutate_add_node()
        self.assertEqual(len(g.get_nodes()), 3, msg)

    def test_mutate_random_weight(self):
        g = Genome(1, 1)

        msg = 'Failed to set random connection weight!'

        # Test with one connection
        g.add_connection(0, 1, weight=5)
        g.mutate_random_weight()
        self.assertNotEqual(g.get_connections()[0].weight, 5, msg)
        self.assertTrue(-1.0, g.get_connections()[0].weight <= 1.0)

        # Test with multiple connections
        g.add_node(0)
        before = [c.weight for c in g.get_connections()]
        g.mutate_random_weight()
        after = [c.weight for c in g.get_connections()]
        self.assertNotEqual(before, after, msg)

    def test_mutate_shift_weight(self):
        g = Genome(1, 1)
        steps = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 5.0]
        error_margin = 0.000000000001

        msg = 'Failed to shift weight correctly!'

        # Test with one connection
        g.add_connection(0, 1, weight=0.0)
        for step in steps:
            current_weight = g.get_connections()[0].weight
            g.mutate_shift_weight(step=step)
            self.assertNotEqual(g.get_connections()[0].weight, current_weight, msg)
            self.assertAlmostEqual(abs(g.get_connections()[0].weight - current_weight), step, delta=error_margin)

        # Test with multiple connections
        g.add_node(0)
        for step in steps:
            before = [c.weight for c in g.get_connections()]
            g.mutate_shift_weight(step=step)
            after = [c.weight for c in g.get_connections()]
            self.assertNotEqual(before, after, msg)

    def test_mutate_toggle_connection(self):
        g = Genome(1, 1)

        msg = 'Failed to toggle connection!'

        # Test with one connection
        g.add_connection(0, 1)
        g.mutate_toggle_connection()
        self.assertFalse(g.get_connections()[0].expressed, msg)
        g.mutate_toggle_connection()
        self.assertTrue(g.get_connections()[0].expressed, msg)

        # Test with multiple connections
        g.add_node(0)
        before = [c.expressed for c in g.get_connections()]
        g.mutate_toggle_connection()
        after = [c.expressed for c in g.get_connections()]
        self.assertNotEqual(before, after, msg)

    def test_set_activation(self):
        g = Genome(3, 3)
        node_ids = range(len(g.get_nodes()))
        act = [activations.relu, activations.gaussian, activations.square]

        # Test to make sure the activation is set correctly
        msg = 'Activation set incorrectly!'
        for nid in node_ids:
            for a in act:
                g.set_activation(nid, a)
                self.assertEqual(g.get_node(nid).activation, a)

    def test_set_connections(self):
        g = Genome(2, 2)

        # Test to make sure connections are set correctly
        msg = 'Connections set incorrectly!'
        connections = [Connection(0, 0, 2), Connection(1, 0, 3), Connection(2, 1, 2), Connection(3, 1, 3)]
        g.set_connections(connections)
        self.assertEqual(g.get_connections(), connections, msg)

        # Make sure weights were compiled correctly
        msg = 'Weights compiled incorrectly!'
        self.assertEqual(g.weights.shape[0], len(g.get_nodes()) - g.shape[0], msg=msg)
        self.assertEqual(g.weights.shape[1], len(g.get_nodes()), msg=msg)
        for i in range(len(g.weights)):
            for j in range(len(g.weights)):
                nid = g._Genome__order[i]
                conn = g.get_connection_from_nodes(j, nid)
                if conn is not None:
                    self.assertEqual(conn.weight, g.weights[i, j], msg)
                else:
                    self.assertEqual(g.weights[i, j], 0.0, msg)

    def test_set_nodes(self):
        g = Genome(1, 1)

        # Test to make sure nodes are set correctly
        msg = 'Nodes set incorrectly!'
        nodes = [Node(0, 'input'), Node(1, 'input'), Node(2, 'output'), Node(3, 'output'), Node(4, 'hidden')]
        g.set_nodes(nodes)
        self.assertEqual(g.get_nodes(), nodes, msg)

        # Make sure weights were compiled correctly
        msg = 'Weights compiled incorrectly!'
        self.assertEqual(g.weights.shape[0], len(nodes) - g.shape[0], msg=msg)
        self.assertEqual(g.weights.shape[1], len(nodes), msg=msg)


class TestNode(unittest.TestCase):
    def test_constructor(self):
        # Test node attribute values
        msg = 'Failed to assign node attributes correctly!'
        n = Node(0, 'input')
        self.assertEqual(n.id, 0, msg)
        self.assertEqual(n.type, 'input', msg)
        self.assertEqual(n.activation, activations.linear, msg)

        n2 = Node(1, 'output')
        self.assertEqual(n2.id, 1, msg)
        self.assertEqual(n2.type, 'output', msg)
        self.assertEqual(n2.activation, activations.modified_sigmoid, msg)

        n3 = Node(2, 'hidden', activation=activations.absolute)
        self.assertEqual(n3.id, 2, msg)
        self.assertEqual(n3.type, 'hidden', msg)
        self.assertEqual(n3.activation, activations.absolute, msg)

    def test_copy(self):
        n = Node(0, 'hidden')
        nc = n.copy()

        # Test to make sure the copies are the same
        msg = 'Copy is not the same as the original!'
        self.assertEqual(n, nc, msg)
        self.assertEqual(n.id, nc.id, msg)
        self.assertEqual(n.type, nc.type, msg)
        self.assertEqual(n.activation, nc.activation, msg)

        # Test to make sure the copy doesn't change if the original does
        msg = 'Copy changed when the original was changed!'
        n.activation = activations.absolute
        self.assertNotEqual(n.activation, nc.activation, msg)

    def test_get_innovation_number(self):
        g = Genome(10, 10)

        msg = 'Innovation number not unique!'

        # Make sure innovation numbers are unique
        inn_num = 0
        for i in range(10):
            for j in range(10, 20):
                g.add_connection(i, j)
                self.assertEqual(g.get_innovation_number(i, j), inn_num, msg)
                inn_num += 1

        # Make sure it's the same twice
        inn_num = 0
        for i in range(10):
            for j in range(10, 20):
                self.assertEqual(g.get_innovation_number(i, j), inn_num, msg)
                inn_num += 1


class TestConnection(unittest.TestCase):
    def test_constructor(self):
        # Test connection attribute values
        msg = 'Failed to assign connection attributes correctly!'
        c = Connection(0, 0, 2)
        self.assertEqual(c.innovation_number, 0, msg)
        self.assertEqual(c.in_node, 0, msg)
        self.assertEqual(c.out_node, 2, msg)
        self.assertTrue(-1.0 <= c.weight <= 1.0, msg)
        self.assertTrue(c.expressed, msg)

        c2 = Connection(1, 57, 9, weight=-0.4, expressed=False)
        self.assertEqual(c2.innovation_number, 1, msg)
        self.assertEqual(c2.in_node, 57, msg)
        self.assertEqual(c2.out_node, 9, msg)
        self.assertEqual(c2.weight, -0.4, msg)
        self.assertFalse(c2.expressed, msg)

    def test_copy(self):
        c = Connection(0, 0, 2)
        cc = c.copy()

        # Test to make sure the copies are the same
        msg = 'Copy is not the same as the original!'
        self.assertEqual(c, cc, msg)
        self.assertEqual(c.innovation_number, cc.innovation_number, msg)
        self.assertEqual(c.in_node, cc.in_node, msg)
        self.assertEqual(c.out_node, cc.out_node, msg)
        self.assertEqual(c.weight, cc.weight, msg)
        self.assertEqual(c.expressed, cc.expressed, msg)

        # Test to make sure the copy doesn't change if the original does
        msg = 'Copy changed when the original was changed!'
        c.weight = 50
        c.disable()
        self.assertNotEqual(c.weight, cc.weight, msg)
        self.assertNotEqual(c.expressed, cc.expressed, msg)

    def test_disable(self):
        # Test to make sure it returns the correct value
        msg = 'Failed to disable connection correctly!'
        c = Connection(0, 0, 2)
        c.disable()
        self.assertFalse(c.expressed, msg)

    def test_enable(self):
        # Test to make sure it returns the correct value
        msg = 'Failed to enable connection correctly'
        c = Connection(0, 0, 2, expressed=False)
        c.enable()
        self.assertTrue(c.expressed, msg)

    def test_set_weight(self):
        # Test to make sure it returns the correct value
        msg = 'Failed to set connection weight correctly!'
        c = Connection(0, 0, 2)
        c.weight = 117
        self.assertEqual(c.weight, 117, msg)

    def test_set_random_weight(self):
        # Test to make sure it returns a random value between -1 and 1
        msg = 'Failed to set connection weight correctly!'
        c = Connection(0, 0, 2, weight=5)
        c.set_random_weight()
        self.assertNotEqual(c.weight, 5, msg)
        self.assertTrue(-1.0 <= c.weight <= 1.0, msg)

    def test_toggle(self):
        # Test to make sure the connection's expression toggles correctly
        msg = 'Failed to toggle connection\'s expression correctly!'
        c = Connection(0, 0, 2)
        c.toggle()
        self.assertFalse(c.expressed, msg)
        c.toggle()
        self.assertTrue(c.expressed, msg)


if __name__ == '__main__':
    unittest.main()
