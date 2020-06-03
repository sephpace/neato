
import copy
import random
import unittest

from ecosystem import Ecosystem, Species
from genome import Genome


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

        e = Ecosystem(threshold=0.5, dc=1.0, ec=1.0, wc=0.4)

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

        g4 = copy.copy(g3)
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

        e = Ecosystem(threshold=0.5, dc=1.0, ec=1.0, wc=0.4)

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

        g4 = copy.copy(g3)
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
        msg = 'Invalid parameters not caught!'
        with self.assertRaises(AssertionError, msg=msg):
            e.create_initial_population(population_size)

        with self.assertRaises(AssertionError, msg=msg):
            e.create_initial_population(population_size, output_size=3)

        with self.assertRaises(AssertionError, msg=msg):
            e.create_initial_population(population_size, input_size=3)

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

        e = Ecosystem(dc=1.0, ec=1.0, wc=0.4)

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
        e = Ecosystem(dc=1.0, ec=1.0, wc=0.4)

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
