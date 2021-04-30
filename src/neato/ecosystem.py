"""
Neato

MIT License

Copyright (c) 2021 Joseph Pace

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import random

from .genome import Genome


class Ecosystem:
    """
    An ecosystem is a regulated group of species.  It has a controlled population of genomes separated into
    different species based on closeness of topology.

    Species are regulated once per generation.  Roughly fifty percent of each species is killed after a generation
    is completed.  The other half has an opportunity to breed and create new genomes and possibly new species.

    Attributes:
        __connection_log (dict): A dictionary of existing connections within the ecosystem used to assign unique
                                     innovation numbers.
        dc (float):              The disjoint coefficient: adjusts the importance of disjoint connections when determining
                                     genome distance.
        ec (float):              The excess coefficient: adjusts the importance of excess connections when determining 
                                     genome distance.
        generation (int):        The current generation number.
        species (list):          A list of species present in the ecosystem.
        threshold (float):       The threshold of closeness for two genomes to be considered part of the same species.
        wc (float):              The weight coefficient: adjusts the importance of the average weight for connections when
                                     determining genome distance.
    """

    def __init__(self, threshold=0.5, dc=1.0, ec=1.0, wc=0.4):
        """
        Constructor.

        Args:
            threshold (float): The threshold of closeness for two genomes to be considered part of the same species.
            dc (float):        The disjoint coefficient: adjusts the importance of disjoint connections when determining
                                   genome distance.
            ec (float):        The excess coefficient: adjusts the importance of excess connections when determining
                                   genome distance.
            wc (float):        The weight coefficient: adjusts the importance of the average weight for connections when
                                   determining genome distance.
        """
        self.threshold = threshold
        self.dc = dc
        self.ec = ec
        self.wc = wc

        self.__connection_log = {}
        self.species = []
        self.generation = 1

    def __str__(self):
        return f'Generation: {self.generation}  Population: {len(self.get_population())}  Species: {len(self.species)}  Best Fitness: {self.get_best_genome().fitness}'

    def add_genome(self, genome):
        """
        Adds the given genome to a species in the ecosystem.  If no species exist or the new genome isn't close
        enough to any that do exist, it creates a new species and sets the new genome as its representative.

        Args:
            genome (Genome): The genome to add to a species in the ecosystem.
        """
        genome.ecosystem = self
        # Look for a species that the genome fits into and add it if found
        for species in self.species:
            if self.get_distance(genome, species.representative) < self.threshold:
                species.append(genome)
                return

        # Create a new species if it doesn't fit in any of the others (or if there are none)
        self.species.append(Species(len(self.species), genome))

    def adjust_fitness(self, genome):
        """
        Adjusts the fitness of the given genome to implement explicit fitness sharing.

        This regulates the fitness of each species so that no one species takes over the entire population.
        """
        different_count = 0
        for genome2 in self.get_population():
            if self.get_distance(genome, genome2) >= self.threshold:
                different_count += 1
        if different_count > 0:
            genome.fitness = genome.fitness / different_count

    def adjust_population_fitness(self):
        """
        Adjusts the fitness of the population to implement explicit fitness sharing.

        This regulates the fitness of each species so that no one species takes over the entire population.
        """
        for genome in self.get_population():
            self.adjust_fitness(genome)

    def create_genome(self, input_amt, output_amt):
        """
        Creates a genome and adds it to a species.

        Args:
            input_amt (int):  The amount of input nodes for the genome.
            output_amt (int): The amount of output nodes for the genome.
        """
        genome = Genome(input_amt, output_amt, ecosystem=self)
        self.add_genome(genome)

    def create_initial_population(self, population_size, parent_genome=None, input_size=None, output_size=None, mutate=True):
        """
        Creates an initial population of genomes with the given parent genome or input and output lengths.

        The size of the initial populations is equal to the given population size.

        Parent genomes take precedence over input and output lengths.

        If the ecosystem contains genomes already, they will be replaced.

        Args:
            population_size (int):  The amount of genomes in the initial population.
            parent_genome (Genome): An optional genome to base the rest of the population on.
            input_size (int):       The amount of input nodes for each genome in the population.
            output_size (int):      The amount of output nodes for each genome in the population.
            mutate (bool):          If True, each newly created genome will be given a random mutation.
        """
        # Filter arguments
        assert parent_genome is not None or input_size is not None and output_size is not None, 'No parent genome or input and output sizes specified!'
        if parent_genome is not None:
            from_parent = True
        else:
            assert input_size is not None, 'No input size specified!'
            assert output_size is not None, 'No output size specified!'
            from_parent = False

        # Create the initial population
        self.species.clear()
        for i in range(population_size):
            if from_parent:
                g = parent_genome.copy()
                g.ecosystem = self
            else:
                g = Genome(input_size, output_size, ecosystem=self)
            if mutate:
                g.mutate_random()
            self.add_genome(g)

    def cross(self, genome1, genome2):
        """
        Returns a combination of the two given genomes.

        Args:
            genome1 (Genome): The first genome to cross.
            genome2 (Genome): The second genome to cross.

        Returns:
            (Genome): A genome that is a combination of the two given genomes.
        """
        # Find the fitter parent
        if genome1.fitness >= genome2.fitness:
            more_fit_parent = genome1
            less_fit_parent = genome2
        else:
            more_fit_parent = genome2
            less_fit_parent = genome1

        # Combine the genomes
        child_connections = []

        g1_inn_nums = [c.innovation_number for c in genome1.get_connections()]
        g2_inn_nums = [c.innovation_number for c in genome2.get_connections()]
        all_inn_nums = set(g1_inn_nums + g2_inn_nums)

        for inn_num in all_inn_nums:
            # Note: Genes that are in less fit parent and not in more fit parent are intentionally skipped
            if inn_num in more_fit_parent.get_connections() and inn_num in less_fit_parent.get_connections():
                # Matching gene for both parents
                parent_choice = random.choice((more_fit_parent, less_fit_parent))
                child_connections.append(parent_choice.get_connection(inn_num))
            elif inn_num in more_fit_parent.get_connections() and inn_num not in less_fit_parent.get_connections():
                # Disjoint or excess gene for fit parent
                conn = more_fit_parent.get_connection(inn_num)
                child_connections.append(conn)

        # Sort the connections
        child_connections.sort(key=lambda c: c.innovation_number)

        # Get hidden nodes that are used in connections
        conn_hidden_nids = set()
        for conn in child_connections:
            conn_hidden_nids.add(conn.in_node)
            conn_hidden_nids.add(conn.out_node)

        # Copy over fitter parent nodes
        child_nodes = []
        for node in more_fit_parent.get_nodes():
            # Unused hidden nodes are excluded
            if node.type == 'hidden' and node.id not in conn_hidden_nids:
                continue
            child_nodes.append(node)

        # Create the child
        child = Genome(0, 0, ecosystem=self)
        child.set_nodes(child_nodes)
        child.set_connections(child_connections)

        return child

    def get_best_genome(self):
        """
        Returns the fittest genome of the current generation.

        Returns:
            (Genome): The fittest genome of the current generation.
        """
        genomes_by_fitness = sorted(self.get_population(), key=lambda g: g.fitness, reverse=True)
        return genomes_by_fitness[0]

    def get_distance(self, genome1, genome2):
        """
        Returns the the distance between two genomes, meaning a score for how different they are from eachother.

        Args:
            genome1 (Genome): The first genome.
            genome2 (Genome): The second genome.

        Returns:
            (float): The distance between the two genomes.
        """
        # Find the amount of disjoint and excess connections
        g1_inn_nums = [c.innovation_number for c in genome1.get_connections()]
        g2_inn_nums = [c.innovation_number for c in genome2.get_connections()]
        g1_max_inn_num = max(g1_inn_nums) if len(g1_inn_nums) > 0 else 0
        g2_max_inn_num = max(g2_inn_nums) if len(g2_inn_nums) > 0 else 0
        all_inn_nums = set(g1_inn_nums + g2_inn_nums)

        matching_count = 0
        disjoint_count = 0
        excess_count = 0
        weight_difference_sum = 0.0

        for inn_num in all_inn_nums:
            if inn_num in genome1.get_connections() and inn_num in genome2.get_connections():
                matching_count += 1
                weight_difference_sum += abs(genome1.get_connection(inn_num).weight - genome2.get_connection(inn_num).weight)
            elif inn_num in genome1.get_connections() and inn_num not in genome2.get_connections():
                if inn_num < g2_max_inn_num:
                    disjoint_count += 1
                elif inn_num > g2_max_inn_num:
                    excess_count += 1
            elif inn_num not in genome1.get_connections() and inn_num in genome2.get_connections():
                if inn_num < g1_max_inn_num:
                    disjoint_count += 1
                elif inn_num > g1_max_inn_num:
                    excess_count += 1

        # Find the average weight difference for matching genes
        average_weight = weight_difference_sum / matching_count if matching_count > 0 else 0.0

        # Find the max connections
        max_connections = max(len(genome1.get_connections()), len(genome2.get_connections()))

        # Find and return the distance
        distance = (disjoint_count * self.dc / max_connections) if max_connections > 0 else 0.0
        distance += (excess_count * self.ec / max_connections) if max_connections > 0 else 0.0
        distance += (average_weight * self.wc)
        return distance

    def get_innovation_number(self, in_node, out_node):
        """
        Returns the innovation number for a connection with the given input and output.

        Args:
            in_node (int):  The id of the input node for the connection.
            out_node (int): The id of the output node for the connection.

        Returns:
            (int): The innovation number of an existing, matching connection or a new one if no matching connection exists.
        """
        conn = (in_node, out_node)
        if conn in self.__connection_log:
            return self.__connection_log[conn]
        else:
            inn_num = len(self.__connection_log)
            self.__connection_log[conn] = inn_num
            return inn_num

    def get_population(self):
        """
        Returns a list containing every genome in the population.

        Returns:
            (list): Every genome in the population.
        """
        population = []
        for s in self.species:
            population += s
        return population

    def kill(self, genome):
        """
        Removes the genome from the gene pool.

        Args:
            genome (Genome): The genome to kill.
        """
        for species in self.species:
            if genome in species:
                species.remove(genome)

                # Remove the species if it contains no genomes
                if len(species) == 0:
                    self.species.remove(species)

                break

    def kill_percentage(self, percentage):
        """
        Kills the given percentage of the members of each species (approximately).

        Genomes are killed in order of fitness (lowest to highest)

        Will round down if inexact.

        Args:
            percentage (float): The maximum percentage of genomes that will be killed in each species.

        Returns:
            (int): The total amount of genomes that were killed.
        """
        total_killed = 0
        for species in self.species:
            species.sort(key=lambda g: g.fitness)
            amt_killed = 0
            start_amt = len(species)
            while (amt_killed / start_amt) * 100 <= percentage - (1 / start_amt) * 100:
                species.pop(0)
                amt_killed += 1
                total_killed += 1
        return total_killed

    def next_generation(self, kill_percentage=50.0, mutate=True, parent_genome=None):
        """
        Proceeds to the next generation of genomes by doing the following:

        - Adjusts the fitness of the population
        - Kills the given percentage of the population
        - Crosses the fittest genomes for each species together to create new genomes (equal to the amount that were killed
        - Mutates the population (if told to)
        - Increments the generation number

        Args:
            kill_percentage (float): The percentage of genomes to remove from the gene pool.
            mutate (bool):           Determines if the population should be mutated or not.
            parent_genome (Genome):  The optional parent of the next generation.
        """
        if parent_genome is None:
            # Adjust the population's fitness
            self.adjust_population_fitness()

            # Kill a percentage of the population and get the amount to replace
            amt_to_replace = self.kill_percentage(kill_percentage)

            # Create children from the fittest genomes to replace the genomes that were killed
            pop_by_fitness = sorted(self.get_population(), key=lambda g: g.fitness, reverse=True)
            genome_index = 1
            while amt_to_replace > 0:
                child = self.cross(pop_by_fitness[genome_index], pop_by_fitness[genome_index + 1])
                self.add_genome(child)
                if genome_index >= len(pop_by_fitness) - 2:
                    genome_index = 0
                else:
                    genome_index += 1
                amt_to_replace -= 1
        else:
            parent_genome.fitness = 0.0
            self.create_initial_population(len(self.get_population()), parent_genome=parent_genome)

        # Mutate the population
        if mutate:
            for genome in self.get_population():
                genome.mutate_random()

        # Increment the generation number
        self.generation += 1


class Species(list):
    """
    A list of genomes that have close enough topologies to be considered part of the same species.

    When a species is first created, the first member of the species becomes the representative.
    All other genomes are compared to the representative before they can enter into the species and
    only if they are within the threshold.

    Attributes:
        id (int):                The unique id for the species.
        representative (Genome): The first genome in the species.
    """

    def __init__(self, id_num, representative):
        """
        Constructor.

        Args:
            id_num (int):            The unique id for the species.
            representative (Genome): The first genome in the species.
            genomes (list):          The list of genomes that are a part of the species.
        """
        super(Species, self).__init__([representative])
        self.id = id_num
        self.representative = representative

    def __str__(self): return f'Species: {self.id}, Genomes: {len(self)}'
