
import random

from genome import Genome


class Ecosystem:
    """
    An ecosystem is a regulated group of species.  It has a controlled population of genomes separated into
    different species based on closeness of topology.

    Species are regulated once per generation.  Roughly fifty percent of each species is killed after a generation
    is completed.  The other half has an opportunity to breed and create new genomes and possibly new species.

    Attributes:
    __threshold (float):            The threshold of closeness for two genomes to be considered part of the same species
    __disjoint_coefficient (float): A coefficient that adjusts the importance of disjoint connections when determining
                                    genome distance
    __excess_coefficient (float):   A coefficient that adjusts the importance of excess connections when determining
                                    genome distance
    __weight_coefficient (float):   A coefficient that adjusts the importance of the average weight for connections when
                                    determining genome distance
    __inn_num_gen (generator):      A generator that assigns innovation numbers to genomes
    __species (list):               A list of species present in the ecosystem
    """

    def __init__(self, threshold=0.5, disjoint_coefficient=1.0, excess_coefficient=1.0, weight_coefficient=0.4):
        """
        Constructor.

        Parameters:
        threshold (float):            The threshold of closeness for two genomes to be considered part of the same species
        disjoint_coefficient (float): A coefficient that adjusts the importance of disjoint connections when determining
                                      genome distance
        excess_coefficient (float):   A coefficient that adjusts the importance of excess connections when determining
                                      genome distance
        weight_coefficient (float):   A coefficient that adjusts the importance of the average weight for connections when
                                      determining genome distance
        """
        self.__threshold = threshold
        self.__disjoint_coefficient = disjoint_coefficient
        self.__excess_coefficient = excess_coefficient
        self.__weight_coefficient = weight_coefficient
        self.__inn_num_gen = innovation_number_generator()
        self.__inn_num_gen.send(None)
        self.__species = []

    def add_genome(self, genome):
        """
        Adds the given genome to a species in the ecosystem.  If no species exist or the new genome isn't close
        enough to any that do exist, it creates a new species and sets the new genome as its representative.

        Parameters:
        genome (Genome): The genome to add to a species in the ecosystem
        """
        # Look for a species that the genome fits into and add it if found
        for species in self.__species:
            if self.get_distance(genome, species.get_representative()) < self.__threshold:
                species.add(genome)
                return

        # Create a new species if it doesn't fit in any of the others (or if there are none)
        self.__species.append(Species(len(self.__species), genome))

    def adjust_fitness(self, genome):
        """
        Adjusts the fitness of the given genome to implement explicit fitness sharing.

        This regulates the fitness of each species so that no one species takes over the entire population.
        """
        different_count = 0
        for genome2 in self.get_population():
            if self.get_distance(genome, genome2) >= self.__threshold:
                different_count += 1
        if different_count > 0:
            genome.set_fitness(genome.get_fitness() / different_count)

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

        Parameters:
        input_amt (int):  The amount of input nodes for the genome
        output_amt (int): The amount of output nodes for the genome
        """
        genome = Genome(input_amt, output_amt, self.__inn_num_gen)
        self.add_genome(genome)

    def create_initial_population(self, population_size, parent_genome=None, input_length=None, output_length=None, mutate=True):
        """
        Creates an initial population of genomes with the given parent genome or input and output lengths.

        The size of the initial populations is equal to the given population size.

        Parent genomes take precedence over input and output lengths.

        If the ecosystem contains genomes already, they will be replaced.

        Parameters:
        population_size (int):  The amount of genomes in the initial population
        parent_genome (Genome): An optional genome to base the rest of the population on
        input_length (int):     The amount of input nodes for each genome in the population
        output_length (int):    The amount of output nodes for each genome in the population
        mutate (bool):          If True, each newly created genome will be given a random mutation
        """
        # Determine how to create genomes
        if parent_genome is not None:
            from_parent = True
        elif input_length is not None and output_length is not None:
            from_parent = False
        elif input_length is None and output_length is None:
            raise EcosystemError('No input and output length or parent genome specified!')
        elif input_length is not None and output_length is None:
            raise EcosystemError('No output length specified!')
        elif input_length is None and output_length is not None:
            raise EcosystemError('No input length specified!')
        else:
            raise EcosystemError('A problem occurred while creating initial population!')

        # Create the initial population
        self.__species.clear()
        for i in range(population_size):
            if from_parent:
                g = parent_genome.copy()
            else:
                g = Genome(input_length, output_length, self.__inn_num_gen)
            if mutate:
                g.mutate_random()
            self.add_genome(g)

    def cross(self, genome1, genome2):
        """
        Returns a combination of the two given genomes.

        Parameters:
        genome1 (Genome): The first genome to cross
        genome2 (Genome): The second genome to cross

        Returns:
        (Genome): A genome that is a combination of the two given genomes
        """
        # Find the fitter parent
        if genome1.get_fitness() >= genome2.get_fitness():
            more_fit_parent = genome1
            less_fit_parent = genome2
        else:
            more_fit_parent = genome2
            less_fit_parent = genome1

        # Combine the genomes
        child_nodes = []
        child_connections = []

        g1_inn_nums = [c.get_innovation_number() for c in genome1.get_connections()]
        g2_inn_nums = [c.get_innovation_number() for c in genome2.get_connections()]
        all_inn_nums = set(g1_inn_nums + g2_inn_nums)

        for inn_num in all_inn_nums:
            # --- Matching gene for both parents ---
            if inn_num in more_fit_parent.get_connections() and inn_num in less_fit_parent.get_connections():
                rand_choice = random.randint(0, 1)

                # More fit parent
                if rand_choice == 0:
                    conn = more_fit_parent.get_connection(inn_num)

                    if conn.get_in_node() not in [n.get_id() for n in child_nodes]:
                        child_nodes.append(more_fit_parent.get_node(conn.get_in_node()))

                    if conn.get_out_node() not in [n.get_id() for n in child_nodes]:
                        child_nodes.append(more_fit_parent.get_node(conn.get_out_node()))

                # Less fit parent
                else:
                    conn = less_fit_parent.get_connection(inn_num)

                    if conn.get_in_node() not in [n.get_id() for n in child_nodes]:
                        child_nodes.append(less_fit_parent.get_node(conn.get_in_node()))

                    if conn.get_out_node() not in [n.get_id() for n in child_nodes]:
                        child_nodes.append(less_fit_parent.get_node(conn.get_out_node()))

                child_connections.append(conn)

            # --- Disjoint or excess gene for fit parent ---
            elif inn_num in more_fit_parent.get_connections() and inn_num not in less_fit_parent.get_connections():
                conn = more_fit_parent.get_connection(inn_num)

                if conn.get_in_node() not in [n.get_id() for n in child_nodes]:
                    child_nodes.append(more_fit_parent.get_node(conn.get_in_node()))

                if conn.get_out_node() not in [n.get_id() for n in child_nodes]:
                    child_nodes.append(more_fit_parent.get_node(conn.get_out_node()))

                child_connections.append(conn)

        # Sort the nodes and connections
        child_nodes.sort(key=lambda node: node.get_id())
        child_connections.sort(key=lambda c: c.get_innovation_number())

        # Create the child
        child = Genome(0, 0, self.__inn_num_gen)
        child.set_nodes(child_nodes)
        child.set_connections(child_connections)

        return child

    def get_distance(self, genome1, genome2):
        # Find the amount of disjoint and excess connections
        g1_inn_nums = [c.get_innovation_number() for c in genome1.get_connections()]
        g2_inn_nums = [c.get_innovation_number() for c in genome2.get_connections()]
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
                weight_difference_sum += abs(genome1.get_connection(inn_num).get_weight() - genome2.get_connection(inn_num).get_weight())
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
        distance = (disjoint_count * self.__disjoint_coefficient / max_connections) if max_connections > 0 else 0.0
        distance += (excess_count * self.__excess_coefficient / max_connections) if max_connections > 0 else 0.0
        distance += (average_weight * self.__weight_coefficient)
        return distance

    def get_population(self):
        """
        Returns a list containing every genome in the population.

        Returns:
        (list): Every genome in the population
        """
        population = []
        for s in self.__species:
            population += s.get_genomes()
        return population

    def get_species(self): return self.__species


class Species:
    """
    A list of genomes that have close enough topologies to be considered part of the same species.

    When a species is first created, the first member of the species becomes the representative.
    All other genomes are compared to the representative before they can enter into the species and
    only if they are within the threshold.

    Attributes:
    __id_num (int):            The unique id for the species
    __representative (Genome): The first genome in the species
    __genomes (list):          The list of genomes that are a part of the species
    """

    def __init__(self, id_num, representative):
        """
        Constructor.

        Parameters:
        id_num (int):            The unique id for the species
        representative (Genome): The first genome in the species
        genomes (list):          The list of genomes that are a part of the species
        """
        self.__id_num = id_num
        self.__representative = representative
        self.__genomes = []
        self.add(representative)

    def __getitem__(self, genome_index): return self.__genomes[genome_index]

    def __len__(self): return len(self.__genomes)

    def __setitem__(self, key, value): self.__genomes[key] = value

    def add(self, genome): self.__genomes.append(genome)

    def get_genomes(self): return self.__genomes

    def get_id(self): return self.__id_num

    def get_representative(self): return self.__representative

    def pop(self, genome): return self.__genomes.pop(genome)

    def remove(self, genome): self.__genomes.remove(genome)


def innovation_number_generator():
    """
    A generator that given the in node and out node of a connection as a tuple through the send function,
    yields a unique innovation number for that connection.  If the connection already exists within the
    ecosystem, the existing innovation number is yielded.

    The generator function send(None) must be called after declaration.
    """
    inn_log = []
    inn_num = 0
    while True:
        conn = yield inn_num
        if conn in inn_log:
            inn_num = inn_log.index(conn)
        else:
            inn_log.append(conn)
            inn_num = len(inn_log) - 1


class EcosystemError(Exception):
    def __init__(self, message):
        self.message = message
