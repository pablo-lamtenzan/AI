
""" actually working with 
https://towardsdatascience.com/continuous-genetic-algorithm-from-scratch-with-python-ff29deedd099

"""

# Conitious = use of floating numbers or integer instead of binary
# This algo comes from the MIT, thaks to Jhon H.Holland for this good shit

import numpy as np
from numpy.random import randint
from random import random as rnd
from random import gauss, randrange

class CGenetic() :
    def __init__(self) :
        self.init = None

    # init population

    # create and individual with random range of genes values
    def individual(self, nb_genes, upper_limit, lower_limit) :
        individual = [round(rnd() * (upper_limit - lower_limit) + lower_limit, 1)
        for x in range(nb_genes)]
        return individual
    
    # create a population using individual()
    def population(self, nb_individuals, nb_genes, upper_limit, lower_limit) :
        return [self.individual(nb_genes, upper_limit, lower_limit)
        for x in range(nb_individuals)]

    # fitness calc
    def fitness_calc(self, individual) :
        fitness_value = sum(individual)
        return fitness_value

    # selection
    """ there exist a few selection methods:
    - Roullette wheel selection : use probalities calculated by fitness performance
    - Fitness half selection : only the top is selected
    - Random selection : just, random """

    # roulette method -> get a value for each ind [0 ; 1]
    def roulette(self, cum_sum, chance) :
        v = list(cum_sum.copy())
        v.append(chance)
        v = sorted(v)
        return v.index(chance)

    # generic selection fct with method selection
    def selection(self, generation, method = 'Fittest Half') :
        generation['Normalized Fitness'] = \
            sorted([generation['Fitness'][x] / sum(generation['Fitness']) 
            for x in range(len(generation['Fitness']))], reverse = True)
        generation['Cumulative Sum'] = np.array(generation['Normalized Fitness']).cumsum()

        # roullete
        if method == 'Roulette Wheel' :
            selected = []
            for x in range(len(generation['Individuals']) // 2) :
                selected.append(self.roulette(generation['Cumulative Sum'], rnd()))
                while len(set(selected)) != len(selected) :
                    selected[x] = \
                        (self.roulette(generation['Cumulative Sum'], rnd()))
                    selected = {'Individuals' : 
                    [generation['Individuals'][int(selected[x])]
                    for x in range(len(generation['Individuals']) // 2)]
                    ,'Fitness': [generation['Fitness'][int(selected[x])]
                    for x in range(
                        len(generation['Individuals']) // 2)]}

        # fittest halt 
        elif method == 'Fittest Half' :
            selected_individuals = [generation['Individuals'][-x - 1]
                for x in range(int(len(generation['Individuals']) // 2))]
            selected_fitnesses = [generation['Fitness'][-x - 1]
                for x in range(int(len(generation['Individuals']) // 2))]
            selected = {'Individuals' : selected_individuals,
                    'Fitness' : selected_fitnesses}

        # random
        elif method == 'Random' :
            selected_individuals = \
                [generation['Individuals']
                    [randint(1, len(generation['Fitness']))]
                for x in range(int(len(generation['Individuals']) // 2))]
            selected_fitnesses = [generation['Fitness'][-x - 1]
                for x in range(int(len(generation['Individuals']) // 2))]
            selected = {'Individuals': selected_individuals,
                    'Fitness': selected_fitnesses}
        return selected

    # paring
    """ there exist a few paring 2 by 2 methods :
        - Fittest -> using the fitness rank
        - Random
        - Weighted random - > probalities (higger when both are fit) """

    def pairing(self, elit, selected, method = 'Fittest') :
        individuals = [elit['Individuals']]+selected['Individuals']
        fitness = [elit['Fitness']]+selected['Fitness']

        # fittest
        if method == 'Fittest' :
            parents = [[individuals[x],individuals[x + 1]] 
                       for x in range(len(individuals) // 2)]

        # random
        if method == 'Random' :
            parents = []

            for x in range(len(individuals) // 2) :
                parents.append(
                    [individuals[randint(0, (len(individuals) - 1))],
                    individuals[randint(0, (len(individuals) - 1))]])
                while parents[x][0] == parents[x][1] :
                    parents[x][1] = individuals[randint(0,(len(individuals)-1))]

        # weighted rand
        if method == 'Weighted Random' :
            normalized_fitness = sorted([fitness[x] / sum(fitness) 
                for x in range(len(individuals) // 2)], reverse = True)
            cummulitive_sum = np.array(normalized_fitness).cumsum()
            parents = []

            for x in range(len(individuals) // 2) :
                parents.append(
                    [individuals[self.roulette(cummulitive_sum,rnd())],
                    individuals[self.roulette(cummulitive_sum,rnd())]])
                while parents[x][0] == parents[x][1]:
                    parents[x][1] = individuals[self.roulette(cummulitive_sum, rnd())]

        return parents

    # Matting (Crossover)
    """ there exist a few mating methods:
        - Single point : in geneX[4] and geneY[4] -> geneX[0] = geneY[0]
        - Two point : genes between 2 points are replaced
    """

    def mating(self, parents, method = 'Single Point') :

        # single point
        if method == 'Single Point':
            pivot_point = randint(1, len(parents[0]))
            offsprings = [parents[0] \
                [0 : pivot_point] + parents[1][pivot_point :]]
            offsprings.append(parents[1][0:pivot_point] + parents[0][pivot_point:])

        # two points
        if method == 'Two Pionts':
            pivot_point_1 = randint(1, len(parents[0] - 1))
            pivot_point_2 = randint(1, len(parents[0]))

            while pivot_point_2 < pivot_point_1:
                pivot_point_2 = randint(1, len(parents[0]))

            offsprings = [parents[0][0 : pivot_point_1] + parents[1][pivot_point_1 : pivot_point_2] +
                [parents[0][pivot_point_2 :]]]

            offsprings.append([parents[1][0 : pivot_point_1] +
                parents[0][pivot_point_1 : pivot_point_2] +
                [parents[1][pivot_point_2 :]]])

        return offsprings

         
    # Mutations
    """
    Gauss -> gene replaced by a nb in to Gauss distribution
    Reset -> random """

    def mutation(self, individual, upper_limit, lower_limit, muatation_rate = 2, 
    method = 'Reset', standard_deviation = 0.001) :
        
        gene = [randint(0, 7)]
        for x in range(muatation_rate - 1):
            gene.append(randint(0, 7))
            while len(set(gene)) < len(gene):
                gene[x] = randint(0, 7)
        mutated_individual = individual.copy()

        # Gauss method
        if method == 'Gauss':
            for x in range(muatation_rate):
                mutated_individual[x] = \
                round(individual[x] + gauss(0, standard_deviation), 1)

        # Reset
        if method == 'Reset':
            for x in range(muatation_rate):
                mutated_individual[x] = round(rnd()* \
                    (upper_limit-lower_limit)+lower_limit, 1)

        return mutated_individual

    # Next generation using class fcts
    def next_generation(self, gen, upper_limit, lower_limit) :
        elit = {}
        next_gen = {}
        elit['Individuals'] = gen['Individuals'].pop(-1)
        elit['Fitness'] = gen['Fitness'].pop(-1)

        selected = self.selection(gen)
        parents = self.pairing(elit, selected)

        # get ADN of 2 offsprings
        offsprings = [[[self.mating(parents[x])
                    for x in range(len(parents))]
                    [y][z] for z in range(2)] 
                    for y in range(len(parents))]
        offsprings1 = [offsprings[x][0]
                    for x in range(len(parents))]
        offsprings2 = [offsprings[x][1]
                    for x in range(len(parents))]

        # manage mutation for give genetic diversity
        unmutated = selected['Individuals'] + offsprings1 + offsprings2
        mutated = [self.mutation(unmutated[x], upper_limit, lower_limit)
            for x in range(len(gen['Individuals']))]
        
        # sorting
        unsorted_individuals = mutated + [elit['Individuals']]
        unsorted_next_gen = \
            [self.fitness_calc(mutated[x]) 
            for x in range(len(mutated))]
        unsorted_fitness = [unsorted_next_gen[x]
            for x in range(len(gen['Fitness']))] + [elit['Fitness']]
        sorted_next_gen = \
            sorted([[unsorted_individuals[x], unsorted_fitness[x]]
                for x in range(len(unsorted_individuals))], 
                    key = lambda x : x[1])
        
        # get next gen
        next_gen['Individuals'] = [sorted_next_gen[x][0]
            for x in range(len(sorted_next_gen))]
        next_gen['Fitness'] = [sorted_next_gen[x][1]
            for x in range(len(sorted_next_gen))]
        gen['Individuals'].append(elit['Individuals'])
        gen['Fitness'].append(elit['Fitness'])

        return next_gen

    # Termination Criteria (When algo stops making generations)
    """
    kinds :
    - Maximum fitness : if an individual in current gen satisfied a fitness
    - Maximun average fitness : same but using more than 1 individual
    - Max nb of gen
    - Maximum silimilar fitness nb : limmit the times 1 individual can be the best of his gen """

    #imp of 1st

    def fitness_similarity_chech(self, max_fitness, number_of_similarity) :
        result = False
        similarity = 0
        for n in range(len(max_fitness) - 1) :
            if max_fitness[n] == max_fitness[n + 1] :
                similarity += 1
            else:
                similarity = 0
        if similarity == number_of_similarity - 1 :
            result = True
        return result

    # Creating the First Generation
    def first_generation(self, pop):
        fitness = [self.fitness_calculation(pop[x]) 
            for x in range(len(pop))]
        sorted_fitness = sorted([[pop[x], fitness[x]]
            for x in range(len(pop))], key = lambda x : x[1])
        population = [sorted_fitness[x][0] 
            for x in range(len(sorted_fitness))]
        fitness = [sorted_fitness[x][1] 
            for x in range(len(sorted_fitness))]
        return {'Individuals': population, 'Fitness': sorted(fitness)}






    """ TESTER """
    def test(self) :

        # Generations and fitness values will be written to this file
        Result_file = 'GA_Results.txt'

        pop = self.population(20, 8, 1, 0)
        gen = []
        gen.append(self.first_generation(pop))
        fitness_avg = np.array([sum(gen[0]['Fitness'])/
                            len(gen[0]['Fitness'])])
        fitness_max = np.array([max(gen[0]['Fitness'])])
        res = open(Result_file, 'a')
        res.write('\n'+str(gen)+'\n')
        res.close()
        finish = False
        while finish == False:
            if max(fitness_max) > 6:
                break
            if max(fitness_avg) > 5:
                break
            if self.fitness_similarity_chech(fitness_max, 50) == True:
                break
            gen.append(self.next_generation(gen[-1], 1, 0))
            fitness_avg = np.append(fitness_avg, sum(
                gen[-1]['Fitness']) / len(gen[-1]['Fitness']))
            fitness_max = np.append(fitness_max, max(gen[-1]['Fitness']))
            res = open(Result_file, 'a')
            res.write('\n'+str(gen[-1])+'\n')
            res.close()