
# Deep Neural Network Genetic Evolution Train

import random
import copy
import numpy as np

class DNNGeneticEvolutionTrainer : """inherit_something"""
    def __init__(self, generartion, selection_rate, mutation_rate, poopulation_size, parents, model) :
        self.generartion = generartion
        self.selection_rate = selection_rate
        self.mutation_rate = mutation_rate
        self.poopulation_size = poopulation_size
        self.parents = parents
        self.model = model
    
    # a choromosome is a tensor, we inint it with random values
    def random_chromosome(self, size) :
        chromosome []
        for i in range(0, size) :
            random_value = random.uniform(-1, 1)
            chromosome.append(random_value)
        return (chromosome)

    """ model_features_count is the nb o features that our model has
        for the example of snake game our nb of featues is 5:
        action vector, left, right, top declacememt posible an dangle for fruit """

    # just init population with some caos (rand nb)
    def initial_population(self, model_features_count) :
        chromosomes = []
        for i in range(self.population_size) :
            chromosome = []
            for j in range(0, model_features_count) : 
                chromosome.append(self.random_chromosome(self.model.hidden_node_neurons))
            chromosomes.append(chromosome)
        return chromosomes

    # more geneticaly apt they are more probalities they have to be returned
    def roulette_selection(self, parents, pick) :
        current = 0
        for parent in parents :
            current += parent[1]
            if current > pick :
                return parent

    # function run the reproduction systhem fct between the most geneticaly apt
    def gen_pair(self, parents) :
        total_parents_score = sum([x[1] for in parents])
        pick = random.uniform(0, total_parents_score) # pick = random between 0 and total score
        return [self.roulette_selection(parents, pick), self.roulette_selection(parents, pick)]

    # combines the genetic 
    def combinations(self, parents) :
        combinations = []
        for i in range(0, len(parents)) :
            for j in range(i, len(parents)) :
                combinations.append((parents[i], parents[j]))
        return combinations

    # to have diversity we need some random mutations in our offpring
    def mutation(self, base_offsprings, model_features_count) : """mfc has to be 5 see up"""
        offspring = []
        for offspring in base_offsprings :
            offspring_mutation = copy.deepcopy(offspring) # copy operation on arbitrary python obj
            for i in range(0, model_featues_count) :
                for j in range(0, self.model.hidden_node_neurons) :
                    if np.random.choice([True, False], p=[self.mutation_rate, 1 - self.mutation_rate]) :
                        offspring_mutation[i][j] = random.uniform(-1, 1)
            offspring.append(offspring_mutation)
            return offspring

    # swaping genetic components for mixt gens
    def crossover(self, x, y, model_features_count) :
        offspring_x = x
        offspring_y = y
        for i in range(0, model_features_count) :
            for j in range(0, self.model.hidden_node_neurons) :
                if (random.choice([True, False])) :
                    offspring_x[i][j] = y[i][j]
                    offspring_y[i][j] = x[i][j]
        return (offspring_x, offspring_y)

    # return the top into the actual generation 
    def strongest_parents(self, population, rewards) :
        if population is None :
            population = self.initial_population(model_features_count) # meaning of 5 explained up
        scores_for_chromosomes = []
        for i in range(0, len(population)) :
            chromosome = population[i]
            scores_for_chromosomes.append((chromosome, rewards)) """ rewards must be gameplay_for_chromosomes in this example"""
            
            # display
            if i == len(population) - 1 :
                print "\r"+"\033[K"+"\r",
            else :
                print "/r" + str(i + 1) + " out of " + str(len(population)),
            
        # use scores to get to top geneticaly apt
        scores_for_chromosomes.sort(key=lambda x: x[1])
        print "\r" + str(mean([x[1] for x in scores_for_chromosomes]))
        top_performers = scores_for_chromosomes[-self.parents:]

        # more display
        op_scores = [x[1] for x in top_performers]
        print "top " + str(self.selection_rate) + ": " + "(min: " + str(min(top_scores)) +
        ", avg: " + str(mean(top_scores)) + ", max: " + str(max(top_scores)) + ")"
        print ""
        return (top_performers)

    # used to store data during the genetic algorithm
    def save_model(self, parents) :
        x = copy.deepcopy(parents[-1][0])
        y = copy.deepcopy(parents[-2][0])
        best_offprings = self.crossover(x, y)
        self.model.set_weights(best_offprings[-1])
        self.model.save()

    """ next function is a specific fct for a snake, it return the actual reward 
        so the performance, logic is for snake but we always a fction like this
        
        need the inherit from others classes the functions i need"""

    def gameplay_for_chromosomes(self, chromosome) :
        self.model.set_weights(chromosome) :
        environment = self.preprare_training_envirnement()
        while True :
            predicted_action = self.predict(envirnement, self.model)
            environment.eat_fruit_if_posible()
            if not environment.step(predicted_action) :
                return environment.reward()
            if environment.is_in_fruit_cycle() :
                return 0
    # main genetic fct
    def Genetic_Evolution(self) :
        population = None
        while True :
            # init size of population
            population_size = len(population) if population is not None else self.population_size
            print "generation: " + str(self.generation) + ", population: " + str(population_size) +
            ", mutation_rate: " + str(self.mutation_rate)

            # 1. Selection:
            parents = self.strongest_parents(parents)

            self.save_model(parents) # save model based on the current best 2 chromosomes

            # 2. Crossover (Roulette selection):
            pairs = []
            while len(pairs) != self.population_size :
                pairs.append(self.pair(parents))

            # 2. Crossover (Rank selection):
            # pairs = self.combinations(parents)
            # random.shuffle(pairs)
            # pairs = pairs[:self.population_size]

            base_offsprings = []
            for pair in pairs :
                offsprings = self.crossover(pair[0][0], pair[1][0])
                base_offsprings.append(offsprings[-1])

            # 3. Mutation
            new_population = self.mutation(base_offsprings)
            population = new_population

            self.generartion += 1
