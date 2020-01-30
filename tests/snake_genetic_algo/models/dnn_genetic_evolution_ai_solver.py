import random
import copy
import numpy as np
from statistics import mean
from game.models.base_game_model import BaseGameModel
from tf_models.dnn_model import DeepNeuralNetModel
from game.helpers.constants import Constants

"""
import my lib """
from "mypath" import DNNGeneticEvolutionTrainer as dnn



class DNNGeneticEvolutionSolver(BaseGameModel):

    model = None

    def __init__(self):
        BaseGameModel.__init__(self, "Deep Neural Net GE", "deep_neural_net_genetic_evolution", "dnnge")

    def move(self, environment):
        if self.model is None:
            self.model = DeepNeuralNetModel(Constants.MODEL_DIRECTORY + "/dnn_genetic_evolution/")
        BaseGameModel.move(self, environment)
        predicted_action = self._predict(environment, self.model)
        return predicted_action

""" here import my ai and use it to solve this """

class DNNGeneticEvolutionTrainer_(BaseGameModel) :
    BaseGameModel.__init__(self, "Deep Neural Net GE", "deep_neural_net_genetic_evolution_trainer", "dnnget")

    generation = 0
    selection_rate = 0.1
    mutation_rate = 0.01
    population_size = 1000
    parents = int(population_size * selection_rate)
    model = None

    def move(self, environment):
        if self.model is None:
            self.model = DeepNeuralNetModel(Constants.MODEL_DIRECTORY + "dnn_genetic_evolution/")
        BaseGameModel.move(self, environment)
        self._genetic_evolution()

    def _genetic_evolution(self) :
        lib = dnn(self.generation, self.selection_rate, self.mutation_rate, self.population_size, self.parents, self.model)
        population = None
        while True:
            population_size = len(population) if population is not None else lib.population_size
            print "generation: " + str(lib.generation) + ", population: " + str(population_size) + ", mutation_rate: " + str(lib.mutation_rate)

            # 1. Selection
            parents = lib._strongest_parents(population)

            lib._save_model(parents)  # Saving main model based on the current best two chromosomes

            # 2. Crossover (Roulette selection)
            pairs = []
            while len(pairs) != lib.population_size:
                pairs.append(lib._pair(parents))

            # # 2. Crossover (Rank selection)
            # pairs = lib._combinations(parents)
            # random.shuffle(pairs)
            # pairs = pairs[:self.population_size]

            base_offsprings = []
            for pair in pairs:
                offsprings = lib._crossover(pair[0][0], pair[1][0])
                base_offsprings.append(offsprings[-1])

            # 3. Mutation
            new_population = lib._mutation(base_offsprings)
            population = new_population
            lib.generation += 1
        
