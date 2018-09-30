'''
Created on Sep 29, 2018

@author: silvan
'''
import random
import numpy as np

class Solver(object):
    '''
    classdocs
    '''


    def __init__(self, nonogram):
        self.puzzleName = nonogram.puzzle
        self.puzzle = nonogram.puzzle
        self.nonogram = nonogram

        self.populationSize = 100
        self.offspringSize = 400
        self.recombRateRow = 0.1
        self.mutationRate = 0.1
        self.eliteSize = 5
        
    def run(self):
        # initialisation
        population = self.initialisePopulation()
        populationFitness = self.evaluate(population)
        
        while np.min(populationFitness) != 0:
            # parent selection
            matingPool = self.selectParents(population)
            
            # recombination
            offspring = self.recombination(matingPool)
            
            # mutation
            self.mutation(offspring)
            
            # survivor selection
            offspringFitness = self.evaluate(offspring)
            population, populationFitness = self.selectSurvivors(
                population, populationFitness, offspring, offspringFitness)
            
        bestGenotype = population[np.argmin(populationFitness)]
        return {
            'bestGenotype' : bestGenotype}
        
        
        
        
    def initialisePopulation(self):
        population = np.zeros((self.populationSize, self.puzzle['dims'][0], self.puzzle['dims'][1]))
        for ind in range(0,self.populationSize):
            order = range(0, self.puzzle['dims'][0]*self.puzzle['dims'][1])
            random.shuffle(order)
            order = order[:self.puzzle['volume']]
            for r in range(0, self.puzzle['dims'][0]):
                for c in range(0, self.puzzle['dims'][1]):
                    population[ind][r][c] = (r*self.puzzle['dims'][1] + c) in order
        return population
            
    def evaluate(self, population):
        fitnesses = np.zeros(len(population))
        for i in range(0, len(population)):
            ind = population[i]
            fitnesses[i] = self.nonogram.evaluate(ind)
        return fitnesses
    
    def selectParents(self, population):
        # Uniform random selection
        matingPool = []
        for i in (self.offspringSize/2)*[True]:
            matingPool.append([random.sample(population,1)[0],random.sample(population,1)[0]])
        return matingPool
        
    def recombination(self, matingPool):
        # row-wise crossover
        nRows = len(matingPool[0][0])
        offspring = []
        for (parent1, parent2) in matingPool:
            child1 = np.copy(parent1)
            child2 = np.copy(parent2)
            for r in range(0, nRows):
                if random.random() < self.recombRateRow:
                    child1[r] = parent2[r]
                    child2[r] = parent1[r]
            offspring.append(child1)
            offspring.append(child2)
        return offspring
    
    def mutation(self, offspring):
        # cell-wise crossover
        nRows = len(offspring[0])
        mColumns = len(offspring[0][0])
        for genotype in offspring:
            for r in range(0, nRows):
                for c in range(0, mColumns):
                    if random.random() < self.mutationRate:
                        genotype[r][c] = np.abs(genotype[r][c] - 1)

                        
    def selectSurvivors(self, population, populationFitness, offspring, offspringFitness):
        # elitism + best 100-5 of offspring
        eliteIndices = np.argsort(populationFitness)
        elite = np.take(population, eliteIndices[:self.eliteSize], 0)
        eliteFitness = np.take(populationFitness, eliteIndices[:self.eliteSize])
        bestIndices = np.argsort(offspringFitness)
        best = np.take(offspring, bestIndices[:self.populationSize - self.eliteSize], 0)
        bestFitness = np.take(offspringFitness, bestIndices[:self.populationSize - self.eliteSize])

        return np.append(elite, best, 0), np.append(eliteFitness, bestFitness)
        
