'''
Created on Sep 29, 2018

@author: silvan
'''
import random
import nonogram as nn
import numpy as np
from numpy.core.fromnumeric import argmin
import nonogram

class Solver(object):
    '''
    classdocs
    '''


    def __init__(self, nonogram):
        self.puzzle = nonogram.puzzle
        self.nonogram = nonogram

        self.populationSize = 1000
        self.offspringSize = 4000
        self.recombRateRect = 1
        self.mutationShiftStepSize = 0 # st. dev. of shift per row and column
        self.mutationCellSwapRate = 0#1.0 / 30
        self.eliteSize = 5
        
    def run(self):
        # initialisation
        population = self.initialisePopulation()
        populationFitness = self.evaluate(population)
        
        # used for hill-climbing condition
        bestFitness = max(populationFitness)
        gensWithoutImprovement = -1
        doHillClimbing = False
        
        generationCount = 0
        while np.min(populationFitness) != 0:
            if (generationCount % 1 == 0):
                print "Best fitness after generation {}:\t{}".format(
                    generationCount, min(populationFitness))
                hammingAvg, hammingStd = self.diversitySampleHamming(population, 200)
                print 'Hamming sample diversity: avg {}\tstd {}'.format(hammingAvg, hammingStd)
            if (generationCount % 20 == 0):
                nonogram.printPhenotype(population[argmin(populationFitness)])
            generationCount += 1
            
            # determine hill-climbing condition
            if min(populationFitness) >= bestFitness:
                gensWithoutImprovement += 1
            else:
                gensWithoutImprovement = 0
                bestFitness = min(populationFitness)
            if gensWithoutImprovement % 10 == 9: print '{} generations without improvement'.format(gensWithoutImprovement)
            if gensWithoutImprovement > 20 and not doHillClimbing: 
                doHillClimbing = True
                print 'HILLCLIMBING IS NOW APPLIED'
            
            
            print np.sum(population[0])
            print np.sum(population[1])
            
            # parent selection
            matingPool = self.selectParents(population)
            
            # recombination
            offspring = self.recombination(matingPool)
            
            # mutation
            offspring = self.mutation(offspring)
            
            # hill-climbing
            if doHillClimbing:
                offspring, offspringFitness = self.hillClimb(offspring)
            else:
                offspringFitness = self.evaluate(offspring)
                
            # survivor selection
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
        return self.recombinationAreaWise(matingPool)
    
    
    def hillClimb(self, offspring):
        # shift a row to all positions and pick the best one
        nColumns = np.size(offspring, 2)
        offspringFitness = []
        for index in range(0, np.size(offspring, 0)):
            individual = offspring[index]
            row = random.randint(0, np.size(individual, 0) - 1)
            #lazy
            candidates = [np.array(individual)]
            for shift in range(1, nColumns):
                cand = np.copy(individual)
                # shift
                cand[row] = np.append(cand[row][shift:], cand[row][:shift])
                # add to candidates
                candidates.append(cand)
            fitness = self.evaluate(candidates)
            # replace individual with fittest neighbour
            bestIndex = np.argmin(fitness)
            offspring[index] = candidates[bestIndex]
            offspringFitness.append(fitness[bestIndex])
        return offspring, offspringFitness
    
        
    def recombinationAreaWise(self, matingPool):
        # row-wise crossover
        nRows = np.size(matingPool[0][0], 0)
        mColumns = np.size(matingPool[0][0], 1)
        offspring = []
        for (parent1, parent2) in matingPool:
            child1 = np.copy(parent1)
            child2 = np.copy(parent2)
            
            if random.random() < self.recombRateRect:
                areas = self.recombinationAreaWiseAreas(nRows, mColumns)
                for area in areas:
                    child1[area[0]:area[2]+1, area[1]:area[3]+1] = parent2[area[0]:area[2]+1, area[1]:area[3]+1]
                    child2[area[0]:area[2]+1, area[1]:area[3]+1] = parent1[area[0]:area[2]+1, area[1]:area[3]+1]
            
            offspring.append(child1)
            offspring.append(child2)
        return offspring
    
    def recombinationAreaWiseAreas(self, nRows, mColumns):
        # Gives coordinates of one 'rectangle' in the plane.
        # If the first corner is not at the left top of the second corner, 
        # the area spans 'over the edges' and multiple 'correct rectangles' are returned
        area = [random.randint(0, nRows - 1),    # row 1
                random.randint(0, mColumns - 1), # col 1
                random.randint(0, nRows - 1),    # row 2
                random.randint(0, mColumns - 1)] # col 2
        
        # flip in row direction
        if area[0] > area[2]: 
            areas = [[0, area[1], area[2], area[3]],
                     [area[0], area[1], nRows - 1, area[3]]]
        else:
            areas = [area]
        # flip in column direction
        if area[1] > area[3]:
            _areas = []
            for a in areas:
                _areas.append([a[0], 0, a[2], a[3]])
                _areas.append([a[0], a[1], a[2], mColumns - 1])
            areas = _areas
        # return array of sub-areas
        return areas
        
        
    def recombinationAreaWiseAreasInternal(self, nRows, mColumns):
        # Gives coordinates of one 'rectangle' in the plane, no flipping
        area = [random.randint(0, nRows - 1),    # row 1
                random.randint(0, mColumns - 1), # col 1
                random.randint(0, nRows - 1),    # row 2
                random.randint(0, mColumns - 1)] # col 2
        
        return [[min(area[0], area[2]), min(area[1],area[3]),
                max(area[0], area[2]), max(area[1],area[3])]]
        
    
    
    def mutation(self, offspring):
        if self.mutationShiftStepSize > 0:
            offspring = self.mutationShift(offspring)
        if self.mutationCellSwapRate > 0:
            offspring = self.mutationCellSwap(offspring)
        
        return offspring
                    
    
    def mutationCellSwap(self, offspring):
        # swap cells in each individual
        nRows = np.size(offspring, 1)
        mColumns = np.size(offspring, 2)
        for index in range(0, np.size(offspring, 0)):
            individual = offspring[index]
            p = self.mutationCellSwapRate
            n = nRows * mColumns
            # a direct computation of the amount of cells that would be swapped if we looped over all n cells and swapped with probability p (binomial distr.)
            nSwaps = int(random.gauss(p*n, np.sqrt(n*p*(1-p))))
            for i in nSwaps*[True]:
                r1 = random.randint(0, nRows - 1)
                c1 = random.randint(0, mColumns - 1)
                r2 = random.randint(0, nRows - 1)
                c2 = random.randint(0, mColumns - 1)
                v1 = individual[r1][c1]
                v2 = individual[r2][c2]
                individual[r1][c1] = v2
                individual[r2][c2] = v1
        return offspring
                
                
    def mutationShift(self, offspring):
        # shift rows and columns
        nRows = np.size(offspring[0], 0)
        mColumns = np.size(offspring[0], 1)
        for index in range(0, np.size(offspring, 0)):
            individual = offspring[index]
            # decide on order to shift rows and columns
            whatToPick = np.append([range(0,nRows)+range(0,mColumns)],[nRows*[True]+mColumns*[False]],0)
            orderToPick = range(0,nRows+mColumns)
            random.shuffle(orderToPick)
            # shift rows and columns
            for i in orderToPick:
                rc, isRow = whatToPick[:,i]
                if isRow:
                    # shift row
                    r = rc
                    shift = int(np.floor(random.gauss(0, self.mutationShiftStepSize)))
                    if shift < 0: shift += mColumns
                    shift = min(mColumns, max(0, shift))
                    individual[r] = np.append(individual[r][shift:], individual[r][:shift])
                else:
                    # shift column
                    c = rc
                    shift = int(np.floor(random.gauss(0, self.mutationShiftStepSize)))
                    if shift < 0: shift += nRows
                    shift = min(nRows, max(0, shift))
                    individual[:,c] = np.append(individual[:,c][shift:], individual[:,c][:shift])
            offspring[index] = individual
        return offspring
    
                   
    def selectSurvivors(self, population, populationFitness, offspring, offspringFitness):
        # get elite
        eliteIndices = np.argsort(populationFitness)
        elite = np.take(population, eliteIndices[:self.eliteSize], 0)
        eliteFitness = np.take(populationFitness, eliteIndices[:self.eliteSize])
        # combine offspring and elite
        pool = np.append(offspring, elite, 0)
        poolFitness = np.append(offspringFitness, eliteFitness)
        # select best
        bestIndices = np.argsort(poolFitness)
        best = np.take(pool, bestIndices[:self.populationSize], 0)
        bestFitness = np.take(poolFitness, bestIndices[:self.populationSize])

        return best, bestFitness
        
    
    def testRecomRec(self, areas, n, m):
        grid = np.zeros((n, m))
        for a in areas:
            grid[a[0]:a[2]+1, a[1]:a[3]+1] = 1
        n = nn.Nonogram('small')
        nonogram.printPhenotype(grid)
    
    def diversitySampleHamming(self, population, n):
        distances = []
        for i in n*[True]:
            ind1, ind2 = random.sample(population, 2)
            distances.append(self.hammingDistance(ind1, ind2))
        return np.average(distances), np.std(distances)
            
    def hammingDistance(self, ind1, ind2):
        return np.sum(np.logical_xor(ind1, ind2))
    
# n = nonogram.Nonogram('small')
# s = Solver(n)
# off = [[
#             [1,1,0,0,0],
#             [1,1,1,0,0],
#             [1,1,1,1,0],
#             [1,1,0,0,1],
#             [1,0,0,0,0]]]
# off_=s.hillClimb(off, 0)
# print 'now we have, with fitness {}:'.format(n.evaluate(off_[0]))
# print off_


