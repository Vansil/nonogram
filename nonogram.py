'''
Created on Sep 28, 2018

@author: silvan de boer
'''
import numpy as np

class Nonogram(object):
    
    # Puzzle definitions
    def __init__(self, puzzleName):
        self.evaluationCount = 0
        self.puzzle = puzzles[puzzleName]
    
    # Puzzle properties
    def puzzleDims(self):
        return (len(self.puzzle['raw'][0]),
                len(self.puzzle['raw'][1]))
    
    def puzzleVolume(self):
        return sum([sum(x) for x in self.puzzle['raw'][0]])
    
    
    def evaluateDirect(self, phenotype):
        self.evaluationCount+=1
        return sum([phenotype[x][y] != self.puzzle['solution'][x][y] 
                   for x in range(5) for y in range(5)])

    
    
    def evaluate(self, phenotype):
        self.evaluationCount += 1
        fitness = 0
        nRows, mColumns = self.puzzle['dims']
        puzzle = self.puzzle['raw']
        phenotype = np.array(phenotype)
        # rows
        for r in range(0, nRows):
            seriesPuzzle = np.array(puzzle[0][r])
            seriesPheno = self.getSeries(phenotype[r])
            fitness += self.compareSeries(seriesPuzzle, seriesPheno)
        # columns
        for c in range(0, mColumns):
            seriesPuzzle = np.array(puzzle[1][c])
            seriesPheno = self.getSeries(phenotype[:,c])
            fitness += self.compareSeries(seriesPuzzle, seriesPheno)
        return fitness
            
        
    def compareSeries(self, s1, s2):
        minLen = min(len(s1), len(s2))
        s1Compare = np.array(s1[:minLen])
        s2Compare = np.array(s2[:minLen])
        return np.sum(np.abs(s1Compare - s2Compare)) + np.sum(s1[minLen:]) + np.sum(s2[minLen:])
        
    
    def getSeries(self, row):
        row = np.append(row, 0) # makes sure the last series is included
        series = np.array([])
        thisLen = 0
        for i in range(0, len(row)):
            if row[i] == 1:
                thisLen += 1
            else:
                if thisLen > 0:
                    series = np.append(series, thisLen)
                thisLen = 0
        return series
    
    
    
def printPhenotype(phenotype):
    str = ''
    for r in phenotype:
        for c in r:
            if c == 1:
                str += '[]'
            else:
                str += ".."
        str += "\n"
    print str
    
puzzles = {
    'small' : {
        'raw' : (
            [[2],[3],[4],[2,1],[1]],
            [[4],[4],[3],[1],[1]]),
        'dims' : (5,5),
        'volume' : 13,
        'solution' : [
            [0,1,1,0,0],
            [1,1,1,0,0],
            [1,1,1,1,0],
            [1,1,0,0,1],
            [1,0,0,0,0]]},
    'larger' : {
        'source' : 'http://www.nonograms.org/nonograms/i/20745',
        'raw' : 
            ([
                [6,6],
                [4,1,2,3,1,3],
                [2,1,1,1,2,2,1,1,3],
                [2,1,1,14,1,2],
                [1,1,1,1,4,1,1,1],
                [2,1,5,2,3,1,2],
                [3,1,3,1,1,1,1,6],
                [6,4,1,1,4],
                [2,2],
                [7,1,5,1,7],
                [2,3,4,1,3,2],
                [1,1,1,5,1,1,1],
                [1,1,1,2,2,1,1,1],
                [32]
            ], [
                [4],
                [2,3],
                [2,1,2,4],
                [1,1,1,1,2,1],
                [3,2,1,2],
                [2,1,1,1,1],
                [1,1,2,1,1],
                [2,1,1,2],
                [1,1,2,2,1],
                [2,1,6],
                [2,1,2,1,2],
                [2,1,1,1,1,1],
                [1,3,1],
                [1,1,1],
                [2,1,1],
                [1,3,1],
                [1,2,3,1],
                [1,3,1],
                [1,1,1,1],
                [1,2,1,1,1],
                [1,1,1],
                [2,1],
                [1,1,1],
                [1,3,1],
                [4,1,1,1,1],
                [2,1,2,1,2],
                [1,1,6],
                [2,2,2,1],
                [1,1,1,2],
                [1,1,2,1,1],
                [2,1,1,1],
                [1,1,1,1,1,2],
                [2,1,2,2,1],
                [2,1,1,4],
                [2,3],
                [4]
            ]),
        'dims' : (14, 36),
        'volume' : 214},
    'medium' : {
        'source' : 'http://www.nonograms.org/nonograms/i/20685',
        'raw' :
            ([
                [2],
                [8],
                [3,2,4],
                [2,2,1,2],
                [1,2,2,1],
                [5,1,2],
                [5,1,1],
                [6,1],
                [1,5],
                [2],
                [1],
                [2],
                [1,1],
                [2,2],
                [2,3,1,2]
            ],[
                [3],
                [2,1,1],
                [1,1,1],
                [2,2],
                [1,3,3],
                [1,2,1,3,2],
                [3,4,1],
                [3,2,2],
                [2,1,2],
                [4,1],
                [2,5,1],
                [1,2],
                [2,1,1],
                [3,1,1],
                [4]
            ]),
        'dims' : (15, 15),
        'volume' : 79
    }
}
