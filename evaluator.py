'''
Created on Sep 29, 2018

@author: silvan
'''
import nonogram
import solver

class Evaluator(object):

    def __init__(self):
        self.test()
    
    
    def test(self):
        gram = nonogram.Nonogram('small')
        sol = solver.Solver(gram)
        evaluation = sol.run()
        
        print "Evaluations: {}\nSolution:".format(gram.evaluationCount)
        print evaluation['bestGenotype']
        
        
Evaluator()