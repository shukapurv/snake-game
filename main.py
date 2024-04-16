"""
main.py
~~~~~~~~~~
Main file for this project
"""

from game import*
from genetic_algorithm import *


"""
Watch games of snake played by my best neural nets !
"""
net = NeuralNetwork()
game = Game()

net.load(filename_weights='saved/apurv1_weights.npy', filename_biases='saved/apurv1_biases.npy')
game.start(display=True, neural_net=net)

net.load(filename_weights='saved/amit_weights.npy', filename_biases='saved/amit_biases.npy')
game.start(display=True, neural_net=net)


"""
Play a game of snake !

I do not recommend it as it is in first person and not that fun
But if you want, you can
"""
# game = Game()
# game.start(playable=True, display=True, speed=10)


"""
Train your own snakes !

Starts the genetic algorithm with parameters that I've already tested
Best snake of each generation is saved in current folder
The training speed depend a lot on your CPU and its cores number

Contact me if you know how to make it run on GPU
"""
# gen = GeneticAlgorithm(population_size=1000, crossover_method='neuron', mutation_method='weight')
# gen.start()