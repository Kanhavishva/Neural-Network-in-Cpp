## Simple Neural Network in C++
I am a beginner in machine learning, so this neural network is created for the purpose of learning. It's not fully functional but playing with it, is fun for me.

![XOR Network Testing](https://github.com/Kanhavishva/Neural-Network-in-Cpp/raw/master/Screenshot%20from%202017-05-16%2015-19-05.png?raw=true "Network Testing")
![Error plot](https://github.com/Kanhavishva/Neural-Network-in-Cpp/raw/master/Screenshot%20from%202017-05-16%2015-19-57.png?raw=true "Error plot")

## Features:
* It's fast and small
* Mathematical operations are performed matrix-wise
* Network can be created from a text file describing network details
* After training, the network can be saved to a text file describing the network details, trianed network are saved with the network parameters(biases and weights) and untrained network are saved with structure details only like number of layers, number of neurons and type of activation functions etc. This text file can be used to build the trained network again.
* At the end of training, error plot can be drawn using python library 'matplotlib' with c++.

## Depedencies:
* libgsl, gsl_blas  - GNU Scientific Library - https://www.gnu.org/software/gsl/‎
* matplotlibcpp - Extremely simple yet powerful header-only C++ plotting library built on the popular matplotlib - https://github.com/lava/matplotlib-cpp‎
* libpython - Python Library - http://www.python.org/

## Guide:
* I followed the tutorial "Back-Propagation is very simple. Who made it Complicated ?" - https://medium.com/becoming-human/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c
* A Step by Step Backpropagation Example – Matt Mazur - https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
* A Gentle Introduction to Artificial Neural Networks | The Clever Machine - https://theclevermachine.wordpress.com/2014/09/11/a-gentle-introduction-to-artificial-neural-networks/
