## Simple Neural Network in C++
I am a beginner in machine learning, so this neural network is created for the purpose of learning.

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
