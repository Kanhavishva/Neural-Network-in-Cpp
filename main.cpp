#include <iostream>
#include <math.h>
#include <cstring>
#include <gsl/gsl_blas.h>
#include <time.h>
#include <stdlib.h>
#include "network.h"
#include "trainer.h"

using namespace std;


int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    int    num_layers    = 2;
    int    num_inputs    = 2;
    int    num_output    = 1;
    double learning_rate = 0.3;
    double momentum      = 0.9;

    int    epoch         = 500;
    int    batch_size    = 4;

    int    samples_count = 4;

    gsl_matrix * dataset = gsl_matrix_calloc (samples_count, num_inputs);
    gsl_matrix * targets = gsl_matrix_calloc (samples_count, num_output);

    // XOR problem dataset
    gsl_matrix_set (dataset, 0, 0, 0); gsl_matrix_set (dataset, 0, 1, 0);
    gsl_matrix_set (dataset, 1, 0, 0); gsl_matrix_set (dataset, 1, 1, 1);
    gsl_matrix_set (dataset, 2, 0, 1); gsl_matrix_set (dataset, 2, 1, 0);
    gsl_matrix_set (dataset, 3, 0, 1); gsl_matrix_set (dataset, 3, 1, 1);
    // XOR target
    gsl_matrix_set (targets, 0, 0, 0);
    gsl_matrix_set (targets, 1, 0, 1);
    gsl_matrix_set (targets, 2, 0, 1);
    gsl_matrix_set (targets, 3, 0, 0);

    srand (time(NULL));
    cout << std::fixed;
    cout.precision(15);


    network * net = new network(learning_rate, momentum, num_inputs, true, num_layers, 3, num_output);

    //net->set_layer_activation_function (net->layers[0], RELU);
    //net->set_layer_activation_function (net->layers[1], RELU);

    //network * net = new network("XOR.txt");   //<<------ create network using saved network file
    trainer * trainer_  = new trainer(net, dataset, targets);

    if(trainer_->is_OK == true)
    {
        trainer_->train (epoch, batch_size, true);
        trainer_->test (dataset, targets);
    }

    net->save_network ("XOR.txt");


    delete dataset;
    delete targets;
    delete net;
    delete trainer_;
    return 0;
}
