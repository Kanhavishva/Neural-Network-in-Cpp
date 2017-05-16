//#include "matrix.h"
#include "matplotlib.h"
#include "network.h"
#include <iostream>
#include <math.h>
#include <cstring>
#include "network.h"
#include "trainer.h"
#include <gsl/gsl_blas.h>
#include <time.h>
#include <stdlib.h>

using namespace std;
namespace plt = matplotlibcpp;

trainer::trainer(network * net, gsl_matrix * dataset, gsl_matrix * target)
{
    if(dataset->size2 == net->n_num_input || target->size2 == net->n_num_output)
    {
        this->dataset = dataset;
        this->target  = target;
        this->trained = false;
        this->is_OK   = true;
        this->net     = net;

        cout << std::fixed;
        cout.precision(15);
    }
    else
    {
        this->is_OK   = false;
        std::cout << "dataset or target size mis-match with network" << std::endl;
    }

}

trainer::~trainer ()
{
    this->error_vector.clear ();
}

void trainer::train (int epoch, int batch_size, bool draw_error_plot)
{
    this->batch_size    = batch_size;
    this->num_samples   = this->dataset->size1;
    int sample_index    = 0;
    int sample_remain   = this->num_samples;
    bool epoch_complete = false;

    for(int iter=0; iter < epoch; iter++)
    {
        this->batch_size    = batch_size;
        sample_index        = 0;
        sample_remain       = this->num_samples;
        epoch_complete      = false;

        do
        {
            for(int i=0; i < this->batch_size; i++)
            {
                matrix_copy (this->net->dataset_matrix, 0, 0, this->dataset, sample_index + i, 0);
                matrix_copy (this->net->dataset_matrix, 0, 1, this->dataset, sample_index + i, 1);

                matrix_copy (this->net->target_matrix,  0, 0, this->target,  sample_index + i, 0);

                this->net->process ();
                this->net->accumulate_parameters ();
            }
            this->net->update_parameters ();
            this->net->clear_accumulated_parameters ();

            sample_remain = sample_remain - this->batch_size;
            if(sample_remain > 0)
            {
                if(sample_remain >= this->batch_size)
                {
                    sample_index += this->batch_size;
                }
                else
                {
                    this->batch_size = sample_remain;
                    sample_index += this->batch_size;
                }
            }
            else if(sample_remain <= 0)
            {
                epoch_complete = true;
            }

        }while(epoch_complete == false);

        cout << "Iterations  = " << iter << endl; cout << endl;
        cout << "Total error = " << this->net->layers[this->net->n_num_layer-1]->error << endl;
        cout << "--------------------------------------------------------" << endl;

        if(draw_error_plot == true)
        {
            this->error_vector.push_back (this->net->layers[this->net->n_num_layer-1]->error);
        }

    }

    this->trained = true;
    this->net->trained = true;

    if(draw_error_plot == true)
    {
        plt::plot   (this->error_vector      );
        plt::grid   (true                    );
        plt::title  ("Error plot"            );
        plt::xlabel ("Iterations -->"        );
        plt::ylabel ("Mean Squared Error -->");
        plt::show   ();
    }

}

void trainer::test (gsl_matrix * dataset, gsl_matrix * target)
{
    this->dataset = dataset;
    this->target  = target;

    for(int i=0; i < (int)this->dataset->size1; i++)
        {
            cout << "TESTING . . ." << endl;

            matrix_copy(this->net->dataset_matrix, 0, 0, this->dataset, i, 0);
            matrix_copy(this->net->dataset_matrix, 0, 1, this->dataset, i, 1);
            matrix_copy(this->net->target_matrix,  0, 0, this->target,  i, 0);

            this->net->process ();

            matrix_print (this->net->layers[0]->l_raw_input_matrix,                                    "inputs");
            matrix_print (this->net->layers[this->net->n_num_layer-1]->target_for_output_layer_matrix, "target");
            matrix_print (this->net->layers[this->net->n_num_layer-1]->l_output_matrix,                "output");
        }
}
