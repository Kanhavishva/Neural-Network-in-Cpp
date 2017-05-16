#include "layer.h"
#include "matrix.h"
#include <math.h>
#include <stdlib.h>

layer::layer(int            input_count,
             int            neuron_count,
             double         momentum,
             act_func       activation_function,
             bool           randomize,
             bool           is_first_layer,
             bool           is_output_layer,
             layer      *   p,
             layer      *   n,
             const char *   name)
{
    this->layer_name             = name;
    this->l_num_in               = input_count;
    this->l_num_neuron           = neuron_count;
    this->global_momentum        = momentum;
    this->is_first_layer         = is_first_layer;
    this->is_output_layer        = is_output_layer;

    this->l_bias_matrix          = gsl_matrix_calloc(1,              this->l_num_neuron);
    this->l_bias_temp_matrix     = gsl_matrix_calloc(1,              this->l_num_neuron);
    this->l_weights_matrix       = gsl_matrix_calloc(this->l_num_in, this->l_num_neuron);

    this->l_input_matrix         = gsl_matrix_calloc(1,              this->l_num_neuron);
    this->l_output_matrix        = gsl_matrix_calloc(1,              this->l_num_neuron);

    if(randomize == true)
    {
        matrix_fill (this->l_bias_matrix,       -1.0, 1.0);
        matrix_fill (this->l_weights_matrix,    -1.0, 1.0);
    }

    this->activation_function = activation_function;

    // error matrices
    this->error_derivative_wrt_output_matrix     = gsl_matrix_calloc (1, this->l_num_neuron);
    this->output_derivative_wrt_input_matrix     = gsl_matrix_calloc (1, this->l_num_neuron);
    this->input_derivative_wrt_weights_matrix    = gsl_matrix_calloc (this->l_num_in, this->l_num_neuron);
    this->error_derivative_wrt_weights_matrix    = gsl_matrix_calloc (this->l_num_in, this->l_num_neuron);
    this->error_derivative_wrt_bias_matrix       = gsl_matrix_calloc (1, this->l_num_neuron);

    this->accumulated_delta_bias                 = gsl_matrix_calloc (1,              this->l_num_neuron);
    this->accumulated_delta_weights              = gsl_matrix_calloc (this->l_num_in, this->l_num_neuron);

    // connect layers
    this->prev  = p;
    this->next  = n;
}

layer::~layer ()
{
    gsl_matrix_free(this->l_bias_matrix             );
    gsl_matrix_free(this->l_bias_temp_matrix        );
    gsl_matrix_free(this->l_weights_matrix          );
    gsl_matrix_free(this->l_input_matrix            );
    gsl_matrix_free(this->l_output_matrix           );

    gsl_matrix_free(this->accumulated_delta_bias    );
    gsl_matrix_free(this->accumulated_delta_weights );

    gsl_matrix_free(this->error_derivative_wrt_output_matrix  );
    gsl_matrix_free(this->output_derivative_wrt_input_matrix  );
    gsl_matrix_free(this->input_derivative_wrt_weights_matrix );
    gsl_matrix_free(this->error_derivative_wrt_weights_matrix );
    gsl_matrix_free(this->error_derivative_wrt_bias_matrix    );

}

void layer::weighted_sum ()
{

    matrix_copy(this->l_bias_temp_matrix, this->l_bias_matrix);

    gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1.0, this->l_raw_input_matrix, this->l_weights_matrix, 1.0, this->l_bias_temp_matrix);

    matrix_copy(this->l_input_matrix, this->l_bias_temp_matrix);
}

void layer::activate ()
{
    switch (this->activation_function)
    {
    case LINEAR:  this->Linear ();
        break;
    case SIGMOID: this->Sigmoid ();
        break;
    case TANH:    this->Tanh ();
        break;
    case RELU:    this->Relu ();
        break;
    case SOFTMAX: this->Softmax ();
        break;
    }
}

void layer::Sigmoid ()
{
    int i, j;

    for (i = 0; i < (int)this->l_input_matrix->size1; i++)
    {
        for (j = 0; j < (int)this->l_input_matrix->size2; j ++)
        {
            gsl_matrix_set(this->l_output_matrix, i, j, (1.0/(1.0 + exp(-(gsl_matrix_get(this->l_input_matrix, i, j))))));
        }
    }
}

void layer::Tanh ()
{
    int i, j;

    for (i = 0; i < (int)this->l_input_matrix->size1; i++)
    {
        for (j = 0; j < (int)this->l_input_matrix->size2; j ++)
        {
            gsl_matrix_set(this->l_output_matrix, i, j, (tanh(gsl_matrix_get(this->l_input_matrix, i, j))));
        }
    }
}

void layer::Relu ()
{
    int i, j;
    double x;

    for (i = 0; i < (int)this->l_input_matrix->size1; i++)
    {
        for (j = 0; j < (int)this->l_input_matrix->size2; j ++)
        {
            x = gsl_matrix_get(this->l_input_matrix, i, j);
            gsl_matrix_set(this->l_output_matrix, i, j, x < 0 ? 0:x); // *(1.0 + exp())*/);
        }
    }
}

void layer::Linear ()
{
    int i, j;

    for (i = 0; i < (int)this->l_input_matrix->size1; i++)
    {
        for (j = 0; j < (int)this->l_input_matrix->size2; j ++)
        {
            gsl_matrix_set(this->l_output_matrix, i, j, (gsl_matrix_get(this->l_input_matrix, i, j)));
        }
    }
}

void layer::Softmax ()
{

}

void layer::calculate_error ()
{
    int i, j;
    double E = 0.0;

    if(this->is_output_layer == true)
    {
        int m, n;
        m = this->target_for_output_layer_matrix->size1;
        n = this->target_for_output_layer_matrix->size2; //this->l_num_neuron;
        double Yi, Oi;

        for(i=0; i<m; i++)
        {
            for(j=0; j<n; j++)
            {
                Yi = gsl_matrix_get (this->target_for_output_layer_matrix, i, j);
                Oi = gsl_matrix_get (this->l_output_matrix, i, j);

                //E += (-1) * ( Yi * log(Oi) + (1 - Yi) * log(1 - Oi));

                E += (0.5)*(pow((Yi - Oi), 2));     // mean squared error
            }
        }

        this->error = E;

    }
    else
    {
        ;
    }
}

void layer::error_derivative_wrt_output()
{
    double E_wrt_O = 0.0;
    int  i, j, m, n;

    if(this->is_output_layer == true)
    {
        double Yi=0, Oi=0;
        m = this->error_derivative_wrt_output_matrix->size1;
        n = this->error_derivative_wrt_output_matrix->size2;

        for(i=0; i<m; i++)
        {
            for(j=0; j<n; j++)
            {
                Yi = gsl_matrix_get (this->target_for_output_layer_matrix, i, j);
                Oi = gsl_matrix_get (this->l_output_matrix, i, j);
                //E_wrt_O = (-1) * (Yi * (1 / Oi) + (1 - Yi) * (1 / (1 - Oi)));
                E_wrt_O = (-1)*(Yi - Oi);
                gsl_matrix_set (this->error_derivative_wrt_output_matrix, 0, j, E_wrt_O);
            }
        }
    }
    else
    {
        m = this->error_derivative_wrt_output_matrix->size1;
        n = this->error_derivative_wrt_output_matrix->size2;


        double next_error_wrt_out;
        double next_out_wrt_in;
        double next_weights;
        double error_wrt_h_out = 0.0;

        for(j=0; j < this->l_num_neuron; j++)
        {

            error_wrt_h_out = 0.0;

            for(i=0; i < this->next->l_num_neuron; i++)
            {

                next_error_wrt_out  = gsl_matrix_get(this->next->error_derivative_wrt_output_matrix, 0, i);
                next_out_wrt_in     = gsl_matrix_get(this->next->output_derivative_wrt_input_matrix, 0, i);
                next_weights        = gsl_matrix_get(this->next->l_weights_matrix, j, i);

                error_wrt_h_out    += next_error_wrt_out * next_out_wrt_in * next_weights;
            }

            gsl_matrix_set (this->error_derivative_wrt_output_matrix, 0, j, error_wrt_h_out);

        }
    }

}

void layer::output_derivative_wrt_input()
{
    // Activation function derivatives
    int i, j;
    double Out = 0.0;

    switch (this->activation_function) {
    case SIGMOID:
        for (i = 0; i < (int)this->l_output_matrix->size1; i++)
        {
            for (j = 0; j < (int)this->l_output_matrix->size2; j++)
            {
                Out = gsl_matrix_get(this->l_output_matrix, i, j);
                gsl_matrix_set(this->output_derivative_wrt_input_matrix, i, j, (Out*(1.0-Out)));
            }
        }
        break;
    case TANH:
        for (i = 0; i < (int)this->l_output_matrix->size1; i++)
        {
            for (j = 0; j < (int)this->l_output_matrix->size2; j++)
            {
                Out = gsl_matrix_get(this->l_output_matrix, i, j);
                gsl_matrix_set(this->output_derivative_wrt_input_matrix, i, j, (1.0 - pow(Out, 2)));
            }
        }
        break;
    case RELU:
        for (i = 0; i < (int)this->l_output_matrix->size1; i++)
        {
            for (j = 0; j < (int)this->l_output_matrix->size2; j++)
            {
                Out = gsl_matrix_get(this->l_output_matrix, i, j); //
                gsl_matrix_set(this->output_derivative_wrt_input_matrix, i, j, Out < 0 ? 0:1); // (1.0/(1.0 + exp(-Out))));  //(1.0/(1.0 + exp(-()))))
            }
        }
        break;
    case LINEAR:
        for (i = 0; i < (int)this->l_output_matrix->size1; i++)
        {
            for (j = 0; j < (int)this->l_output_matrix->size2; j++)
            {
                //Out = gsl_matrix_get(this->l_output_matrix, i, j); //
                gsl_matrix_set(this->output_derivative_wrt_input_matrix, i, j, 1.0); // (1.0/(1.0 + exp(-Out))));  //(1.0/(1.0 + exp(-()))))
            }
        }
        break;
    case SOFTMAX:
        break;

    }
}

void layer::input_derivative_wrt_weights()
{
    int i, j;
    double x = 0.0;

    if(this->is_first_layer == true)
    {
        for (i = 0; i < (int)this->input_derivative_wrt_weights_matrix->size1; i++)
        {
            x = gsl_matrix_get (this->l_raw_input_matrix, 0, i);

            for (j = 0; j < (int)this->input_derivative_wrt_weights_matrix->size2; j++)
            {
                gsl_matrix_set (this->input_derivative_wrt_weights_matrix, i, j, x);
            }
        }
    }
    else
    {
        for (i = 0; i < (int)this->input_derivative_wrt_weights_matrix->size1; i++)
        {
            x = gsl_matrix_get (this->prev->l_output_matrix, 0, i);

            for (j = 0; j < (int)this->input_derivative_wrt_weights_matrix->size2; j++)
            {
                gsl_matrix_set (this->input_derivative_wrt_weights_matrix, i, j, x);
            }
        }
    }

}

void layer::error_derivative_wrt_weights()
{
    int i, j;
    double x = 0.0, y = 0.0, z = 0.0;

    this->error_derivative_wrt_output ();
    this->output_derivative_wrt_input ();
    this->input_derivative_wrt_weights ();

    // error derivatives with respect to weights
    for(i=0; i < (int)this->error_derivative_wrt_weights_matrix->size2; i++)
    {
        x = gsl_matrix_get (this->error_derivative_wrt_output_matrix, 0, i);
        y = gsl_matrix_get (this->output_derivative_wrt_input_matrix, 0, i);

        for(j=0; j < (int)this->error_derivative_wrt_weights_matrix->size1; j++)
        {
            z = gsl_matrix_get (this->input_derivative_wrt_weights_matrix, j, i);
            z = x * y * z + (gsl_matrix_get (this->error_derivative_wrt_weights_matrix, j, i) * this->global_momentum);
            gsl_matrix_set (this->error_derivative_wrt_weights_matrix, j, i, z);
        }
    }

    // error derivatives with respect to bias
    for(i=0; i < (int)this->error_derivative_wrt_bias_matrix->size2; i++)
    {
        x = gsl_matrix_get (this->error_derivative_wrt_output_matrix, 0, i);
        y = gsl_matrix_get (this->output_derivative_wrt_input_matrix, 0, i);

        z = x * y * 1 + (gsl_matrix_get (this->error_derivative_wrt_bias_matrix, 0, i) * this->global_momentum);

        gsl_matrix_set (this->error_derivative_wrt_bias_matrix, 0, i, z);
    }
}












