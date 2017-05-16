#ifndef LAYER_H
#define LAYER_H
#include <gsl/gsl_blas.h>

enum act_func
{
    LINEAR = 0,
    SIGMOID,
    TANH,
    RELU,
    SOFTMAX
};
//act_func activation;

class layer
{
public:
    layer(){;}
    layer(int            input_count,
          int            neuron_count,
          double         momentum,
          act_func       activation_function,
          bool           randomize,
          bool           is_first_layer,
          bool           is_output_layer,
          layer      *   p,
          layer      *   n,
          const char *   name);
    ~layer();

public:
    int             l_num_in;
    int             l_num_neuron;
    const char *    layer_name;
    gsl_matrix *    l_raw_input_matrix;
    gsl_matrix *    l_weights_matrix;
    gsl_matrix *    l_bias_matrix;
    gsl_matrix *    l_bias_temp_matrix;
    gsl_matrix *    l_input_matrix;
    gsl_matrix *    l_output_matrix;
    layer      *    prev;
    layer      *    next;
    gsl_matrix *    target_for_output_layer_matrix;
    gsl_matrix *    accumulated_delta_bias;
    gsl_matrix *    accumulated_delta_weights;
    gsl_matrix *    error_derivative_wrt_weights_matrix;
    gsl_matrix *    error_derivative_wrt_bias_matrix;
    gsl_matrix *    error_derivative_wrt_output_matrix;
    gsl_matrix *    output_derivative_wrt_input_matrix;
    gsl_matrix *    input_derivative_wrt_weights_matrix;
    double          error;
    double          global_momentum;
    act_func        activation_function;

    void            weighted_sum                    ();
    void            activate                        ();
    void            error_derivative_wrt_weights    ();
    void            calculate_error                 ();

private:
    bool            is_output_layer;
    bool            is_first_layer;

    void            Linear                          ();
    void            Sigmoid                         ();
    void            Tanh                            ();
    void            Relu                            ();
    void            Softmax                         ();
    void            error_derivative_wrt_output     ();
    void            output_derivative_wrt_input     ();
    void            input_derivative_wrt_weights    ();


};

#endif // LAYER_H
