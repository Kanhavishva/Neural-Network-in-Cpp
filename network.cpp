#include <math.h>
#include <cstdarg>
#include <cstring>
#include "network.h"
#include "matrix.h"

network::network(double     learning_rate,
                 double     momentum,
                 int        num_input,
                 bool       parameters_randomize,
                 int        num_layer,
                 ...                )
{

    // ectract layers and neurons info from parameters list
    int     i;
    int     _neurons = 0;
    int     _n_at_layer[num_layer];
    va_list _sizes;

    va_start(_sizes, num_layer);
    for(i = 0; i < (int) num_layer; i++)
    {
        _neurons = va_arg(_sizes, unsigned int);
        if(_neurons < 0 || _neurons > 1000000)
            printf ("Error in: %s\n", __func__);
        _n_at_layer[i] = _neurons;
    }
    va_end(_sizes);

    this->trained           = false;
    this->n_num_input       = num_input;
    this->n_num_output      = _n_at_layer[num_layer-1];
    this->n_num_layer       = num_layer;
    this->learning_rate     = learning_rate;
    this->momentum          = momentum;
    this->dataset_matrix    = gsl_matrix_calloc (1, this->n_num_input);
    this->target_matrix     = gsl_matrix_calloc (1, this->n_num_output);

    this->layers            = new layer*[this->n_num_layer];

    for(i = 0; i < (int)this->n_num_layer; i++)
    {
        if(i == 0)
        {
            this->layers[i] = new layer(this->n_num_input, _n_at_layer[i], this->momentum,
                                        SIGMOID, parameters_randomize, true, false, NULL, NULL, "First");

            this->layers[i]->l_raw_input_matrix = this->dataset_matrix;
        }
        else
        {
            if(i == (int)this->n_num_layer - 1)
            {
                this->layers[i] = new layer(this->layers[i-1]->l_num_neuron, this->n_num_output, this->momentum,
                                            SIGMOID, parameters_randomize, false, true, this->layers[i-1], NULL, "Output");

                this->layers[i-1]->next                         = this->layers[i];
                this->layers[i]->target_for_output_layer_matrix = this->target_matrix;
                this->layers[i]->l_raw_input_matrix             = this->layers[i-1]->l_output_matrix;
            }
            else
            {
                this->layers[i] = new layer(this->layers[i-1]->l_num_neuron, _n_at_layer[i], this->momentum,
                                            SIGMOID, parameters_randomize, false, false, this->layers[i-1], NULL, "Hidden");

                this->layers[i-1]->next              = this->layers[i];
                this->layers[i]->l_raw_input_matrix  = this->layers[i-1]->l_output_matrix;

            }
        }
    }
}

network::network(const char * net_file)
{
    int       i, ret;
    char      str[50];
    double    val_double  = 0;

    FILE        * netfile = fopen(net_file, "r");
    net_details * nd      = this->parse_details (netfile);

    this->trained         = nd->trained;
    this->n_num_input     = nd->in;
    this->n_num_output    = nd->out;
    this->n_num_layer     = nd->layer;
    this->learning_rate   = nd->lr;
    this->momentum        = nd->mt;

    this->dataset_matrix  = gsl_matrix_calloc (1, this->n_num_input);
    this->target_matrix   = gsl_matrix_calloc (1, this->n_num_output);

    this->layers          = new layer*[this->n_num_layer];

    for(i = 0; i < (int)this->n_num_layer; i++)
    {
        if(i == 0)
        {
            this->layers[i] = new layer(this->n_num_input, nd->layers[i].neurons, this->momentum,
                                        nd->layers[i].func, !this->trained , true, false, NULL, NULL, "First");

            this->layers[i]->l_raw_input_matrix = this->dataset_matrix;
        }
        else
        {
            if(i == (int)this->n_num_layer - 1)
            {
                this->layers[i] = new layer(this->layers[i-1]->l_num_neuron, this->n_num_output, this->momentum,
                                            nd->layers[i].func, !this->trained , false, true, this->layers[i-1], NULL, "Output");

                this->layers[i-1]->next                         = this->layers[i];
                this->layers[i]->target_for_output_layer_matrix = this->target_matrix;
                this->layers[i]->l_raw_input_matrix             = this->layers[i-1]->l_output_matrix;
            }
            else
            {
                this->layers[i] = new layer(this->layers[i-1]->l_num_neuron, nd->layers[i].neurons, this->momentum,
                                            nd->layers[i].func, !this->trained , false, false, this->layers[i-1], NULL, "Hidden");

                this->layers[i-1]->next             = this->layers[i];
                this->layers[i]->l_raw_input_matrix = this->layers[i-1]->l_output_matrix;

            }
        }
    }

    // update bias and weights matrix
    memset (str, 0, sizeof(str));

    for(i = 0; i < (int)this->n_num_layer; i++)
    {
        ret = fscanf(netfile, "%s", str); // bias name
        ret = fscanf(netfile, "%s", str); // bias rows
        ret = fscanf(netfile, "%s", str); // bias cols
        for(int x = 0; x < (int)this->layers[i]->l_bias_matrix->size1; x++)
        {
            for(int y = 0; y < (int)this->layers[i]->l_bias_matrix->size2; y++)
            {
                memset (str, 0, sizeof(str));
                ret = fscanf(netfile, "%s", str); // bias value
                val_double = atof(str);
                gsl_matrix_set (this->layers[i]->l_bias_matrix, x, y, val_double);
            }
        }

        ret = fscanf(netfile, "%s", str); // weights name
        ret = fscanf(netfile, "%s", str); // weights rows
        ret = fscanf(netfile, "%s", str); // weights cols
        for(int x = 0; x < (int)this->layers[i]->l_weights_matrix->size1; x++)
        {
            for(int y = 0; y < (int)this->layers[i]->l_weights_matrix->size2; y++)
            {
                memset (str, 0, sizeof(str));
                ret = fscanf(netfile, "%s", str); // weights value
                val_double = atof(str);
                gsl_matrix_set (this->layers[i]->l_weights_matrix, x, y, val_double);
            }
        }
    }
    (void)ret;
    fclose(netfile);
}

network::~network ()
{
    gsl_matrix_free(this->dataset_matrix);
    gsl_matrix_free(this->target_matrix);
    delete[] this->layers;
}

void network::set_layer_activation_function(layer * lyr, act_func func)
{
    lyr->activation_function = func;
}

void network::process ()
{
    int i;

    // forward pass
    for (i = 0; i < (int)this->n_num_layer; i++)
    {
        this->layers[i]->weighted_sum ();
        this->layers[i]->activate ();
    }

    // backward pass
    // calculate error at the output layer

    this->layers[this->n_num_layer-1]->calculate_error ();

    for(i = this->n_num_layer-1; i >= 0; i--)
    {
        this->layers[i]->error_derivative_wrt_weights ();
    }

}

void network::update_parameters ()
{
    int    i, j, k;
    double new_w = 0.0;
    double old_w = 0.0;
    double acc_delta_w = 0.0;

    double new_b = 0.0;
    double old_b = 0.0;
    double acc_delta_b = 0.0;

    for (i=0; i < (int)this->n_num_layer; i++)
    {
        for(j=0; j < (int)this->layers[i]->l_weights_matrix->size1; j++)
        {
            for(k=0; k < (int)this->layers[i]->l_weights_matrix->size2; k++)
            {
                old_w       = gsl_matrix_get (this->layers[i]->l_weights_matrix, j ,k);
                acc_delta_w = gsl_matrix_get (this->layers[i]->accumulated_delta_weights, j ,k);
                new_w       = old_w - acc_delta_w;
                gsl_matrix_set (this->layers[i]->l_weights_matrix, j, k, new_w);
            }
        }

        for(j=0; j < (int)this->layers[i]->l_bias_matrix->size1; j++)
        {
            for(k=0; k < (int)this->layers[i]->l_bias_matrix->size2; k++)
            {
                old_b       = gsl_matrix_get (this->layers[i]->l_bias_matrix, j ,k);
                acc_delta_b = gsl_matrix_get (this->layers[i]->accumulated_delta_bias, j ,k);
                new_b       = old_b - acc_delta_b;
                gsl_matrix_set (this->layers[i]->l_bias_matrix, j, k, new_b);
            }
        }
    }
}

void network::accumulate_parameters ()
{
    int    i, j, k;
    double new_w    = 0.0;
    double old_w    = 0.0;
    double delta_w  = 0.0;

    double new_b    = 0.0;
    double old_b    = 0.0;
    double delta_b  = 0.0;

    for (i=0; i < (int)this->n_num_layer; i++)
    {
        for(j=0; j < (int)this->layers[i]->accumulated_delta_weights->size1; j++)
        {
            for(k=0; k < (int)this->layers[i]->accumulated_delta_weights->size2; k++)
            {
                old_w   = gsl_matrix_get (this->layers[i]->accumulated_delta_weights, j ,k);
                delta_w = gsl_matrix_get (this->layers[i]->error_derivative_wrt_weights_matrix, j ,k);
                new_w   = old_w + this->learning_rate * delta_w;
                gsl_matrix_set (this->layers[i]->accumulated_delta_weights, j, k, new_w);
            }
        }

        for(j=0; j < (int)this->layers[i]->accumulated_delta_bias->size1; j++)
        {
            for(k=0; k < (int)this->layers[i]->accumulated_delta_bias->size2; k++)
            {
                old_b   = gsl_matrix_get (this->layers[i]->accumulated_delta_bias, j ,k);
                delta_b = gsl_matrix_get (this->layers[i]->error_derivative_wrt_bias_matrix, j ,k);
                new_b   = old_b + this->learning_rate * delta_b;
                gsl_matrix_set (this->layers[i]->accumulated_delta_bias, j, k, new_b);
            }
        }
    }
}

void network::clear_accumulated_parameters ()
{
    int i, j, k;

    for (i=0; i < (int)this->n_num_layer; i++)
    {
        for(j=0; j < (int)this->layers[i]->accumulated_delta_weights->size1; j++)
        {
            for(k=0; k < (int)this->layers[i]->accumulated_delta_weights->size2; k++)
            {
                gsl_matrix_set (this->layers[i]->accumulated_delta_weights, j, k, 0.0);
            }
        }

        for(j=0; j < (int)this->layers[i]->accumulated_delta_bias->size1; j++)
        {
            for(k=0; k < (int)this->layers[i]->accumulated_delta_bias->size2; k++)
            {
                gsl_matrix_set (this->layers[i]->accumulated_delta_bias, j, k, 0.0);
            }
        }
    }
}

act_func network::get_act_function (const char *str)
{
    act_func func = SIGMOID;

    if(strcmp (str, "SIGMOID") == 0)
    {
        func =  SIGMOID;
    }
    else if(strcmp (str, "TANH") == 0)
    {
        func =  TANH;
    }
    else if(strcmp (str, "RELU") == 0)
    {
        func =  RELU;
    }

    return func;
}

net_details * network::parse_details (FILE *f)
{
    int           ret;
    char          str[50];
    double        val_double;
    int           val_int;

    net_details * nd = new net_details;

    memset (str, 0, sizeof(str));

    ret = fscanf(f, "%s", str); //name

    memset (str, 0, sizeof(str));
    ret = fscanf(f, "%s", str); //input:
    memset (str, 0, sizeof(str));
    ret = fscanf(f, "%s", str);
    val_int = atoi (str);
    nd->in  = val_int;

    memset (str, 0, sizeof(str));
    ret = fscanf(f, "%s", str); //output:
    memset (str, 0, sizeof(str));
    ret = fscanf(f, "%s", str);
    val_int = atoi (str);
    nd->out = val_int;

    memset (str, 0, sizeof(str));
    ret = fscanf(f, "%s", str); //layer:
    memset (str, 0, sizeof(str));
    ret = fscanf(f, "%s", str);
    val_int = atoi (str);
    nd->layer = val_int;

    memset (str, 0, sizeof(str));
    ret = fscanf(f, "%s", str); //lr:
    memset (str, 0, sizeof(str));
    ret = fscanf(f, "%s", str);
    val_double = atof (str);
    nd->lr = val_double;

    memset (str, 0, sizeof(str));
    ret = fscanf(f, "%s", str); //mt:
    memset (str, 0, sizeof(str));
    ret = fscanf(f, "%s", str);
    val_double = atof (str);
    nd->mt = val_double;

    nd->layers = new struct layer_details[nd->layer];

    for(int i=0; i < nd->layer; i++)
    {
        memset (str, 0, sizeof(str));
        ret = fscanf(f, "%s", str); //layer %d
        memset (str, 0, sizeof(str));
        ret = fscanf(f, "%s", str); //neurons %d
        val_int = atoi(str);
        nd->layers[i].neurons = val_int;
        memset (str, 0, sizeof(str));
        ret = fscanf(f, "%s", str); //func %d
        nd->layers[i].func = get_act_function (str);
        //act_func k = nd->layers[i].func;

        if(i == 0)
        {
            nd->layers[i].in_count = nd->in;
        }
        else
        {
            nd->layers[i].in_count = nd->layers[i-1].neurons;
        }
    }

    memset (str, 0, sizeof(str));
    ret = fscanf(f, "%s", str); //trained status:
    memset (str, 0, sizeof(str));
    ret = fscanf(f, "%s", str);
    val_int = atoi (str);
    if(val_int == 1)
    {
        nd->trained = true;
    }
    else
    {
        nd->trained = false;
    }

    (void)ret;
    return nd;
}

const char * network::get_act_function_name (act_func func)
{
    switch (func) {
    case SIGMOID:
        return "SIGMOID";
        break;
    case SOFTMAX:
        return "SOFTMAX";
        break;
    case LINEAR:
        return "LINEAR";
        break;
    case RELU:
        return "RELU";
        break;
    case TANH:
        return "TANH";
        break;
    }
    return 0;
}

void network::save_network (const char * name)
{
    FILE * f = fopen (name, "w");

    if(f != NULL)
    {
        fprintf (f, "%s\n",                  name                );
        fprintf (f, "input: %d;\n",          this->n_num_input   );
        fprintf (f, "output: %d;\n",         this->n_num_output  );
        fprintf (f, "layers: %d;\n",         this->n_num_layer   );
        fprintf (f, "learnig_rate: %f;\n",   this->learning_rate );
        fprintf (f, "momentum: %f;\n",       this->momentum      );

        for(int i=0; i < (int)this->n_num_layer; i++)
        {
            fprintf (f, "layer%d: %d; %s\n", i, this->layers[i]->l_num_neuron, get_act_function_name (this->layers[i]->activation_function));
        }

        if(this->trained == true)
        {
            fprintf (f, "trained: %d;\n", 1);
        }
        else
        {
            fprintf (f, "trained: %d;\n", 0);
        }

        if(this->trained == true)
        {
            for(int i=0; i < (int)this->n_num_layer; i++)
            {
                matrix_save (this->layers[i]->l_bias_matrix, f, "bias", i);
                matrix_save (this->layers[i]->l_weights_matrix, f, "weights", i);
            }
        }

    }
    fclose(f);
}



























