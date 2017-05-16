#ifndef NET_H
#define NET_H
#include "layer.h"

struct layer_details
{
    int             in_count;
    int             neurons;
    act_func        func;
};

typedef struct
{
    int             in;
    int             out;
    int             layer;
    double          lr;
    double          mt;
    bool            trained;
    layer_details * layers;
}net_details;

class network
{
public:
    network(double      learning_rate,
            double      momentum,
            int         num_input,
            bool        parameters_randomize,
            int         num_layer,
            ...                 );
    network(const char * netfile);
    ~network();

    bool                trained;
    double              learning_rate;
    double              momentum;
    double              total_error;
    unsigned int        n_num_input;
    unsigned int        n_num_layer;
    unsigned int        n_num_output;
    gsl_matrix   *      dataset_matrix;
    gsl_matrix   *      target_matrix;
    layer       **      layers;

    void                process                         ();
    void                update_parameters               ();
    void                accumulate_parameters           ();
    void                clear_accumulated_parameters    ();
    void                save_network                    (const char * name);
    void                set_layer_activation_function   (layer * lyr, act_func func);

private:
    act_func            get_act_function                (const char * str);
    net_details *       parse_details                   (FILE * f);
    const char  *       get_act_function_name           (enum act_func func);
};

#endif // NET_H
