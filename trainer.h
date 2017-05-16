#ifndef TRAINER_H
#define TRAINER_H
#include "matrix.h"
#include <vector>

class trainer
{
public:
    trainer(network * net, gsl_matrix * dataset, gsl_matrix * target);
    ~trainer();

    bool                trained;
    bool                is_OK;
    int                 num_samples;
    std::vector<double> error_vector;

    void                train       (int epoch, int batch_size, bool draw_error_plot = false);
    void                test        (gsl_matrix * dataset, gsl_matrix * target);

private:
    network    *        net;
    gsl_matrix *        dataset;
    gsl_matrix *        target;
    int                 batch_size;

};

#endif // TRAINER_H
