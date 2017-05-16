#ifndef MATRIX_H
#define MATRIX_H

void            matrix_print            (gsl_matrix * x, const char * name);
void            matrix_fill             (gsl_matrix * x, float low, float high);
gsl_matrix *    matrix_transpose        (gsl_matrix * x);
void            matrix_copy             (gsl_matrix * dst, gsl_matrix * src);
void            matrix_copy             (gsl_matrix * dst, int row_d, int col_d, gsl_matrix * src, int row_s, int col_s);
void            matrix_row_copy         (gsl_matrix * dst, int row_d, gsl_matrix * src, int row_s);
double          matrix_add_all_elements (gsl_matrix * x);
void            matrix_save             (gsl_matrix * mat, FILE * f, const char * name, int num);

#endif // MATRIX_H
