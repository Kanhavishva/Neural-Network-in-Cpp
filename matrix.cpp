#include <iostream>
#include <cstdlib>
#include <math.h>
#include <gsl/gsl_blas.h>
#include "matrix.h"

using namespace std;

void matrix_print(gsl_matrix * x, const char * name)
{

    cout.precision(15);
    cout << std::fixed;

    unsigned int i, j;


    for (i = 0; i < x->size1; i++)
    {

       for (j = 0; j < x->size2; j++)
       {
           double val = gsl_matrix_get (x, i, j);

           if(val == -0.0)
               val = 0.0;

           if(i==0 && j==0)
               cout << "[";


           if(i == 0 )
           {
               if(j==0)
               {
                   if(val < 0)
                   {
                        cout << " " << val ;
                   }
                   else
                   {
                       cout <<  "  " << val ;
                   }
               }
               else
               {
                   if(val < 0)
                   {
                       cout <<  "  " << val ;
                   }
                   else
                   {
                        cout << "   " << val ;
                   }
               }
           }
           else
           {
               if(val < 0)
               {
                   cout <<  "  " << val ;
               }
               else
               {
                   cout <<  "   " << val ;
               }


            }

           if(i+1== x->size1 && j+1== x->size2)
               cout << "  ] " << x->size1 << "x" << x->size2 << " : " <<  name;

       }
       cout << endl;
    }
    cout << endl;
}

void matrix_fill(gsl_matrix * x, float low, float high)
{
    unsigned int i, j;
    for (i = 0; i < x->size1; i++)
    {
       for (j = 0; j < x->size2; j++)
       {
           gsl_matrix_set (x, i, j, low + static_cast <float> (rand())/( static_cast <float> (RAND_MAX/(high-low))));
       }
    }
}


gsl_matrix * matrix_transpose(gsl_matrix * x)
{
    unsigned int i, j;
    unsigned int rows = x->size1;
    unsigned int cols = x->size2;

    gsl_matrix * Tx = gsl_matrix_calloc (cols, rows);

    for(i=0; i<cols; i++)
    {
        for(j=0; j<rows; j++)
        {
            gsl_matrix_set (Tx, i, j, gsl_matrix_get(x, j, i));
        }
    }

    return Tx;
}


void matrix_copy(gsl_matrix * dst, gsl_matrix * src)
{
    unsigned int i, j;
    if(dst->size1 == src->size1 && dst->size2 == src->size2)
    {
        for(i=0; i<dst->size1; i++)
        {
            for(j=0; j<dst->size2; j++)
            {
                gsl_matrix_set(dst, i, j,  gsl_matrix_get(src, i, j));
            }
        }
    }
    else {
       dst = NULL;
    }
}

void matrix_copy(gsl_matrix * dst, int row_d, int col_d, gsl_matrix * src, int row_s, int col_s)
{
    gsl_matrix_set(dst, row_d, col_d,  gsl_matrix_get(src, row_s, col_s));
}

void matrix_row_copy(gsl_matrix * dst, int row_d, gsl_matrix * src, int row_s)
{
    if(dst->size2 >= src->size2)
    {
        for(int i = 0; i < (int)src->size2; i++)
        {
            gsl_matrix_set(dst, row_d, i,  gsl_matrix_get(src, row_s, i));
        }
    }
    else
    {
        cout <<  "in function: " << __func__ << "martix size mis-match" << endl;
    }
}

double matrix_add_all_elements(gsl_matrix * x)
{
    unsigned int i, j;
    unsigned int rows = x->size1;
    unsigned int cols = x->size2;

    double sum = 0.0;

    for(i=0; i<rows; i++)
    {
        for(j=0; j<cols; j++)
        {
            sum += gsl_matrix_get (x, i, j);
        }
    }
    return sum;
}

void matrix_save (gsl_matrix * mat, FILE * f, const char * name, int num)
{//
    if(f != NULL)
    {
        fprintf(f, "<");
        //fprintf(f, "Matrix");
        //fprintf(f, " ");
        fprintf(f, "%s", name);
        fprintf(f, "%d", num);
        fprintf(f, " ");
        fprintf(f, "%zu", mat->size1);
        fprintf(f, " ");
        fprintf(f, "%zu", mat->size2);
        fprintf(f, " ");

        for(int i = 0; i < (int)mat->size1; i++)
        {
            for(int j = 0; j < (int)mat->size2; j++)
            {
                if(i==0 && j==0)
                {
                    fprintf(f, "%-.15f", gsl_matrix_get(mat, i, j));
                }
                else
                {
                    fprintf(f, " ");
                    fprintf(f, "%-.15f", gsl_matrix_get(mat, i, j));
                }

            }
        }
        fprintf(f, ">");
        fprintf(f, "\n");
    }
}
