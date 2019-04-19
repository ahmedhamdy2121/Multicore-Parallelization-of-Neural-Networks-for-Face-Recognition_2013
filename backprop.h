/*
 ******************************************************************
 * HISTORY
 * 15-Oct-94  Jeff Shufelt (js), Carnegie Mellon University
 *      Prepared for 15-681, Fall 1994.
 *
 ******************************************************************
 */

#ifndef _BACKPROP_H_

#define _BACKPROP_H_

#define DBL_MIN 1e-50

/*** global variables, they being initialized in bpnn_initialize function ***/
double max_weight_change_err; // weight change error
// make each thread has its own copy
#pragma omp threadprivate(max_weight_change_err)

/***
The neural network data structure.  The network is assumed to
be a fully-connected feedforward three-layer network.
Unit 0 in each layer of units is the threshold unit; this means
that the remaining units are indexed from 1 to n, inclusive.
 ***/

typedef struct {
  int input_n;                  /* number of input units */
  int hidden_n;                 /* number of hidden units */
  int output_n;                 /* number of output units */

  double *input_units;          /* the input units */
  double *hidden_units;         /* the hidden units */
  double *output_units;         /* the output units */


  double *hidden_delta;         /* storage for hidden unit error */
  double *output_delta;         /* storage for output unit error */

  double *target;               /* storage for target vector */

  double **input_weights;       /* weights from input to hidden layer */
  double **hidden_weights;      /* weights from hidden to output layer */

  /*** The next two are for momentum, they are delta not weight. i.e. w_old - w_old-1 ***/
  double **input_prev_weights;  /* previous **change** on input to hidden wgt */
  double **hidden_prev_weights; /* previous **change** on hidden to output wgt */
} BPNN;

double drnd();
double dpn1();
double dpn2();

/*** User-level functions ***/

void bpnn_initialize();

BPNN *bpnn_create();
void bpnn_free();

void bpnn_train();
void bpnn_feedforward();


#endif
