/*
 ******************************************************************
 * HISTORY
 * 15-Oct-94  Jeff Shufelt (js), Carnegie Mellon University
 *	Prepared for 15-681, Fall 1994.
 *
 ******************************************************************
 */

#include <stdio.h>
#include <stdlib.h> /* for rand() and srand() */
#include <math.h>
#include <fcntl.h> /* for open() */
#include <unistd.h> /* for close() */
#include "backprop.h"

// calculate abs
#define ABS(x)          (((x) > 0.0) ? (x) : (-(x)))

/*** ++++++++++++++++++++++++ Internal functions only +++++++++++++++++++++++++++ ***/


/*** Return random number between 0.0 and 1.0 ***/
double drnd()
{
    double min = 0.0;
    double max = 1.0;
    double rnd = min + (double) (rand() / (double) (RAND_MAX + 1) * (max - min + 1));
    double intPart;
    double fractpart = modf (rnd*rnd*rnd*10000 , &intPart);
    return fractpart;
}


/*** Return random number between -1.0 and 1.0 ***/
double dpn1()
{
    return ((drnd() * 2.0) - 1.0);
}


/*** Return random number between -0.5 and 0.5 ***/
double dpn2()
{
    return ((drnd() ) - 0.5);
}

/*** The squashing function.  Currently, it's a sigmoid. ***/
double squash(x)
double x;
{
    return (1.0 / (1.0 + exp(-x)));
}


/*** Allocate 1d array of doubles ***/
double *alloc_1d_dbl(n)
int n;
{
    double *new;

    new = (double *) malloc ((unsigned) (n * sizeof (double)));
    if (new == NULL)
    {
        printf("ALLOC_1D_DBL: Couldn't allocate array of doubles\n");
        return (NULL);
    }
    return (new);
}


/*** Allocate 2d array of doubles ***/
double **alloc_2d_dbl(m, n)
int m, n;
{
    int i;
    double **new;

    new = (double **) malloc ((unsigned) (m * sizeof (double *)));
    if (new == NULL)
    {
        printf("ALLOC_2D_DBL: Couldn't allocate array of dbl ptrs\n");
        return (NULL);
    }

    for (i = 0; i < m; i++)
    {
        new[i] = alloc_1d_dbl(n);
    }

    return (new);
}


void bpnn_randomize_weights(w, m, n)
double **w;
int m, n;
{
    int i, j;

    for (i = 0; i <= m; i++)
    {
        for (j = 0; j <= n; j++)
        {
            w[i][j] = dpn2();
        }
    }
}


void bpnn_zero_weights(w, m, n)
double **w;
int m, n;
{
    int i, j;

    for (i = 0; i <= m; i++)
    {
        for (j = 0; j <= n; j++)
        {
            w[i][j] = 0.0;
        }
    }
}

/**
Copy the weights from w1 to w
**/
void copy_weights(w, w1, ncurrent, nprev)
double **w, **w1;
int ncurrent, nprev;
{
    int j, k;

    for (k = 1; k <= ncurrent; k++)
    {
        for (j = 0; j <= nprev; j++)
        {
            w[j][k] = w1[j][k];
        }
    }
}


BPNN *bpnn_internal_create(n_in, n_hidden, n_out)
int n_in, n_hidden, n_out;
{
    BPNN *newnet;

    newnet = (BPNN *) malloc (sizeof (BPNN));

    if (newnet == NULL)
    {
        printf("BPNN_CREATE: Couldn't allocate neural network\n");
        return (NULL);
    }

    newnet->input_n = n_in;
    newnet->hidden_n = n_hidden;
    newnet->output_n = n_out;

    // +1 as 0 is the phantom input and the real nodes from 1 ... n
    newnet->input_units = alloc_1d_dbl(n_in + 1);
    newnet->hidden_units = alloc_1d_dbl(n_hidden + 1);
    newnet->output_units = alloc_1d_dbl(n_out + 1);

    // to hold the deltak and deltaj
    newnet->hidden_delta = alloc_1d_dbl(n_hidden + 1);
    newnet->output_delta = alloc_1d_dbl(n_out + 1);

    // target output
    newnet->target = alloc_1d_dbl(n_out + 1);

    newnet->input_weights = alloc_2d_dbl(n_in + 1, n_hidden + 1);
    newnet->hidden_weights = alloc_2d_dbl(n_hidden + 1, n_out + 1);

    // for momentum
    newnet->input_prev_weights = alloc_2d_dbl(n_in + 1, n_hidden + 1);
    newnet->hidden_prev_weights = alloc_2d_dbl(n_hidden + 1, n_out + 1);

    return (newnet);
}


/**
This function make the forward path
It takes:
l1: the first layer of nodes
l2: the second layer of nodes
conn: the weights matrix from first layer to second layer
n1: number of nodes in first layer
n2: number of nodes in second layer
**/
void bpnn_layerforward(l1, l2, conn, n1, n2)
double *l1, *l2, **conn;
int n1, n2;
{
    double sum;
    int j, k;

    /*** Set up thresholding unit (phantom input) ***/
    l1[0] = 1.0;

    /*** For each unit in second layer ***/
    for (j = 1; j <= n2; j++)
    {
        /*** Compute weighted sum of its inputs ***/
        sum = 0.0;
        for (k = 0; k <= n1; k++)
        {
            sum += l1[k] * conn[k][j]; // calculate summation of all the input to the current neuron
        }
        l2[j] = squash(sum); // calculate activation
    }
}


/**
NN Model: i -> j -> k
This function just compute delta_k = (y_dk - y_k) * y_k * (1 - y_k)
It takes:
delta: a vector to store the delta_k in it
target: a vector of the target value (y_dk)
output: a vector of the output value (y_k)
nj: number of nodes in output layer -> it is the size of the target vector
err: an address to return the total error
**/
void bpnn_output_error(delta, target, output, nj, err)
double *delta, *target, *output, *err;
int nj;
{
    int j;
    double o, t, errsum;

    errsum = 0.0;
    for (j = 1; j <= nj; j++)
    {
        o = output[j];
        t = target[j];
        delta[j] = (t - o) * o * (1.0 - o); // (y_dk - y_k) * y_k * (1 - y_k)

        errsum += ABS(delta[j]);
    }

    *err = errsum;
}


/**
NN Model: i -> j -> k
This function just compute delta_j = y_j * (1 - y_j) * sum_k(delta_k * w_jk)
It takes:
delta_h: a vector to store the delta_j in it
nh: number of hidden neurons in the hidden layer
delta_o: it is the delta_k we just computed in the above function
no: number of output neurons in output layer
who: it is a matrix of hidden weights
hidden: it is a vector of hidden nodes
err: an address to return the total error
**/
void bpnn_hidden_error(delta_h, nh, delta_o, no, who, hidden, err)
double *delta_h, *delta_o, *hidden, **who, *err;
int nh, no;
{
    int j, k;
    double h, sum, errsum;

    errsum = 0.0;
    for (j = 1; j <= nh; j++)
    {
        h = hidden[j];

        sum = 0.0;
        for (k = 1; k <= no; k++)
        {
            sum += delta_o[k] * who[j][k]; // sum_k(delta_k * w_jk)
        }

        delta_h[j] = h * (1.0 - h) * sum; // y_j * (1 - y_j) * sum

        errsum += ABS(delta_h[j]);
    }

    *err = errsum;
}


/**
NN Model: i -> j -> k
This function making the weight update
delta: is a vector of delta either (delta_k -> update weights from output to hidden layer)
        or (delta_j -> update weights from hidden to input layer)
ndelta: number of elements in the layer (either j or k)
ly: a vector of the previous layer (either hidden "j" or input "i")
nly: the number of nodes in the previous layer
w: the weight matrix I want to update (either w_jk or w_ij)
oldw: the old weight matrix ... for momentum calculations
eta: learning rate
momentum: the momentum
**/
void bpnn_adjust_weights(delta, ndelta, ly, nly, w, oldw, eta, momentum)
double *delta, *ly, **w, **oldw, eta, momentum;
int ndelta, nly;
{
    double new_dw, new_db;
    int k, j;

    /*** Set up thresholding unit (phantom input) ***/
    ly[0] = 1.0;

    for (k = 1; k <= ndelta; k++)
    {
        for (j = 0; j <= nly; j++)
        {
            if (j == 0) // updating bias
            {
                /***
                This will affect the first row in the weights matrix
                Note that: first column on the weights matrix is always the same, because it is redundant.
                ***/
                new_db = (eta * delta[k]) + (momentum * oldw[j][k]); // eta * delta_k OR eta * delta_j
                w[j][k] -= new_db;
                oldw[j][k] = new_db;
            }
            else // updating weight
            {
                new_dw = (eta * delta[k] * ly[j]) + (momentum * oldw[j][k]); // eta * delta_k * y_j OR eta * delta_j * x_i
                w[j][k] += new_dw;
                oldw[j][k] = new_dw;

                // calculating the change in weight error
                if (ABS(new_dw) > max_weight_change_err)
                    max_weight_change_err = ABS(new_dw);
            }
        }
    }
}

/*** +++++++++++++++++++++++++++++++++++++++ END of Internal functions only ++++++++++++++++++++++++++++++++++++++++++ ***/


void bpnn_initialize(seed)
unsigned seed;
{
    printf("Random number generator seed: %d\n", seed);
    srand(seed);

    // init global variable
    max_weight_change_err = DBL_MIN;
}


/***
Creates a new fully-connected network from scratch,
with the given numbers of input, hidden, and output units.
Threshold units are automatically included.  All weights are
randomly initialized.

Space is also allocated for temporary storage (momentum weights,
error computations, etc).

w1: input weights
w2: hidden weights
***/
BPNN *bpnn_create(n_in, n_hidden, n_out, w1, w2)
int n_in, n_hidden, n_out;
double **w1, **w2;
{
    BPNN *newnet;

    newnet = bpnn_internal_create(n_in, n_hidden, n_out);

    if (w1 != NULL && w2 != NULL)
    {
        copy_weights(newnet->input_weights, w1, n_hidden, n_in);
        copy_weights(newnet->hidden_weights, w2, n_out, n_hidden);
    }
    else
    {
        bpnn_randomize_weights(newnet->input_weights, n_in, n_hidden);
        bpnn_randomize_weights(newnet->hidden_weights, n_hidden, n_out);
    }

    // init momentum
    bpnn_zero_weights(newnet->input_prev_weights, n_in, n_hidden);
    bpnn_zero_weights(newnet->hidden_prev_weights, n_hidden, n_out);

    return (newnet);
}


void bpnn_free(net)
BPNN *net;
{
    int n1, n2, i;

    //if (net == NULL) return;

    n1 = net->input_n;
    n2 = net->hidden_n;

    free((char *) net->input_units);
    free((char *) net->hidden_units);
    free((char *) net->output_units);

    free((char *) net->hidden_delta);
    free((char *) net->output_delta);
    free((char *) net->target);

    for (i = 0; i <= n1; i++)
    {
        free((char *) net->input_weights[i]);
        free((char *) net->input_prev_weights[i]);
    }
    free((char *) net->input_weights);
    free((char *) net->input_prev_weights);

    for (i = 0; i <= n2; i++)
    {
        free((char *) net->hidden_weights[i]);
        free((char *) net->hidden_prev_weights[i]);
    }
    free((char *) net->hidden_weights);
    free((char *) net->hidden_prev_weights);

    free((char *) net);
}


/**
This functions is the one make the training
It takes:
net: the NN network structure
eta: learning rate
momentum: momentum of learning to make use of previous values
eo: an address to return the error of the output node
eh: an address to return the error of the hidden niodes
**/
void bpnn_train(net, eta, momentum, eo, eh)
BPNN *net;
double eta, momentum, *eo, *eh;
{
    int in, hid, out;
    double out_err, hid_err;

    in = net->input_n;
    hid = net->hidden_n;
    out = net->output_n;

    /*** Feed forward input activations. ***/
    // between i and j (from input to hidden)
    bpnn_layerforward(net->input_units, net->hidden_units, net->input_weights, in, hid);
    // between j and k (from hidden to output)
    bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);

    /*** Compute error on output and hidden units. ***/
    bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
    bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err);

    *eo = out_err;
    *eh = hid_err;

    /*** Adjust input and hidden weights. ***/
    bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid,
        net->hidden_weights, net->hidden_prev_weights, eta, momentum);
    bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in,
        net->input_weights, net->input_prev_weights, eta, momentum);

}


/**
This function take a net and make a feedforward path only
Used in normal using, and not used in the training
**/
void bpnn_feedforward(net)
BPNN *net;
{
    int in, hid, out;

    in = net->input_n;
    hid = net->hidden_n;
    out = net->output_n;

    /*** Feed forward input activations. ***/
    bpnn_layerforward(net->input_units, net->hidden_units, net->input_weights, in, hid);
    bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);

}
