#include <stdio.h>
#include <omp.h>
#include <stdlib.h> /* for rand() and srand() */
#include <time.h>
#include <math.h>
#include <string.h>
#include "backprop.h"
#include "pgmimage.h"
#include "imagenet.h"

/*** Prototypes ***/
BPNN* backprop_face(IMAGELIST*, int, double, int, int, char*, double**, double**);
void backprob_test(IMAGELIST*, int, BPNN*);
void backprob_test_bulk(IMAGELIST*, BPNN*);
void merge_weights(double**, double**, int, int);
int evaluate_performance(BPNN*);
void printm (double**, int, int, char*);
void printmFile (double**, int, int, char*, char*);
void readFile(double**, int, int, char*);

// 0.4 & 0.1
// 0.3 & 0.3
#define LR 0.4
#define MOM 0.1

int main()
{
    #pragma omp parallel
    printf("Hello from thread %d, nthreads %d\n", omp_get_thread_num(), omp_get_num_threads());

    // =====================================================================================================
    unsigned seed = (unsigned)time(NULL);
    int epochs = 100;
    double error = -1; // default -1
    int hidden = 64;
    int out = 1;
    IMAGELIST *trainlist, *trainlist1, *trainlist2, *testlist;
    IMAGE *iimg;
    int train_n, imgsize;

    /*** Parallel enable flag ***/
    int parallel = 2; // 1 thread

    /*** Read from file flag ***/
    int readFromFile = 0;
    // =====================================================================================================

    /*** Create imagelists ***/
    trainlist = imgl_alloc();
    trainlist1 = imgl_alloc();
    trainlist2 = imgl_alloc();
    testlist = imgl_alloc();

    /*** Load images ***/
    imgl_load_images_from_textfile(testlist, "test.txt");

    // printing the count
    printf("%d images in test set\n", testlist->n);

    /*** init the NN ***/
    bpnn_initialize(seed);

    /***
    or read weights from file
    To initialize the file with random number first, use serial training first, as the readFile will read from:
    Net_initial_input_weights.txt & Net_initial_hidden_weights.txt
    ***/
    BPNN *temp = NULL;
    if (readFromFile == 1)
    {
        int input = 960;
        temp = bpnn_create(input, hidden, out, NULL, NULL);

        // read weights from input to hidden
        readFile(temp->input_weights, temp->input_n, temp->hidden_n, "Net_input_initial_weights.txt");

        // read weights from hidden to output
        readFile(temp->hidden_weights, temp->hidden_n, temp->output_n, "Net_hidden_initial_weights.txt");

        printf("Reading weights from file done! \n");

        //printmFile(temp->input_weights, temp->input_n, temp->hidden_n, "input_weights.txt", "Test");
        //printmFile(temp->hidden_weights, temp->hidden_n, temp->output_n, "hidden_weights.txt", "Test");
    }

    //printf("random number: %lf \n", drnd());
    //printf("random number: %lf \n", dpn1());
    //printf("random number: %lf \n", dpn2());

    //exit(0);

    /*** Creating the main Neural Network ***/
    BPNN *net;
    /*** handle threading ***/
    if (parallel == 1)
    {
        /*** Load images ***/
        imgl_load_images_from_textfile(trainlist, "samples.txt"); // without threading

        // printing the count
        printf("%d images in training set \n", trainlist->n);

        printf("\n");

        /*** The training call ***/
        /*** Series ***/ //86s for 1000 epochs and 128 hidden neurons
        if (readFromFile == 1)
            net = backprop_face(trainlist, epochs, error, hidden, out, "Net", temp->input_weights, temp->hidden_weights);
        else
            net = backprop_face(trainlist, epochs, error, hidden, out, "Net", NULL, NULL);

    }
    else if (parallel == 2)
    {
        /*** Load images ***/
        imgl_load_images_from_textfile(trainlist1, "samples1.txt"); // for thread 0
        imgl_load_images_from_textfile(trainlist2, "samples2.txt"); // for thread 1

        // printing the count
        printf("%d images in training set 1\n", trainlist1->n);
        printf("%d images in training set 2\n", trainlist2->n);

        printf("\n");
        BPNN *net1;

        /*** Creating template for the parallel training ***/
        if (readFromFile == 0)
        {
            train_n = trainlist1->n;
            if (train_n > 0)
            {
                printf("Creating template network for parallel training \n");
                iimg = trainlist1->list[0];
                imgsize = ROWS(iimg) * COLS(iimg);

                /* bthom ===========================
                    make a net with:
                    imgsize inputs, 4 hiden units, and 1 output unit
                */
                temp = bpnn_create(imgsize, hidden, out, NULL, NULL);
            }
            else
            {
                printf("Parallel block: need some images to train on \n");
                return 0;
            }
        }

        /*** The training call ***/
        /*** Parallel ***/ // 50s for 1000 epochs and 128 hidden neurons
        #pragma omp parallel
        {
            int thr = omp_get_thread_num();
            if (thr == 0)
                net = backprop_face(trainlist1, epochs, error, hidden, out, "Net1", temp->input_weights, temp->hidden_weights);
            else
                net1 = backprop_face(trainlist2, epochs, error, hidden, out, "Net2", temp->input_weights, temp->hidden_weights);
        }

        /*** Merging the weights from the two NN ***/
        printf("Merging after threading... \n");
        #pragma omp parallel
        {
            int thr = omp_get_thread_num();
            if (thr == 0)
                merge_weights(net->input_weights, net1->input_weights, net->hidden_n, net->input_n);
            else
                merge_weights(net->hidden_weights, net1->hidden_weights, net->output_n, net->hidden_n);
        }

        printf("New input weights can be found in --> _merged_input_weights.txt \n");
        printmFile(net->input_weights, net->input_n, net->hidden_n, "merged_input_weights.txt", "");
        printf("New hidden weights can be found in --> _merged_hidden_weights.txt \n");
        printmFile(net->hidden_weights, net->hidden_n, net->output_n, "merged_hidden_weights.txt", "");
    }

    printf("\n");
    /*** testing the network ***/

    if (net != NULL) backprob_test_bulk(testlist, net);
    /*
    if (net != NULL) backprob_test(testlist, 0, net);
    if (net != NULL) backprob_test(testlist, 1, net);
    if (net != NULL) backprob_test(testlist, 2, net);
    if (net != NULL) backprob_test(testlist, 3, net);
    if (net != NULL) backprob_test(testlist, 4, net);
    //*/

    return 0;
}


/*** ======================= My Functions ======================= ***/

/***
This function is the responsible of the training at all.
tranlist: list of images to train with
epochs: number of epochs
error: optional parameter with default -1, but if passed with another value,
    it will be a stopping criteria (override number of epochs)
hidden: number of hidden neurons
out: number of output neurons
netname: the Neural Network name
w1: input weight matrix to copy weights from if not null (reuse the BPNN)
w2: hidden weight matrix to copy weights from if not null (reuse the BPNN)
***/
BPNN* backprop_face(trainlist, epochs, error, hidden, out, netname, w1, w2)
IMAGELIST *trainlist;
int epochs, hidden, out;
char *netname;
double **w1, **w2, error;
{
    IMAGE *iimg;
    BPNN *net;
    int train_n, epoch, i, imgsize;
    double out_err, hid_err, sumerr;

    train_n = trainlist->n;

    /************************* Creating new NN ********************************/
    if (train_n > 0)
    {
        printf("Creating new network '%s'\n", netname);
        iimg = trainlist->list[0];
        imgsize = ROWS(iimg) * COLS(iimg);

        /* bthom ===========================
            make a net with:
            imgsize inputs, 4 hiden units, and 1 output unit
        */
        net = bpnn_create(imgsize, hidden, out, w1, w2);
    }
    else
    {
        printf("Need some images to train on\n");
        return NULL;
    }

    // synch the threads here
    #pragma omp barrier

    #pragma omp critical
    {
        /*** Print BPNN information ***/
        printf("Back Propagation Neural Network: \n================================\n");
        printf("Structure: %dx%dx%d network '%s' \n", net->input_n, net->hidden_n, net->output_n, netname);
        printf("Initial input layer weights %dx%d matrix: --> print into file %s_input_initial_weights.txt \n", net->input_n, net->hidden_n, netname);
        printmFile(net->input_weights, net->input_n, net->hidden_n, "input_initial_weights.txt", netname);
        printf("Initial hidden layer weights %dx%d matrix: --> print into file %s_hidden_initial_weights.txt \n", net->hidden_n, net->output_n, netname);
        printmFile(net->hidden_weights, net->hidden_n, net->output_n, "hidden_initial_weights.txt", netname);

        printf("\n"); fflush(stdout);
    }

    // synch the threads here
    #pragma omp barrier

    if (epochs > 0)
    {
        printf("Training underway (going to %d epochs)\n", epochs);
        fflush(stdout);
    }

    /************************* Training the NN ********************************/
    for (epoch = 1; epoch <= epochs; epoch++)
    {
        printf("%3d --> ", epoch); fflush(stdout);

        max_weight_change_err = DBL_MIN;
        sumerr = 0.0;
        for (i = 0; i < train_n; i++)
        {
            /** Set up input units on net with image i **/
            load_input_with_image(trainlist->list[i], net);

            /** Set up target vector for image i **/
            load_target(trainlist->list[i], net);

            /** Run backprop, learning rate 0.3, momentum 0.3 **/
            bpnn_train(net, LR, MOM, &out_err, &hid_err);

            sumerr += (out_err + hid_err);
        }

        // printing max weight change error in each epoch
        printf("Max weight change error: %lf \n", max_weight_change_err);

        printf("        total error: %lf \n", sumerr); fflush(stdout);

        if ( error != -1 && (max_weight_change_err - error) < error)
        {
            printf("Error boundary reached! Breaking the training loop \n");
            break;
        }
    }
    // END OF Training

    printf("\n"); fflush(stdout);

    // synch the threads here
    #pragma omp barrier

    #pragma omp critical
    {
        /*** Print Status Information and final performance ***/
        printf("Finishing %dx%dx%d network to '%s'\n", net->input_n, net->hidden_n, net->output_n, netname);
        printf("Final result for Back Propagation Neural Network: \n=================================================\n");
        printf("Structure: %dx%dx%d network '%s' \n", net->input_n, net->hidden_n, net->output_n, netname);
        printf("Final input layer weights %dx%d matrix: --> print into file %s_input_final_weights.txt \n", net->input_n, net->hidden_n, netname);
        printmFile(net->input_weights, net->input_n, net->hidden_n, "input_final_weights.txt", netname);
        printf("Final hidden layer weights %dx%d matrix: --> print into file %s_hidden_final_weights.txt \n", net->hidden_n, net->output_n, netname);
        printmFile(net->hidden_weights, net->hidden_n, net->output_n, "hidden_final_weights.txt", netname);
        printf("Max weight change error: %lf \n", max_weight_change_err);
        printf("Output Error Delta: %lf \n", net->output_delta[1]);

        printf("\n"); fflush(stdout);
    }

    return (net);
}


/***
This method will take the avg of weight and store it in w.
***/
void merge_weights(w, w1, ncurrent, nprev)
double **w, **w1;
int ncurrent, nprev;
{
    int j, k;
    double avg;

    // collapse(2)
    #pragma omp for collapse(2)
    for (k = 1; k <= ncurrent; k++)
    {
        for (j = 0; j <= nprev; j++)
        {
            avg = (w[j][k] + w1[j][k]) / 2.0;
            w[j][k] = avg;
        }
    }
}

/***
This function takes list of images and make a loop call on that list to the validation function
This function validate the network against a list of samples
testlist: list of images to train with
net: the neural network needed to be validated
***/
void backprob_test_bulk(testlist, net)
IMAGELIST *testlist;
BPNN *net;
{
    int i, train_n;

    train_n = testlist->n;

    for (i = 0; i < train_n; i++)
    {
        backprob_test(testlist, i, net);
    }
}


/***
This function used to validate the network against one sample only
testlist: list of images to train with
imagenumber: the image location in the image list
net: the neural network needed to be validated
***/
void backprob_test(testlist, imagenumber, net)
IMAGELIST *testlist;
int imagenumber;
BPNN *net;
{
    int res;
    /*** Load the image into the input layer. ***/
    load_input_with_image(testlist->list[imagenumber], net);

    /*** Run the net on this input. ***/
    bpnn_feedforward(net);

    /*** Set up the target vector for this image. ***/
    load_target(testlist->list[imagenumber], net);

    /*** See if it got it right. ***/
    res = evaluate_performance(net);

    /*** Correct result ***/
    if (res)
    {
        printf("Test PASSED: %s \n", NAME(testlist->list[imagenumber]));
    }
    /*** Incorrect result ***/
    else
    {
        printf("XXXXX Test FAILED: %s \n", NAME(testlist->list[imagenumber]));
    }

    fflush(stdout);
}


/***
This function for internal use. It check the actual output against the target output and then returns either true or false
***/
int evaluate_performance(net)
BPNN *net;
{
	/*** If the target unit is on... ***/
	if (net->target[1] > 0.5)
	{
		/*** If the output unit is on, then we correctly recognized me! ***/
		if (net->output_units[1] > 0.5)
			return (1);
		/*** otherwise, we didn't think it was me... ***/
		else
			return (0);

	/*** Else, the target unit is off... ***/
	}
	else
	{

		/*** If the output unit is on, then we mistakenly thought it was me ***/
		if (net->output_units[1] > 0.5)
			return (0);

		/*** else, we correctly realized that it wasn't me ***/
     	else
			return (1);
	}

}


void printm (double **mat, int n, int m, char *s)
{
    int i,j;
 	printf(">>>> Print: %s \n", s);

 	/***
    First column of the weights matrix is always the same, because it is redundant, we don't print it
    Bias is the first row of the weights matrix, that is why we began to print from row: 0 and col: 1
    ***/
    for(i = 0; i <= n; i++)
	{
      	for(j = 1; j <= m; j++)
		{
        	printf("%3.4lf\t" , mat[i][j]);
    	}
    	printf("\n\n");
   }
}

/***
Writing weights matrix to file
***/
void printmFile (double **mat, int n, int m, char *s, char *netname)
{
    int i,j;
    char filename[256];
    FILE *fp;

    strcpy (filename, netname);
    strcat (filename, "_");
    strcat (filename, s);

    fp = fopen(filename, "w+");
    if(fp == NULL)
    {
        printf("printmFile: Couldn't write into '%s'\n", filename);
        return;
    }

    /***
    First column of the weights matrix is always the same, because it is redundant, we don't print it
    Bias weight is the first row of the weights matrix, that is why we began to print from row: 0 and col: 1
    ***/
    for(i = 0; i <= n; i++)
	{
      	for(j = 1; j <= m; j++)
		{
        	fprintf(fp, "%3.5lf\t" , mat[i][j]);
    	}
    	fprintf(fp, "\n\n");
   }

   fclose(fp);
}


/***
Reading weights matrix from file
***/
void readFile(double **mat, int n, int m, char *s)
{
    int i,j;
    FILE *fp;

    fp = fopen(s, "r");
    if(fp == NULL)
    {
        printf("readFile: Couldn't open '%s'\n", s);
        return;
    }

    /***
    First column of the weights matrix is always the same, because it is redundant, we don't print it
    Bias weight is the first row of the weights matrix, that is why we began to print from row: 0 and col: 1
    ***/
    for(i = 0; i <= n; i++)
	{
      	for(j = 1; j <= m; j++)
		{
        	fscanf(fp, "%lf" , &mat[i][j]);
    	}
   }

   fclose(fp);
}
