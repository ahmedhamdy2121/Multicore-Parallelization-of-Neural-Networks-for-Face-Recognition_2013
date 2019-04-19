# Multicore-Parallelization-of-Neural-Networks-for-Face-Recognition
Multicore Parallelization of Neural Networks for Face Recognition, a neural network built on top of CMU code and has been updated, enhanced, and parallelized. (2013)

Report: https://1drv.ms/b/s!AjpxhCM4OMFHg_APRliCaNCRj-kvBg

This folder contains:

- Samples folder: which has 96 images for many people.

- all.txt: it has list of all the samples. No actual use for such file.

- Source Code:
	- backprop.c
	- imagenet.c
	- pgmimage.c
	- main.c
	- backprop.h
	- imagenet.h
	- pgmimage.h

- Input files:
##############
- samples.txt: this file will be used in series training, it has the list of training samples.
	
- samples1.txt: this file will be used in parallel training for thread 1, it has the list of training samples.
	
- samples2.txt: this file will be used in parallel training for thread 2, it has the list of training samples.
	
- test.txt: this file will be used in both series and parallel training, it has the list of images that will be used in the validation process
	
- Output files:
###############
- Net_hidden_initial_weights.txt: output of series training
- Net_hidden_final_weights.txt: output of series training
- Net_input_initial_weights.txt: output of series training
- Net_input_final_weights.txt: output of series training

- Net1_hidden_initial_weights.txt: output of parallel training, thread 1
- Net1_hidden_final_weights.txt: output of parallel training, thread 1
- Net1_input_initial_weights.txt: output of parallel training, thread 1
- Net1_input_final_weights.txt: output of parallel training, thread 1

- Net2_hidden_initial_weights.txt: output of parallel training, thread 2
- Net2_hidden_final_weights.txt: output of parallel training, thread 2
- Net2_input_intial_weights.txt: output of parallel training, thread 2
- Net2_input_final_weights.txt: output of parallel training, thread 2

- _merged_hidden_weights.txt: output of parallel training, after merging weights of all threads
- _merged_input_weights.txt: output of parallel training, after merging weights of all threads
