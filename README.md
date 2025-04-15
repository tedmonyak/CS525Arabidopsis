CS 525 Final Project Proposal
Ted Monyak and Jack Forman
Goal
Predict DNA accessibility sites across different Arabidopsis experiments.
Problem Framing:
Given input DNA sequence, predict read coverage as a continuous, quantitative response variable.
Data
Raw Data
Raw read coverage local filepaths (similar to those from Bassenji)
Local filepaths of (BED) peaks (like those from Assignment 3)
Arabidopsis genome
Metadata
Source of biological samples (project ID and Accession)
Plant tissue identifier
Training Data
Chromosomes 1-4 will be randomly split 80/20 into train and validation data. Chromosome 5 will be held out as a test set.
Data Loading
Load in the multiple Arabidopsis experiments read coverage data, and Arabidopsis genome
Generate testing, training, and validation set generators
Use generator objects that will get the one hot encoded training, testing, and validation datasets so that the datasets can be randomized for each run
Output: List of sequences, and coverage map of those sequences
Biological Datasets
Biologically relevant parts of the genome will be curated into datasets to explore how the models perform with known biological functions using annotations. All of these annotations will hopefully be available on ENCODE or other online resources.

Promoter dataset
Enhancer dataset
CTCF dataset 
Model
Architecture
We are proposing 3 different model architectures based on what we have discussed in class:
Basset model
Small input sequence length
3 convolutional filters
Bassenji model
Large input sequence length (10s of kb)
4 convolutional filters + 5 dilated convolutional filters (Arabidopsis has a ~10x smaller genome than humans)
Bassenji model with transformers
Use positional encoding + multi-head attention layer
Hyperparameters
There are various hyperparameters that we aim to experiment with. Since we are predicting a continuous variable with a regression function, we will use a Poisson regression loss function, as done in the Bassenji model. We may look into GPyOpt (https://github.com/SheffieldML/GPyOpt) for hyperparameter optimization, but will likely just experiment with the hyperparameters manually.
Hyper Parameters to Test (not all hyper parameters apply to all networks):
Learning rate
Number of layers
Batch size
Convolutional filter size
Number of convolutional filters
Input dropout rate (to inform performance on noisy data)
Dropout rate
Num. attention heads
Input layer size
Read length
Prediction
Our prediction is that the Bassenji model will be the best performing, followed by the basic Bassenji, followed by the Basset. Since we have already implemented something similar to the Basset model, this will serve as a useful benchmark.
Biological Interpretation
We aim to provide a rigorous interpretation of the biological significance of our model results, taking into account performance across different cell types, optimal read length, optimal convolutional filter size and number of filters, and other relevant aspects of model architecture.
Work Plan
Breakdown of Work
Input data loading and pre-processing: Work together

HPC setup: Work together

Model Implementation: Both Independently

Hyperparameter tuning: Work independently and compare

Timeline
April 14: Meet with Asa
April 15: Submit Proposal
April 18: Finish Data input pipeline
April 25: HPC setup
April 25: Implementation of Basset and Bassenji Models
May 2: Attention model implementation
May 9: Biological conclusions
May 9: Submission
