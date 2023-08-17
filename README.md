# PepCNN
Protein-peptide interaction is a very important biological process as it plays role in many cellular processes, but it is also involved in abnormal cellular behaviors which lead to diseases like cancer. Studying these interaction is therefore vital for understanding protein functions as well as discovering of drugs for disease treatment. The physical understanding of the interactions by the use of experimental approach of studying the interactions are laborious, time-consuming, and expensive. In this regard, we have developed a new prediction method called PepCNN which uses structure and sequence-based information from primary protein sequences to predict the peptide-binding residues. The combination of half sphere exposure structural information, position specific scoring matrix and pre-trained transformer language model based sequence information, and convolutional neural network from deep learning resulted in a superior performance compared to the state-of-the-art methods on the two datatsets. 

# Download and Use
There are two ways to use the provided codes for each dataset. 
1. Load the trained PepCNN model
   The result obtained in our work can be replicated by executing dataset1_PepCNN.py script for Dataset1, and dataset2_PepCNN.py script for Dataset2.
2. Retrain the CNN model
