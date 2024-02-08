# PepCNN
Protein-peptide interaction is a very important biological process as it plays role in many cellular processes, but it is also involved in abnormal cellular behaviors which lead to diseases like cancer. Studying these interaction is therefore vital for understanding protein functions as well as discovering of drugs for disease treatment. The physical understanding of the interactions by the use of experimental approach of studying the interactions are laborious, time-consuming, and expensive. In this regard, we have developed a new prediction method called PepCNN which uses structure and sequence-based information from primary protein sequences to predict the peptide-binding residues. The combination of half sphere exposure structural information, position specific scoring matrix and pre-trained transformer language model based sequence information, and convolutional neural network from deep learning resulted in a superior performance compared to the state-of-the-art methods on the two datatsets. 

![Architecture](https://github.com/abelavit/PepCNN/assets/36461816/711066e5-aac9-4e3e-afcd-7223cf544f05)

# Download and Use
There are two ways to use the provided codes for each dataset. 
## 1. Load the trained PepCNN model
   The result obtained in our work can be replicated by executing dataset1_PepCNN.py script for Dataset1, and dataset2_PepCNN.py script for Dataset2. For instance, to obtain the result of PepCNN on dataset1, run the dataset1_PepCNN.py script after downloading the following files by going to this [link](https://figshare.com/projects/Load_protein-peptide_binding_PepCNN_model/176094) (caution: data size is around 1.3GB for each dataset): 
   - model weights: dataset1_best_model_weights.h5
   - training set negative samples: dataset1_Train_Negatives_All.dat
   - training set positive samples: dataset1_Train_Positives.dat
   - testing set: dataset1_Test_Samples.dat
## 2. Train the CNN model
To train the network from scratch, it can be done by executing dataset1_PepCNN_train.py script for Dataset1, and dataset2_PepCNN_train.py script for Dataset2. For instance, to train the network on dataset1, run the dataset1_PepCNN_train.py script after downloading the following files by going to this [link](https://figshare.com/projects/Train_the_CNN_model/176151) (caution: data size is 1.22GB for both datasets): 
   - testing protein sequences: Dataset1_test.tsv
   - protein sequences excluding testing sequences: Dataset1_train.tsv
   - pre-trained transformer embeddings: T5_Features.dat
   - PSSM features: PSSM_Features.dat
   - HSE features: HSE_Features.dat

Package verions:
Python 3.10.12,
Pandas 1.5.3,
Pickle 4.0,
Numpy 1.25.2,
scikit-learn 1.2.2,
Matplotlib 3.7.2,
Tensorflow 2.12.0

