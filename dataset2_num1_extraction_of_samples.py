# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 10:18:50 2023

@author: abelac
"""
import pickle
import pandas as pd
import numpy as np
import math


# function to extract samples
def peptide_feat(window_size, Protein_seq, Feat, j): # funtion to extract peptide length and feature based on window size
    
    if (j - math.ceil(window_size/2)) < -1: # not enough amino acid at N terminus to form peptide
        peptide1 = Protein_seq[j:math.floor(window_size/2)+j+1] # +1 since the stop value for slicing is exclusive
        peptide2 = Protein_seq[j+1:math.floor(window_size/2)+j+1] # other peptide half but excluding the central amino acid
        peptide = peptide2[::-1] + peptide1
        
        feat1 = Feat[j:math.floor(window_size/2)+j+1] # +1 since the stop value for slicing is exclusive
        feat2 = Feat[j+1:math.floor(window_size/2)+j+1] # other peptide half but excluding the central amino acid
        final_feat = np.concatenate((feat2[::-1], feat1))
        mirrored = 'Yes'
        
    elif ((len(Protein_seq) - (j+1)) < (math.floor(window_size/2))): # not enough amino acid at C terminus to form peptide
        peptide1 = Protein_seq[j-math.floor(window_size/2):j+1]
        peptide2 = Protein_seq[j-math.floor(window_size/2):j]
        peptide = peptide1 + peptide2[::-1]
        
        feat1 = Feat[j-math.floor(window_size/2):j+1]
        feat2 = Feat[j-math.floor(window_size/2):j]
        final_feat = np.concatenate((feat1, feat2[::-1]))
        mirrored = 'Yes'
        
    else:
        peptide = Protein_seq[j-math.floor(window_size/2):math.floor(window_size/2)+j+1]
        final_feat = Feat[j-math.floor(window_size/2):math.floor(window_size/2)+j+1]
        mirrored = 'No'
        
    return peptide, final_feat, mirrored




# Prepare data
Dataset_test_tsv = pd.read_table("Dataset2_test.tsv")
Dataset_train_tsv = pd.read_table("Dataset2_train.tsv")

file = open("T5_Features.dat",'rb')
Proteins = pickle.load(file)
file = open("HSE_Features.dat",'rb')
Proteins2 = pickle.load(file)
file = open("PSSM_Features.dat",'rb')
Proteins3 = pickle.load(file)

column_headers = list(Proteins.columns.values)
DatasetTestProteins = pd.DataFrame(columns = column_headers)
DatasetTestProteins2 = pd.DataFrame(columns = column_headers)
DatasetTestProteins3 = pd.DataFrame(columns = column_headers)

matching_index = 0
for i in range(len(Dataset_test_tsv)):
    for j in range(len(Proteins)):
        if (Dataset_test_tsv['seq'][i].upper() == Proteins['Prot_seq'][j].upper()):           
            DatasetTestProteins.loc[matching_index] = Proteins.loc[j]
            matching_index += 1
            break
matching_index = 0
for i in range(len(Dataset_test_tsv)):
    for j in range(len(Proteins2)):
        if (Dataset_test_tsv['seq'][i].upper() == Proteins2['Prot_seq'][j].upper()):
            DatasetTestProteins2.loc[matching_index] = Proteins2.loc[j]
            matching_index += 1
            break
matching_index = 0
for i in range(len(Dataset_test_tsv)):
    for j in range(len(Proteins3)):
        if (Dataset_test_tsv['seq'][i].upper() == Proteins3['Prot_seq'][j].upper()):
            DatasetTestProteins3.loc[matching_index] = Proteins3.loc[j]
            matching_index += 1
            break   
            
DatasetTrainProteins = pd.DataFrame(columns = column_headers)
DatasetTrainProteins2 = pd.DataFrame(columns = column_headers)
DatasetTrainProteins3 = pd.DataFrame(columns = column_headers)

matching_index = 0
for i in range(len(Dataset_train_tsv)):
    for j in range(len(Proteins)):
        if (Dataset_train_tsv['seq'][i].upper() == Proteins['Prot_seq'][j].upper()):       
            DatasetTrainProteins.loc[matching_index] = Proteins.loc[j]
            matching_index += 1
            break

matching_index = 0
for i in range(len(Dataset_train_tsv)):
    for j in range(len(Proteins2)):
        if (Dataset_train_tsv['seq'][i].upper() == Proteins2['Prot_seq'][j].upper()):
            DatasetTrainProteins2.loc[matching_index] = Proteins2.loc[j]
            matching_index += 1
            break    
matching_index = 0
for i in range(len(Dataset_train_tsv)):
    for j in range(len(Proteins3)):
        if (Dataset_train_tsv['seq'][i].upper() == Proteins3['Prot_seq'][j].upper()):
            DatasetTrainProteins3.loc[matching_index] = Proteins3.loc[j]
            matching_index += 1
            break 

# generate samples for Test protein sequences
column_names = ['Code','Protein_len','Seq_num','Amino_Acid','Position','Label','Peptide','Mirrored','Feature','Prot_positives']
Test_Samples = pd.DataFrame(columns = column_names)
#Test_Negatives = pd.DataFrame(columns = column_names)

Pos_index = 0
Neg_index = 0
window_size = 1 # -0 to +0
seq_num = 0

# extract feature and peptide for all sites 
for i in range(len(DatasetTestProteins)):
    Protein_seq = DatasetTestProteins['Prot_seq'][i]
    Feat = DatasetTestProteins['Feat'][i] # transpose the feature matrix
    Feat2 = DatasetTestProteins2['Feat'][i]
    Feat3 = DatasetTestProteins3['Feat'][i]
    positive_counts = DatasetTestProteins['Prot_label'][i].count('1')
    
    seq_num += 1
    for j in range(len(Protein_seq)): # go through the protein seq
        
        A_sample = pd.DataFrame(columns = column_names) # create new dataframe using same column names. This dataframe will just have 1 entry.
        A_sample.loc[0,'Code'] = DatasetTestProteins['Prot_name'][i] # store the protein name
        A_sample.loc[0,'Protein_len'] = DatasetTestProteins['Prot_len'][i] # store the protein length
        A_sample.loc[0,'Label'] = DatasetTestProteins['Prot_label'][i][j]
        A_sample.loc[0,'Prot_positives'] = positive_counts
        A_sample.loc[0,'Amino_Acid'] = Protein_seq[j] # store the amino acid 
        A_sample.loc[0,'Position'] = j # store the position of the amino acid
        
        peptide, T5_feat, mirrored = peptide_feat(window_size, Protein_seq, Feat, j) # call the function to extract peptide and feature based on window size
        peptide, HSE_feat, mirrored = peptide_feat(window_size, Protein_seq, Feat2, j)
        peptide, PSSM_feat, mirrored = peptide_feat(window_size, Protein_seq, Feat3, j)
        
        A_sample.loc[0,'Peptide'] = peptide
        Feat_vec = np.concatenate((T5_feat.mean(0),HSE_feat.flatten(),PSSM_feat.flatten()))
        A_sample.loc[0,'Feature'] = np.float32(Feat_vec)
        A_sample.loc[0,'Seq_num'] = seq_num
        A_sample.loc[0,'Mirrored'] = mirrored
        
        
        Test_Samples = pd.concat([Test_Samples, A_sample], ignore_index=True, axis=0)
            
                  
    print('Test Protein ' + str(i+1) + ' out of ' + str(len(DatasetTestProteins)))
print('Number of Proteins in Test: ' + str(len(DatasetTestProteins)))
print('Number of samples in Test: ' + str(len(Test_Samples)))

pickle.dump(Test_Samples,open("dataset2_Test_Samples_rerun.dat","wb"))

# generate samples for Train protein sequences
Train_Positives = pd.DataFrame(columns = column_names)
Train_Negatives_All = pd.DataFrame(columns = column_names)

Pos_index = 0
Neg_index = 0
seq_num = 0

# extract feature and peptide for all sites 
for i in range(len(DatasetTrainProteins)):
    Protein_seq = DatasetTrainProteins['Prot_seq'][i]
    Feat = DatasetTrainProteins['Feat'][i] # transpose the feature matrix
    Feat2 = DatasetTrainProteins2['Feat'][i]
    Feat3 = DatasetTrainProteins3['Feat'][i]
    positive_counts = DatasetTrainProteins['Prot_label'][i].count('1')
    
    seq_num += 1
    for j in range(len(Protein_seq)): # go through the protein seq
            
        A_sample = pd.DataFrame(columns = column_names) # create new dataframe using same column names. This dataframe will just have 1 entry.
        A_sample.loc[0,'Code'] = DatasetTrainProteins['Prot_name'][i] # store the protein name
        A_sample.loc[0,'Protein_len'] = DatasetTrainProteins['Prot_len'][i] # store the protein length
        A_sample.loc[0,'Label'] = DatasetTrainProteins['Prot_label'][i][j]
        A_sample.loc[0,'Prot_positives'] = positive_counts
        A_sample.loc[0,'Amino_Acid'] = Protein_seq[j] # store the amino acid 
        A_sample.loc[0,'Position'] = j # store the position of the amino acid
            
        peptide, T5_feat, mirrored = peptide_feat(window_size, Protein_seq, Feat, j) # call the function to extract peptide and feature based on window size
        peptide, HSE_feat, mirrored = peptide_feat(window_size, Protein_seq, Feat2, j)
        peptide, PSSM_feat, mirrored = peptide_feat(window_size, Protein_seq, Feat3, j)
            
        A_sample.loc[0,'Peptide'] = peptide
        Feat_vec = np.concatenate((T5_feat.mean(0),HSE_feat.flatten(),PSSM_feat.flatten()))
        A_sample.loc[0,'Feature'] = np.float32(Feat_vec)        
        A_sample.loc[0,'Seq_num'] = seq_num
        A_sample.loc[0,'Mirrored'] = mirrored
                        
        if A_sample.loc[0,'Label'] == '1':
            Train_Positives = pd.concat([Train_Positives, A_sample], ignore_index=True, axis=0)
               
        else: 
            Train_Negatives_All = pd.concat([Train_Negatives_All, A_sample], ignore_index=True, axis=0)
      
            
    print('Train Protein ' + str(i+1) + ' out of ' + str(len(DatasetTrainProteins)))
print('Number of Proteins in Train: ' + str(len(DatasetTrainProteins)))
print('Feature vector size: ' + str(Test_Samples['Feature'][0].shape))
print('Num of Train Positives: ' + str(len(Train_Positives)))
print('Num of Train Negatives (All): ' + str(len(Train_Negatives_All)))
pickle.dump(Train_Positives,open("dataset2_Train_Positives_rerun.dat","wb"))
pickle.dump(Train_Negatives_All,open("dataset2_Train_Negatives_All_rerun.dat","wb"))


