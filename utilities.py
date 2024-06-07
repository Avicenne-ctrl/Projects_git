# %% [code] {"execution":{"iopub.status.busy":"2024-05-18T17:07:34.744205Z","iopub.execute_input":"2024-05-18T17:07:34.744636Z","iopub.status.idle":"2024-05-18T17:07:38.105671Z","shell.execute_reply.started":"2024-05-18T17:07:34.744603Z","shell.execute_reply":"2024-05-18T17:07:38.104531Z"}}
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
import matplotlib
import seaborn as sns
import webcolors


""" 
------------------------------------------------------------------------------------------------------------------------------------        
     Function to use KnnImputer on a dataset
------------------------------------------------------------------------------------------------------------------------------------        
"""

## on renvoie la liste des variables à labiler
def need_labelize(data):
    return [key for key in data if data[key].dtype == 'object' or data[key].dtype == 'category']

## on remplace toutes les valeurs maquantes par np.nan pour uniformiser pour pouvoir les retrouver dans le dic des labels
def replace_Null_by_Na(data):
    for key in data:
        if data[key].isna().any():
            #print('missing value : ', key)
            data[key] = data[key].fillna(np.nan)
    return data

def pre_knnimput(train, test):
    
    ## on uniformise les nan pour les valeurs manquantes
    
    train = replace_Null_by_Na(train)
    test = replace_Null_by_Na(test)
    
    ## variables à numériser
    list_need_labelize = need_labelize(train)
    encode = LabelEncoder()
        
        
    ## on labelise en sauvegardant le dico des labels 
    map_label = {}
    
    for lab in train:
        if lab in list_need_labelize:
            print("need labelize : ", lab)
            encode.fit(train[lab])
            train[lab] = encode.transform(train[lab])
            test[lab] = encode.transform(test[lab])
            
            # Récupérer les correspondances entre les labels et les classes d'origine
            label_to_class = dict(zip(encode.classes_, encode.transform(encode.classes_)))
            
            if np.nan in label_to_class:
                ## on sauvegarde la traçabilité des labels
                map_label[lab] = label_to_class[np.nan]
            
            # Afficher les correspondances
            #print(label_to_class)
        
    for key in map_label:
        if key in train:
            train[key] = train[key].replace(map_label[key], np.nan)
        if key in test:
            test[key] = test[key].replace(map_label[key], np.nan)
            
    imputer = KNNImputer()
    
    train = pd.DataFrame(imputer.fit_transform(train), columns = [i for i in train])
    test = pd.DataFrame(imputer.transform(test), columns = [i for i in test])
        
    return train, test

# ------------------------------------------------------------------------------------------------------------------------------------        
# Fonction pour remplacer les valeurs manquantes du dataset de manière automatique (moyenne pour float et 
# valeur majoritaire pour str)
# ------------------------------------------------------------------------------------------------------------------------------------        

    
def fillna_process(data, details):
    """This function takes an dataset and automatically fill na the missing value with the most common
        method : for float missing value --> fillna(mean)
                 for string missing value --> fillna(most_common_value)

    Args:
        data (pd.DataFrame): the dataset we want to fill na
        details (Boolean): if we want to show the details of the process

    Returns:
        data : the final dataset without the missing value
    """
    for key in data:
        if data[key].isna().any():
            if data[key].dtype == 'object':
                if details:
                    print('label à fill : ', key)
                    print('on remplace par la majoritée : ', data[key].value_counts().idxmax())
                
                ## on remplace par la valeur majoritaire
                most_common = data[key].value_counts().idxmax()
                data[key] = data[key].fillna(most_common)

            else:
                if details:
                    print('label à fill : ', key)
                    print('on remplace par la moyenne : ', data[key].mean())
                
                ## on remplace par la moyenne
                data[key] = data[key].fillna(data[key].mean())

    return data
    




# %% [code]
