# %% [code] {"execution":{"iopub.status.busy":"2024-05-18T17:07:34.744205Z","iopub.execute_input":"2024-05-18T17:07:34.744636Z","iopub.status.idle":"2024-05-18T17:07:38.105671Z","shell.execute_reply.started":"2024-05-18T17:07:34.744603Z","shell.execute_reply":"2024-05-18T17:07:38.104531Z"}}
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
import matplotlib
import seaborn as sns
import webcolors
from sklearn.neighbors import NearestNeighbors
from typing import List, Tuple, Optional


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
    
    
"""
Functions I used for the Flask project, but you can re use for KNN search
"""

def get_labels(df):
    """
    Function to labelize a column of a DataFrame, and save the corresponding label
    
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
    
    Returns:
        Dico_encoded: A dictionary mapping original values to their encoded labels.
        df: the encoded dataset
    """
    
    encode = LabelEncoder()
    Dico_encoded = {}
    for column in df.columns:
        # encode columns
        column_label = encode.fit_transform(df[column])
        df[column]= column_label
        #save encoded labels
        Dico_encoded[column] = dict(zip(encode.classes_, encode.transform(encode.classes_)))
        
    return Dico_encoded, df


### on prend les données sélectionnées depuis la page web et on les labelisent pour les faire correspondre au données
### du dataset labelisé (lb)
### criteres est un dico avec pour chaque label du dataset l'option choisi, ex : (couleur_haut, vert)
### la sortie est : features = noms des labels sélectionnés et need = valeur choisie pour chaque label
def get_label_queries(reference_case, Dico):
    """
    This function take the case we want similarities from the dataset,
    and labelize it thanks to the Dico which contains the label associated to the value

    Args:
        criteres (List): List of values selected by the user -> {'column1': 'value1', 'column2': 'value2'...}
        Dico (Dict): Dict of columns dataset with label corresponding -> {'column1': {'value1': 0, 'value2': 1},
                                                                          'column2': {'value1': 2, 'value2': 1}...}

    Returns:
        label_queries: dict of column concern with its value labelized
    """
    
    reference_case_encoded = {}
    for key in reference_case.keys():
        reference_case_encoded[key] = Dico[key][reference_case[key]]
        
    return reference_case_encoded

### pour faire la recherche des voisins, on réduit le dataset labelisé (lb) aux labels sélectionnés (X)
### puis on cherche les voisins des valeurs need qui correspondent aux labels des choix de l'utilisateurs de la page web
def get_knn_similarities(df_labelize, nb_similarities, reference_case_encoded, List_index):
    """
    This fonction takes the critere of search on the dataset, and use KNN
    to return the line of the dataset which corresponde to the request

    Args:
        df_labelize (pd.DataFrame): the labelized dataframe where we look for similar case
        nb_similarities (int): the number of similar cases we want
        reference_case_encoded (Dict): dict of columns we search similarities with it encoded value corresponding
        List_index (list): the list of index were we could find the similar case
        
    Returns:
        sol: list of index corresponding to similar case
    """
    # we initiate the KNN with the labelize dataset with only the columns corresponding to the choices of the user
    knn = NearestNeighbors(algorithm = "brute", metric = "euclidean")
    df_target = df_labelize[reference_case_encoded.keys()]
    knn = knn.fit(df_target)
    
    # the most similar line in the dataset
    distance, indices = list(knn.kneighbors(np.array([list(reference_case_encoded.values())]), nb_similarities, return_distance = True))
    sol = [List_index[indice] for indice in indices[0]]
        
    return sol

def get_search_by_knn(reference_case, nb_similarities, df, List_index):
    """
    This algo takes the input selected by the user on the web site, and return the id of the
    images that correspond to the input (request)
    
    Args:
        reference_case (Dict): the case we want to find similarities, with columns and value corresponding
        nb_similarities (int): the number of similar case the function will returns
        df (pd.DataFrame): the non labelize DataFrame
        List_index (list): list of the index of the dataframe df
        
    returns:
        sol: list of index corresponding to similar case
    
    """
    
    # first we remove id columns which must be specified as List
    df = df.drop(['name'], axis= 1)
    # dico of label, data labelize and image id list
    Dico, df_labelize= get_labels(df)
    
    # check the non empty values selected by the user
    reference_case_encoded = {}
    for key in reference_case.keys():
        # we check that there is a value selected
        if len(reference_case[key]) !=0:
            reference_case_encoded[key] = reference_case[key]
            
    print("search of similare case to:", reference_case_encoded)
    # encode user choice
    label_queries= get_label_queries(reference_case_encoded, Dico = Dico)
    
    sol = get_knn_similarities(df_labelize, int(nb_similarities), label_queries, List_index)

    return sol






# %% [code]
