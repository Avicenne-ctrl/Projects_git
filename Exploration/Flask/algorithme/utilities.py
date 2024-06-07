# code algo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from typing import List, Tuple, Optional

def get_labels(df):
    """
    Optimized function to labelize a column of a DataFrame, and save the corresponding label
    
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
    
    Returns:
        dict: A dictionary mapping original values to their encoded labels.
        df: the encoded dataset
    """
    # save id images and the drop
    image_id= df['name']
    df = df.drop(['name'], axis= 1)
    
    encode = LabelEncoder()
    Dico = {}
    for column in df.columns:
        # encode columns
        column_label = encode.fit_transform(df[column])
        df[column]= column_label
        #save encoded labels
        Dico[column] = dict(zip(encode.classes_, encode.transform(encode.classes_)))
        
    return Dico, df, image_id


### on prend les données sélectionnées depuis la page web et on les labelisent pour les faire correspondre au données
### du dataset labelisé (lb)
### criteres est un dico avec pour chaque label du dataset l'option choisi, ex : (couleur_haut, vert)
### la sortie est : features = noms des labels sélectionnés et need = valeur choisie pour chaque label
def get_label_queries(input, Dico):
    """
    This function take the value selected by the user on the web site,
    and labelize its thanks to the Dico which contains the label associated to the value

    Args:
        criteres (List): List of values selected by the user
        Dico (Dict): Dict of columns dataset with label corresponding

    Returns:
        features: 
    """
    label_queries = {}
    for key in input.keys():
        label_queries[key] = Dico[key][input[key]]
        
    return label_queries

### pour faire la recherche des voisins, on réduit le dataset labelisé (lb) aux labels sélectionnés (X)
### puis on cherche les voisins des valeurs need qui correspondent aux labels des choix de l'utilisateurs de la page web
def inspirations(df_labelize, quantite, label_queries, id_images):
    """
    This fonction takes the critere of search on the dataset, and use KNN 
    to return the line of the dataset which corresponde to the request

    Args:
        lb (_type_): _description_
        df (_type_): _description_
        quantite (_type_): _description_
        features (_type_): _description_
        need (_type_): _description_
        person (_type_): _description_

    Returns:
        _type_: _description_
    """
    # we initiate the KNN with the labelize dataset with only the columns corresponding to the choices of the user
    knn = NearestNeighbors(algorithm = "brute", metric = "euclidean")
    df_target = df_labelize[label_queries.keys()]
    knn = knn.fit(df_target)
    
    # the most similar line in the dataset
    indices = list(knn.kneighbors(np.array([list(label_queries.values())]), quantite, return_distance = False))
    sol = [id_images[indice] for indice in indices[0]]
        
    return sol

def get_web_input(entrees, quantite, df):
    """
    This algo takes the input selected by the user on the web site, and return the id of the
    images that correspond to the input (request)
    """
    # dico of label, data labelize and image id list
    Dico, df_labelize, image_id= get_labels(df)
    
    # check the non empty values selected by the user
    new_input = {}
    for key in entrees.keys():
        # we check that there is a value selected
        if len(entrees[key]) !=0:
            new_input[key] = entrees[key]
            
    print(new_input)
    # encode user choice
    label_queries= get_label_queries(new_input, Dico = Dico)
    
    sol = inspirations(df_labelize, int(quantite), label_queries, image_id)

    return sol

