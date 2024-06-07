# code algo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from typing import List, Tuple, Optional

lb = pd.DataFrame()
encode = LabelEncoder()
scale = MinMaxScaler()

# Dataset with image path and its clothes, color description
df = pd.read_csv("/Users/avicenne/Downloads/descriptions_images.csv")


def labelize_label(label: str, data: pd.DataFrame):
    """
    Optimized function to labelize a column of a DataFrame.
    
    Args:
        label (str): The name of the column to labelize.
        data (pd.DataFrame): The DataFrame containing the data.
    
    Returns:
        dict: A dictionary mapping original values to their encoded labels.
    """
    # Initialize the label encoder
    encode = LabelEncoder()
    
    # Fit the label encoder and transform the column values
    column = data[label]
    column_label = encode.fit_transform(column)
    
    # Create a dictionary mapping original values to their encoded labels
    return dict(zip(column, column_label))


### on prend les données sélectionnées depuis la page web et on les labelisent pour les faire correspondre au données
### du dataset labelisé (lb)
### criteres est un dico avec pour chaque label du dataset l'option choisi, ex : (couleur_haut, vert)
### la sortie est : features = noms des labels sélectionnés et need = valeur choisie pour chaque label
def labelize_var(criteres: , Dico):
    need = []
    features = []
    for crit in criteres:
        features.append(crit)
        need.append(Dico[crit][criteres[crit]])
    return features, need

### pour faire la recherche des voisins, on réduit le dataset labelisé (lb) aux labels sélectionnés (X)
### puis on cherche les voisins des valeurs need qui correspondent aux labels des choix de l'utilisateurs de la page web
def inspirations(lb, df, quantite, features, need, person):
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
    
    knn = NearestNeighbors(algorithm = "brute", metric = "euclidean")
    X = lb[features]
    knn = knn.fit(X)
    indices = list(knn.kneighbors([need], quantite, return_distance = False))
    
    sol = []
    for indice in indices[0]:
        file = person[indice]
        sol.append(file)
    return sol



def get_web_input(entrees, quantite, imageList):
    """
    This algo takes the input selected by the user on the web site, and return the id of the
    images that correspond to the input (request)
    """
    ### On crée un dico qui ne contient que les variables sélectionnées par l'utilisateur
    criteres = {}
    for i in entrees:
        if len(i[1]) !=0:
            criteres[i[0]] = i[1]
            
    print(criteres)
    ### on transforme les options choisis en label en se synchronisant avec le dataset labelisé
    ### on renvoie donc le noms des variables sélectionnées et les labels
    
    features, need = labelize_var(criteres, Dico = Dico)
    sol = inspirations(lb, df, int(quantite), features, need, person=person)
    #print("selection algo : ", sol)
    ### renvoie le nom des images avec le .jpeg, pour faciliter l'affichage. On a donc les images correspondant à nos critères
    return sol

