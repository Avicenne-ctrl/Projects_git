from algorithme.fonctions import *

### il suffit de charger le dataset ici et modifier les choix sélectionnables dans index.html

### ici pour chaque label on affiche les valeurs uniques qui deront être modifiée dans index.html
for i in df:
    print(i, df[i].unique())

Dico = {}


### lb = copie du dataset qui va être labelisé
### Dico = on enregistre les données et leurs labels associés
### Dico est un dico de sous dico
### pour chaque label du dataset on enregistre les labels correspondant au string
### cela permet de mémoriser que vert correspond à 1 et que lors de la recherche
### il faudra convertir le choix de l'utilisateur, vert, à 1 pour retrouver la bonne couleur lors de l'exécution de KNN
person = np.array(df['name'])

for j in df:
    lb[j] = df[j]
    Dico[j] = labelize_label(j, lb)
    
### on labelise les données du dataset = nécessaire pour utilisation de l'algo KNN
for feat in lb:
    lb[feat] = encode.fit_transform(lb[feat])
    
