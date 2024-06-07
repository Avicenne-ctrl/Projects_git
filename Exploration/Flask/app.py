from flask import Flask, render_template, request
import algorithme.utilities 
import zipp
import os
import pandas as pd

# Dataset with image path and its clothes, color description
df = pd.read_csv("/Users/avicenne/Downloads/descriptions_images.csv")

app = Flask(__name__)

# the folder where images are stored
Img = "static/images/data"

@app.route("/")
@app.route('/', methods=['GET', 'POST'])
def index():
    
    # we get all the images in the static folder in order to display them in the menu of the web site
    several_images = os.listdir(Img)
    imageList = [Img +'/'+ image for image in several_images if image != ".DS_Store"]
    
    if request.method == 'POST':
        
        # we get the values selected by the user on the web site
        color_tshirt = request.form.get('couleurHaut')
        color_jean = request.form.get('couleurBas')
        type_tshirt = request.form.get('haut')
        type_jean = request.form.get('bas')
        quantite = request.form.get('quantite')
        
        # if not correct selection, we stay at the home page: if no quantity selected or nothing selected
        if len(quantite) == 0 or (len(color_tshirt) + len(color_jean)+len(type_tshirt) + len(type_jean) == 0):
            return render_template('index.html', imageList = imageList)
        
        # the input are stored in a dataset
        entrees = {'couleur_haut': color_tshirt, 'couleur_bas': color_jean,
                   'type_haut': type_tshirt, 'type_bas': type_jean}
        

        # the main algo which will returns the images for the corresponding input
        resultats = algorithme.utilities.get_web_input(entrees, quantite, df)
        
        
        # we load images selected by the main algo, the most similar ones
        file = []
        for result in resultats:
            file.append(os.path.join(Img, result))

        # we display images on the web site
        return render_template('index.html', imageList = file)
    return render_template('index.html', imageList = imageList)

if __name__ == '__main__':
    app.run(debug=True, threaded=False)


