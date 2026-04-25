# --- APPLICATION PRINCIPALE (BACKEND) ---
# Ce fichier fait le lien entre tes données (CSV/Modèles) et tes pages web (HTML)

from matplotlib.pylab import cast
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import urllib.request
import pickle

app = Flask(__name__)

# --- 1. CHARGEMENT DES DONNÉES CSV ---
# C'est ici que l'on utilise le fichier CSV que tu as préparé !
def create_similarity():
    # Remplace 'main_data.csv' par le nom de ton fichier final généré dans tes Notebooks
    data = pd.read_csv('main_data.csv') 
    
    # On crée la matrice pour comparer les films entre eux
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    
    # On calcule le score de similarité (Cosine Similarity)
    similarity = cosine_similarity(count_matrix)
    return data, similarity

# --- 2. FONCTION DE RECOMMANDATION ---
def rcmd(m):
    m = m.lower()
    try:
        data.head()
        similarity.shape
    except:
        data, similarity = create_similarity()
    
    # Si le film n'est pas dans notre CSV
    if m not in data['movie_title'].unique():
        return 'Désolé! Ce film n\'est pas dans notre base de données.'
    else:
        # On cherche les 10 films les plus proches
        i = data.loc[data['movie_title']==m].index[0]
        lst = list(enumerate(similarity[i]))
        lst = sorted(lst, key = lambda x:x[1] ,reverse=True)
        lst = lst[1:11] 
        
        l = []
        for i in range(len(lst)):
            a = lst[i][0]
            l.append(data['movie_title'][a])
        return l

# --- 3. ROUTES WEB ---

@app.route("/")
@app.route("/home")
def home():
    # Au chargement de l'accueil, on lit le CSV pour envoyer la liste des films à la barre de recherche
    data = pd.read_csv('main_data.csv')
    suggestions = list(data['movie_title'].str.capitalize())
    return render_template('home.html', suggestions=suggestions)

@app.route("/recommend", methods=["GET"])
def recommend():
    # On récupère le film tapé par l'utilisateur
    movie = request.args.get('movie')
    
    if not movie:
        return render_template('home.html')

    # 1. On récupère les recommandations depuis notre fonction
    rcmd_movies = rcmd(movie)
    
    # Si le film n'existe pas
    if type(rcmd_movies) == type('string'):
        return render_template('home.html', error="Film introuvable")

    # --- AJOUT : Récupération des infos depuis ton CSV ---
    data = pd.read_csv('main_data.csv')
    movie_info = data[data['movie_title'].str.lower() == movie.lower()]
    
    if not movie_info.empty:
        genre = movie_info.iloc[0]['genres']
        director = movie_info.iloc[0]['director_name']
        cast = movie_info.iloc[0]['actor_1_name']
    else:
        genre = director = cast = "Non disponible"
    # -----------------------------------------------------

    # 2. Récupération des affiches et détails via l'API TMDB
    api_key = '9b614336ce925d3c20eda0a366586464' 
    url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={urllib.parse.quote(movie)}"
    
    try:
        response = urllib.request.urlopen(url)
        json_data = json.loads(response.read())
        movie_details = json_data['results'][0]
        poster = f"https://image.tmdb.org/t/p/original{movie_details['poster_path']}"
        overview = movie_details['overview']
    except:
        poster = "https://via.placeholder.com/500x750?text=Affiche+Non+Trouvée"
        overview = "Description non disponible."

    # 3. Récupération des affiches pour les films recommandés
    movie_cards = {}
    for recommended_movie in rcmd_movies:
        url_rec = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={urllib.parse.quote(recommended_movie)}"
        try:
            resp_rec = urllib.request.urlopen(url_rec)
            data_rec = json.loads(resp_rec.read())
            poster_rec = f"https://image.tmdb.org/t/p/original{data_rec['results'][0]['poster_path']}"
            movie_cards[poster_rec] = recommended_movie
        except:
            pass

    # 4. Analyse de sentiment
    try:
        clf = pickle.load(open('nlp_model.pkl', 'rb'))
        vectorizer = pickle.load(open('tranform.pkl','rb'))
        vect = vectorizer.transform([overview])
        prediction = clf.predict(vect)
        sentiment_result = "Les critiques sont globalement Positives 😊" if prediction[0] == 1 else "Les critiques sont globalement Négatives 😞"
    except:
        sentiment_result = "Analyse de sentiment indisponible."

    # On envoie tout au template, y compris les nouvelles variables !
    return render_template('recommend.html', 
                           title=movie.capitalize(), 
                           poster=poster, 
                           overview=overview,
                           sentiment_result=sentiment_result,
                           movie_cards=movie_cards,
                           genre=genre,
                           director=director,
                           cast=cast)
if __name__ == '__main__':
    # Lance le serveur local
    app.run(debug=True)