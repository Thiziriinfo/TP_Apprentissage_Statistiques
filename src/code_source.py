
#%%
# Chargement des bibliothèques 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from svm_source import *
from sklearn import svm
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from time import time



#%%
###############################################  Question 01 ##################################################

# Chargement des données Iris et prétraitement
iris = datasets.load_iris()
X = iris.data
scaler = StandardScaler()  # Normalisation des données
X = scaler.fit_transform(X)
y = iris.target

# On garde uniquement deux classes pour avoir une classification binaire (classe 1 et 2)
X = X[y != 0, :2]  # Prendre seulement 2 caractéristiques pour visualisation
y = y[y != 0]

# Mélanger les données et diviser en train/test
X, y = shuffle(X, y, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)



#%% 
# Noyau linéaire
# GridSearch pour le noyau linéaire avec différents paramètres de régularisation (C)
parameters = {'kernel': ['linear'], 'C': list(np.logspace(-3, 3, 200))}

clf_linear = GridSearchCV(SVC(), parameters, cv=5)  # Recherche des meilleurs paramètres avec validation croisée
clf_linear.fit(X_train, y_train)  # Entraînement du modèle

# Calcul du score (précision) sur les ensembles d'entraînement et de test
print('Meilleurs paramètres pour noyau linéaire:', clf_linear.best_params_)
print('Score de généralisation pour noyau linéaire: train = %.2f, test = %.2f' %
      (clf_linear.score(X_train, y_train), clf_linear.score(X_test, y_test)))

#%% 
# #######################################Question02##############################################################
 
# Noyau polynomial
# GridSearch pour le noyau polynomial avec paramètres C, gamma, et degree
Cs = list(np.logspace(-3, 3, 5))
gammas = 10. ** np.arange(1, 2)
degrees = np.r_[ 1,2, 3]

parameters = {'kernel': ['poly'], 'C': Cs, 'gamma': gammas, 'degree': degrees}

clf_poly = GridSearchCV(SVC(), parameters, cv=5)  # Recherche des meilleurs paramètres pour noyau polynomial
clf_poly.fit(X_train, y_train)  # Entraînement du modèle

# Affichage des meilleurs paramètres et des scores
print('Meilleurs paramètres pour noyau polynomial:', clf_poly.best_params_)
print('Score de généralisation pour noyau polynomial: train = %.2f, test = %.2f' %
      (clf_poly.score(X_train, y_train), clf_poly.score(X_test, y_test)))

#%% 
########### Visualisation des résultats avec frontiere #############

def f_linear(xx):
    """Classificateur pour noyau linéaire"""
    return clf_linear.predict(xx.reshape(1, -1))

def f_poly(xx):
    """Classificateur pour noyau polynomial"""
    return clf_poly.predict(xx.reshape(1, -1))

# Fonction pour afficher la frontière de décision
def frontiere(classifier, X, y, step=50):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, step),
                         np.linspace(y_min, y_max, step))
    Z = np.array([classifier(np.array([xx_, yy_])) for xx_, yy_ in zip(np.ravel(xx), np.ravel(yy))])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Iris dataset (2 classes)")

plt.subplot(132)
frontiere(f_linear, X, y)
plt.title("Frontière de décision noyau linéaire")

plt.subplot(133)
frontiere(f_poly, X, y)
plt.title("Frontière de décision noyau polynomial")

plt.tight_layout()
plt.show()




###############################################################################
#                         Classification de visages
###############################################################################
"""
The dataset used in this example is a preprocessed excerpt
of the "Labeled Faces in the Wild", aka LFW_:

  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

  _LFW: http://vis-www.cs.umass.edu/lfw/
"""

#%%
# Chargement des données 
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4,
                              color=True, funneled=False, slice_=None,
                              download_if_missing=True)
# data_home='.'
#%%
# Examinez les tableaux d'images pour en déterminer les dimensions 
images = lfw_people.images
n_samples, h, w, n_colors = images.shape

# L'étiquette à prédire est l'identifiant de la personne
target_names = lfw_people.target_names.tolist()

#%%
# Choisissez une paire à classifier, par exemple
names = ['Tony Blair', 'Colin Powell']
# names = ['Donald Rumsfeld', 'Colin Powell']
#%%
idx0 = (lfw_people.target == target_names.index(names[0]))
idx1 = (lfw_people.target == target_names.index(names[1]))
images = np.r_[images[idx0], images[idx1]]
n_samples = images.shape[0]
y = np.r_[np.zeros(np.sum(idx0)), np.ones(np.sum(idx1))].astype(int)

# Tracez un ensemble d'échantillons des données
plot_gallery(images, np.arange(12))
plt.show()


# Extraire les caractéristiques
#%%
# Caractéristiques utilisant uniquement les illuminations.
X = (np.mean(images, axis=3)).reshape(n_samples, -1)

# # Ou calculez les caractéristiques en utilisant les couleurs (3 fois plus de caractéristiques)
# X = images.copy().reshape(n_samples, -1)
#%%
# Scale features
X -= np.mean(X, axis=0)
X /= np.std(X, axis=0)

#%%
####################################################################
# Split data into a half training and half test set
# X_train, X_test, y_train, y_test, images_train, images_test = \
#    train_test_split(X, y, images, test_size=0.5, random_state=0)
# X_train, X_test, y_train, y_test = \
#    train_test_split(X, y, test_size=0.5, random_state=0)

indices = np.random.permutation(X.shape[0])
train_idx, test_idx = indices[:X.shape[0] // 2], indices[X.shape[0] // 2:]
X_train, X_test = X[train_idx, :], X[test_idx, :]
y_train, y_test = y[train_idx], y[test_idx]
images_train, images_test = images[
    train_idx, :, :, :], images[test_idx, :, :, :]


######################### Question 04 #########################################
# Quantitative evaluation of the model quality on the test set
#%% [code]
#L'étiquette à prédire est l'identifiant de la personne 
print("--- Linear kernel ---")
print("Fitting the classifier to the training set")
t0 = time()

# fit a classifier (linear) and test all the Cs
Cs = 10. ** np.arange(-5, 6)
scores = []

# Boucle sur les valeurs de C
for C in Cs:
    # Créer un classificateur SVM avec le noyau linéaire et le paramètre C
    clf = svm.SVC(kernel='linear', C=C)  #Créez le classificateur SVM
    clf.fit(X_train, y_train)  # Entraîner le modèle sur les données d'entraînement

    # Prédire les labels pour l'ensemble de test
    y_pred = clf.predict(X_test)  #Utilisez le modèle pour prédire sur l'ensemble de test

    # Calculer le score (précision) et l'ajouter à la liste des scores
    score = np.mean(y_pred == y_test)  #Calculez la précision du modèle
    scores.append(score)

# Trouver l'indice du meilleur score
ind = np.argmax(scores)
print("Best C: {}".format(Cs[ind]))

plt.figure()
plt.plot(Cs, scores)
plt.xlabel("Parametres de regularisation C")
plt.ylabel("Scores d'apprentissage")
plt.xscale("log")
plt.tight_layout()
plt.show()
print("Best score: {}".format(np.max(scores)))

print("Predicting the people names on the testing set")
t0 = time()



# predict labels for the X_test images with the best classifier
#%% 
# Créez un classificateur SVM avec le meilleur paramètre C
best_C = Cs[ind]  # Utilisez l'indice du meilleur score trouvé précédemment
clf = svm.SVC(kernel='linear', C=best_C)  #  Créer le classificateur SVM
clf.fit(X_train, y_train)  # Entraîner le modèle sur les données d'entraînement

print("done in %0.3fs" % (time() - t0))
# The chance level is the accuracy that will be reached when constantly predicting the majority class.
print("Chance level : %s" % max(np.mean(y), 1. - np.mean(y)))
print("Accuracy : %s" % clf.score(X_test, y_test))  # Calculer et afficher la précision du modèle

#%%
# Qualitative evaluation of the predictions using matplotlib

prediction_titles = [title(y_pred[i], y_test[i], names)
                     for i in range(y_pred.shape[0])]

plot_gallery(images_test, prediction_titles)
plt.show()

####################################################################
# jeter un oeil sur le coefficients 
plt.figure()
plt.imshow(np.reshape(clf.coef_, (h, w)))
plt.show()


############################ Question 05 ############################
#%% [code]
def run_svm_cv(_X, _y):
    _indices = np.random.permutation(_X.shape[0])
    _train_idx, _test_idx = _indices[:_X.shape[0] // 2], _indices[_X.shape[0] // 2:]
    _X_train, _X_test = _X[_train_idx, :], _X[_test_idx, :]

    _y_train, _y_test = _y[_train_idx], _y[_test_idx]

    _parameters = {'kernel': ['linear'], 'C': list(np.logspace(-3, 3, 5))}
    _svr = svm.SVC()
    _clf_linear = GridSearchCV(_svr, _parameters)
    _clf_linear.fit(_X_train, _y_train)

    print('Generalization score for linear kernel: %s, %s \n' %
          (_clf_linear.score(_X_train, _y_train), _clf_linear.score(_X_test, _y_test)))

# Evaluation du modèle sans variable de nuisance
print("Score sans variable de nuisance")
run_svm_cv(X, y)  # Exécute la fonction avec les données sans bruit

# Evaluation du modèle avec des variables de nuisance
print("Score avec variable de nuisance")
n_features = X.shape[1]
# On ajoute des variables de nuisance (bruit)
sigma = 1
noise = sigma * np.random.randn(n_samples, 300)  # Ajout de 300 features bruitées
X_noisy = np.concatenate((X, noise), axis=1)
X_noisy = X_noisy[np.random.permutation(X.shape[0])]

# Exécute la fonction avec les données bruitées
run_svm_cv(X_noisy, y)



#################################### Question 06 ###########################
#%% [code]
print("Score après réduction de dimension")

n_components = 100 
pca = PCA(n_components=n_components).fit(X_noisy)

# Transformation des données bruitées avec PCA
X_noisy_pca = pca.transform(X_noisy)

# Affichage de la variance expliquée pour s'assurer de la qualité de la réduction
print(f"Variance expliquée avec {n_components} composantes principales : {np.sum(pca.explained_variance_ratio_):.2f}")

# Application du modèle SVM sur les données réduites
run_svm_cv(X_noisy_pca, y)  # Utilisation du modèle SVM sur les données réduites par PCA



