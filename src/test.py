import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

###############################################################################
# Entraînement du modèle avec noyau linéaire vs noyau polynomial
###############################################################################

#%% Q1 Noyau linéaire
# GridSearch pour le noyau linéaire avec différents paramètres de régularisation (C)
parameters = {'kernel': ['linear'], 'C': list(np.logspace(-3, 3, 200))}

clf_linear = GridSearchCV(SVC(), parameters, cv=5)  # Recherche des meilleurs paramètres avec validation croisée
clf_linear.fit(X_train, y_train)  # Entraînement du modèle

# Calcul du score (précision) sur les ensembles d'entraînement et de test
print('Meilleurs paramètres pour noyau linéaire:', clf_linear.best_params_)
print('Score de généralisation pour noyau linéaire: train = %.2f, test = %.2f' %
      (clf_linear.score(X_train, y_train), clf_linear.score(X_test, y_test)))

#%% Q2 Noyau polynomial
# GridSearch pour le noyau polynomial avec paramètres C, gamma, et degree
Cs = list(np.logspace(-3, 3, 5))
gammas = 10. ** np.arange(1, 2)
degrees = np.r_[1, 2, 3]

parameters = {'kernel': ['poly'], 'C': Cs, 'gamma': gammas, 'degree': degrees}

clf_poly = GridSearchCV(SVC(), parameters, cv=5)  # Recherche des meilleurs paramètres pour noyau polynomial
clf_poly.fit(X_train, y_train)  # Entraînement du modèle

# Affichage des meilleurs paramètres et des scores
print('Meilleurs paramètres pour noyau polynomial:', clf_poly.best_params_)
print('Score de généralisation pour noyau polynomial: train = %.2f, test = %.2f' %
      (clf_poly.score(X_train, y_train), clf_poly.score(X_test, y_test)))

#%% Visualisation des résultats avec frontiere

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
