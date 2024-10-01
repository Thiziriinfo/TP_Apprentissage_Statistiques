# TP - Support Vector Machines (SVM)

## Auteurs :
- **Nom** : ABCHICHE
- **Prénom** : THIZIRI

## Description :
Ce dépôt contient le compte-rendu du TP sur les machines à vecteurs de support (SVM). Le compte-rendu a été rédigé en \LaTeX{} et compile tous les résultats obtenus lors des différentes étapes du TP, ainsi que le code Python utilisé pour la classification des données. Le TP vise à appliquer des techniques de classification à l'aide des machines à vecteurs de support (SVM) sur différents jeux de données. Nous avons notamment comparé l'impact des noyaux linéaires et polynomiaux, exploré l'effet des variables de nuisance, et amélioré les performances grâce à une réduction de dimension par PCA.

## Arborescence du projet :
- **src/** : Contient les fichiers Python nécessaires à l'exécution des analyses (ex. : `script.py`).
- **images/** : Contient les images générées par les scripts Python.
- **TPABCHICHETHIZIRI_SVM.tex** : Le fichier \LaTeX{} contenant le compte-rendu complet du TP.
- **TPABCHICHETHIZIRI_SVM.pdf** : Le fichier PDF généré à partir du fichier \LaTeX{}.
- **README.md** : Ce fichier, contenant les instructions pour compiler le projet.
- **.gitignore** : Fichier pour ignorer les fichiers inutiles dans le dépôt (fichiers temporaires, logs, etc.).
- **requirements.txt** : Liste des dépendances Python nécessaires à l'exécution du projet.

## Instructions de compilation

Pour générer le PDF à partir du fichier LaTeX, suivez ces étapes :

1. **Installer \LaTeX{} :**
   - Si vous n'avez pas \LaTeX{} installé sur votre machine, vous pouvez le télécharger à partir de [TeX Live](https://www.tug.org/texlive/) ou [MiKTeX](https://miktex.org/).
   - Alternativement, vous pouvez utiliser un service en ligne comme [Overleaf](https://www.overleaf.com/) qui permet de travailler directement dans le navigateur sans installation.

2. **Compiler le fichier :**
   - Ouvrez un terminal (ou, si vous utilisez Overleaf, téléchargez votre fichier `.tex` sur la plateforme).
   - Si vous utilisez un terminal, naviguez jusqu'au répertoire contenant le fichier LaTeX :
     ```bash
     cd /chemin/vers/le/dossier
     ```
   - Pour compiler le fichier `TPABCHICHETHIZIRI_SVM.tex`, exécutez la commande suivante :
     ```bash
     pdflatex TPABCHICHETHIZIRI_SVM.tex
     ```
   - Cette commande va générer un fichier PDF à partir du fichier `.tex`. Vous devrez peut-être exécuter la commande plusieurs fois pour résoudre toutes les références et les citations.

3. **Vérification :**
   - Une fois la compilation terminée, vérifiez que le fichier PDF (`TPABCHICHETHIZIRI_SVM.pdf`) a été créé dans le même répertoire. Ouvrez-le pour visualiser le rapport.

### Exécuter le code Python

Pour exécuter le code Python contenu dans ce projet, suivez ces étapes :

1. **Installer les dépendances :**
   - Assurez-vous que vous avez installé les bibliothèques nécessaires. Vous pouvez le faire en exécutant :
     ```bash
     pip install -r requirements.txt
     ```

2. **Naviguer vers le dossier des scripts :**
   - Ouvrez un terminal et accédez au dossier `src/` où se trouvent les scripts Python :
     ```bash
     cd src/
     ```

3. **Exécuter le script Python :**
   - Pour exécuter le script principal, utilisez la commande suivante :
     ```bash
     python code_source.py
     ```
   - Si vous souhaitez lancer l'interface graphique pour l'exploration des SVM, exécutez également :
     ```bash
     python svm_gui.py
     ```

4. **Vérification des résultats :**
   - Après l'exécution des scripts, vérifiez les résultats affichés dans le terminal. Les images générées seront sauvegardées dans le dossier `images/`.

## Remarques :
- Assurez-vous que tous les fichiers nécessaires (code Python, images, fichiers .tex) sont présents dans leur répertoire respectif avant de commencer la compilation et l'exécution.
- Si vous rencontrez des problèmes lors de la compilation ou de l'exécution des scripts, consultez la documentation des bibliothèques utilisées ou les logs d'erreur.

---
### 1. Cloner le dépôt :
```bash
git clone [https://github.com/Thiziriinfo/TP_Apprentissage_Statistiques.git]
