# Segmentation de clientèle — Analyse RFM et K-means

[English](README.md) | **Français**

Segmentation des clients d'un vrai détaillant en ligne à l'aide de l'analyse RFM (Récence, Fréquence, Montant) et du clustering K-means, pour répondre à une question que se pose tout commerçant : *qui sont mes meilleurs clients, et lesquels suis-je sur le point de perdre ?*

**Constat principal :** environ **20 % des clients génèrent 74 % du chiffre d'affaires** — et près d'un quart des clients sont de bons acheteurs qui ont cessé de revenir.

**🚀 [Démo en ligne](https://isadora-customer-segmentation.streamlit.app)** — explorez les segments en 3D et classez un client vous-même (interface en anglais).

---

## Contexte

Avant de me réorienter en informatique, j'ai géré une entreprise de commerce de détail dans la mode. Savoir quels clients récompenser, lesquels reconquérir et lesquels laisser partir était une question quotidienne — à laquelle je répondais à l'intuition. Ce projet y répond avec des données, à l'aide des techniques du cours *Unsupervised Learning* de la spécialisation Machine Learning d'Andrew Ng.

## Données

[Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) (UCI Machine Learning Repository) : **1 067 371 transactions** d'un détaillant britannique d'articles-cadeaux en ligne, de décembre 2009 à décembre 2011. Une grande partie de ses clients sont des grossistes.

> Chen, D. (2012). *Online Retail II* [Jeu de données]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D — sous licence CC BY 4.0.

Les données ne sont pas incluses dans ce dépôt (45 Mo). Voir [Exécuter le projet](#exécuter-le-projet).

## Résultats

K-means a trouvé **quatre segments naturels** parmi 5 839 clients :

| Segment | Clients | Dernier achat | Commandes | Dépense typique | Part des ventes |
|---|---|---|---|---|---|
| **Clients fidèles à forte valeur** (*Champions*) | 20,2 % | il y a ~2 semaines | 13 | 4 863 £ | **73,7 %** |
| **Clients à risque** (*At risk*) | 24,7 % | il y a ~6 mois | 4 | 1 443 £ | 16,2 % |
| **Clients prometteurs** (*Promising*) | 21,6 % | il y a ~3 semaines | 3 | 713 £ | 6,4 % |
| **Clients perdus** (*Lost*) | 33,5 % | il y a plus d'un an | 1 | 269 £ | 3,7 % |

*Les valeurs typiques sont des médianes. Les noms entre parenthèses sont ceux utilisés dans les notebooks.*

**Ce qu'une entreprise pourrait en faire :**
- **Fidèles à forte valeur** — les protéger : avantages fidélité, service prioritaire. En perdre quelques-uns ferait très mal.
- **À risque** — la cible la plus rentable : c'étaient de bons clients, ils ne sont pas revenus depuis six mois (pas même pour les fêtes 2011), et ils sont encore récupérables. Faire revenir un client coûte moins cher que d'en trouver un nouveau.
- **Prometteurs** — encourager le prochain achat pour installer l'habitude.
- **Perdus** — tout au plus une relance peu coûteuse : un tiers des clients, mais moins de 4 % des ventes.

## Défis et solutions

**1. Des commandes annulées cachées dans les données.** Après un premier nettoyage, un client figurait parmi les dix plus gros acheteurs avec seulement 2 commandes — un profil incohérent. L'enquête a révélé une commande de **80 995 unités** d'un seul article, annulée le jour même. Retirer les lignes d'annulation ne suffisait pas : la commande d'origine restait dans les données, comme si la vente avait eu lieu.
→ Chaque annulation est désormais associée à sa commande d'origine (même client, même produit, même quantité, passée avant l'annulation), et les deux sont retirées : **6 332 commandes et 673 413 £ de ventes « fantômes »** éliminées.

**2. Des achats sans client.** 235 151 transactions (22 %) n'avaient pas d'identifiant client et ne pouvaient pas être segmentées — probablement des achats sans compte. Elles ont été retirées et documentées comme limite. Chaque étape du nettoyage est journalisée ; au total, **72,2 % des lignes ont été conservées**.

**3. Quelques géants qui faussent tout.** Le top 1 % des clients représente **31 % du chiffre d'affaires**. Comme K-means calcule des distances, ces grossistes auraient dominé le clustering. Plutôt que de les supprimer — ce sont de vrais clients, et les plus précieux —, une **transformation logarithmique** compresse les grandes valeurs, suivie d'une **standardisation** pour qu'aucun indicateur ne pèse plus que les autres à cause de ses unités.

**4. Choisir le nombre de segments.** Le score de silhouette est le plus élevé à k = 2, mais deux segments sont trop grossiers pour agir. Le score présente ensuite un **pic local à k = 4** (0,367, au-dessus de k = 3 et de k = 5), et la courbe du coude s'infléchit dans la même zone : quatre segments, c'est à la fois appuyé par les données *et* exploitable.

**5. Vérifier une hypothèse.** La distribution de la récence montrait un pic inhabituel autour de 400 jours. Hypothèse : des acheteurs des fêtes jamais revenus. Vérification sur les 769 clients concernés : leur dernier achat date de l'automne 2010 — surtout **octobre et novembre**, pas décembre. Pour un fournisseur d'articles-cadeaux dont beaucoup de clients sont des grossistes, cela évoque des **commerçants venus faire leurs stocks avant Noël**, qui n'ont pas recommandé l'année suivante.

**6. Scores RFM classiques ou K-means.** La méthode traditionnelle (scores de 1 à 5 par quintile) a été calculée pour comparaison. Elle découpe les clients en groupes de même taille, que ces groupes aient un sens ou non — la répartition des scores est presque plate. K-means trouve les frontières à partir des données elles-mêmes.

## Limites

- Les données montrent **que** les clients à risque sont partis, pas **pourquoi** (un concurrent ? une déception ?). Il faudrait d'autres sources pour le savoir : sondages, historique du service client.
- Les **retours partiels** (par exemple 3 unités retournées sur 10) ne peuvent pas être associés à une commande précise et ne sont pas déduits.
- Les clients prometteurs sont actifs, mais sans la date de leur premier achat, on ne peut pas affirmer qu'ils sont **nouveaux**.

## Structure du projet

```
customer-segmentation/
├── notebooks/
│   ├── 01_cleaning.ipynb     # nettoyage, avec un journal de chaque étape
│   ├── 02_rfm.ipynb          # indicateurs RFM, transformation log, scores classiques
│   └── 03_clustering.ipynb   # K-means, choix de k, profils des segments
├── app/
│   ├── streamlit_app.py      # démo interactive (Streamlit)
│   ├── rfm_segments.csv      # résultats des segments utilisés par la démo
│   └── requirements.txt
├── requirements.txt
└── data/                     # non versionné — voir ci-dessous
```

## Exécuter le projet

```bash
git clone https://github.com/IsadoraEaston/customer-segmentation.git
cd customer-segmentation
python -m venv .venv
.venv\Scripts\activate        # Windows  (macOS/Linux : source .venv/bin/activate)
pip install -r requirements.txt
```

Téléchargez le jeu de données depuis la [page UCI](https://archive.ics.uci.edu/dataset/502/online+retail+ii), décompressez-le et placez `online_retail_II.xlsx` dans un dossier `data/`. Exécutez ensuite les notebooks dans l'ordre (01 → 02 → 03) avec `jupyter notebook`.

## Prochaines étapes

- Vérifier si les clients à risque ont fait plus de retours partiels avant de partir (un signe possible de déception)
- Comparer le K-means de scikit-learn avec ma propre implémentation réalisée dans les labos de la spécialisation ML

## Technologies

Python · pandas · NumPy · scikit-learn · Matplotlib · Plotly · Streamlit · Jupyter

---

**Isabelle D. Easton** — [Portfolio](https://isadoraeaston.github.io) · [GitHub](https://github.com/IsadoraEaston)
