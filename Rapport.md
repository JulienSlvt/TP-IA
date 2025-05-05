Auteurs :  
Albin Martin  
Julien Salvat

# Intelligence Artificielle - Réseaux de Neurones

## Etude "théorique" de cas simples

### Influence de η 

- η = 0 :  En reprenant la formule : $\Delta w_{ji} = \eta e^{-\frac{\|j - j^*\|_c^2}{2\sigma^2}} (x_i - w_{ji})$  
    On voit que si η = 0, alors $\Delta w_{ji} = 0^{truc}$ = 0, donc la prochaine valeur des poids du neurone j ne changera pas.

- η = 1 :
    On voit que si η = 1, alors $\Delta w_{ji} = 1 * e^{0} (x_i - w_{ji})$, car $\|j - j^*\|_c^2 = 0$, en effet, le neurone j est le neurone gagnant, donc il n'y a pas de distance entre lui et lui-même (j* = j).  
    On a alors $\Delta w_{ji} = 1 * (x_i - w_{ji})$ = $x_i - w_{ji}$.   
    Calculons la nouvelle valeur du poids $W^*_{new}$ :  
    $$W^*_{new} = W^*_{current} + \Delta W^*$$
    En remplaçant $\Delta W^*$ par l'expression que nous venons de trouver (sous forme vectorielle, $\Delta W^* = X - W^*_{current}$) :
    $$W^*_{new} = W^*_{current} + (X - W^*_{current})$$
    $$W^*_{new} = W^*_{current} - W^*_{current} + X$$
    $$W^*_{new} = X$$

    La prochaine valeur des poids du neurone gagnant (W*) sera exactement égale au vecteur d'entrée courant X

- η ∈ ]0,1[ :  
    On voit que si η ∈ ]0,1[, on a de même $\Delta w_{ji} = η * (x_i - w_{ji})$ avec $0 < η < 1$.  
    On a alors : $$W^*_{new} = W^*_{current} + η * (X - W^*_{current})$$
    soit : $$W^*_{new} = (1 - η) * W^*_{current} + η * X$$

    On remarque que la formule correspond à une moyenne pondérée entre le vecteur d'entrée courant X et la valeur courante des poids du neurone gagnant W*. On peut alors dire que : 
    - Si η est proche de 0, la nouvelle valeur des poids sera proche de la valeur courante des poids du neurone gagnant W*.
    - Si η est proche de 1, la nouvelle valeur des poids sera proche du vecteur d'entrée courant X.
    - Si η = 0.5, la nouvelle valeur des poids sera à mi-chemin entre le vecteur d'entrée courant X et la valeur courante des poids du neurone gagnant W*.

- η > 1 : 
    On voit que si η > 1, on a de même : $$W^*_{new} = (1 - η) * W^*_{current} + η * X$$ avec $η > 1$.
    On a alors que (1 - η) < 0, ce qui signifie que W^*_{new} sera en dehors de l'intervalle [W^*_{current}, X]. 
    Cela a pour conséquence que : 
    - le neuronne gagnant dépassera la cible X à chaque itération, et donc il ne convergera pas vers la cible X.
    - les poids oscilleront autour de la cible X, mais ne convergeront pas vers elle.
    - cette divergence peut entraîner une instabilité dans l'apprentissage.

### Influence de σ : 

- Si σ augmente, l'influence du neurone gagnant j* s'étend à un voisinage plus large. Par conséquent, les poids des neurones proches de j* seront davantage modifiés en direction de l'entrée courante, comme le montre l'équation de mise à jour, le terme exponentiel augmente pour ces voisins, ce qui amplifie la variation des poids.
$$\Delta w_{ji} = \eta e^{-\frac{||j - j^*||_c^2}{2\sigma^2}} (x_i - w_{ji})$$  


-  Si σ est plus grand à convergence, σ influence une zone plus large autour du neurone gagnant, donc plus de neurones apprennent l'entrée courante. Un grand σ au début favorise une organisation globale, alors qu'un grand σ à convergence donne une auto-organisation plus lâche.

- Pour quantifier l'influence de σ, on peut utiliser la mesure de la moyenne des distances entre les vecteurs de poids des neurones voisins.  
Cette formule représente la moyenne des distances moyennes entre les voisins de chaque neurone. Pour chaque neuronne de C, on calcule la moyenne du poids entre le neurone parcouru et ses voisins puis additionne les résultats et en fait une moyenne. 

    -   M(σ): Prend σ comme entrée et renvoie une valeur qui représente le "resserrement" des poids des neurones voisins. 
    -   N: L'ensemble des neurones.
    -   V(j): L'ensemble des neurones voisins du neurone j.
    -   P(j): Le poids du neurone j.

$$M(\sigma) = \frac{1}{|\mathcal{N}|} \sum_{j \in \mathcal{N}} \frac{1}{|\mathcal{V}(j)|} \sum_{k \in \mathcal{V}(j)} ||P_j - P_k||$$


Pour conclure, σ contrôle l'étendue de ce voisinage. Une grande valeur de σ signifie qu'un grand nombre de neurones autour du gagnant seront mis à jour, tandis qu'une petite valeur signifie que seuls les neurones très proches seront affectés. De plus, si σ est grand, les neuronnes proches du gagnant vont peu apprendre, alors que si σ est petit, seuls les neurones très proches du gagnant seront mis à jour de manière significative.

### Influence de la distribution d’entrée
#### X1 et X2 présentés autant de fois :
Si X1 et X2 sont présentés autant de fois et avec un η faible et suffisamment de présentations, le neurone va autant apprendre les deux entrées et donc pour minimiser l'erreur, les poids des neurones convergeront vers la moyenne.
$$P = \frac{X_1 + X_2}{2}$$

#### X1 est présenté n fois plus que X2
Il se passe la même chose que précédemment pour les p premières itérations, donc le poid est identique à la moyenne des deux entrées.
Puis, à partir de la n+1 itération, le neurone va apprendre X1 n fois et donc le poids va converger vers X1.
$$P = \frac{X_1 + X_2}{2}$$
$$P = \frac{n \cdot X_1 + X_2}{n + 1}$$

#### Entrées provenant d'une base de données quelconque
Si les entrées proviennent d'une base de données quelconque, la moyenne des entrées va converger vers la moyenne de la base de données.
$$P = \frac{X_1 + X_2 + ... + X_n}{n}$$

#### Carte à plusieurs neurones
Si la carte a plusieurs neurones, ces derniers vont apprendre les entrées de la base de données, et adapter leurs poids pour se rapprocher de ces dernières. De plus, comme un neurone apprend plus des entrées aux poids qui lui sont le plus ressemblants, il finit par se spécialiser dans une certaine région ou un certain motif sur la carte.

Au cours de l'entrainement, les neuronnes vont se répartir de manière à minimiser la distance entre les données d'entrée et les poids des neurones. Ainsi, les neurones auront tendance à se concentrer dans les régions de l'espace d'entrée où la densité de données est la plus élevée. Et comme, plus d'entrées y sont présentées, l'ajustement des poids est plus fréquents dans ces zones de la carte.

La quantification vectorielle permet de mesurer ce phénomène en quantifiant la similarité entre les vecteurs d'entrée et les vecteurs de poids des neurones. Les neurones dont les poids sont les plus proches des données d'entrée seront activés, ce qui permet de regrouper les données similaires en clusters. La distribution des neurones sur la carte reflète donc la distribution des données dans l'espace d'entrée.

## Analyse de l’algorithme
### Hypothèses
#### Influence de η :
Si η est très petit (proche de 0), alors l'apprentissage sera lent et nécessitera un grand nombre d'itérations pour converger vers. 

Si η est grand (plus que 1), l'apprentissage sera rapide initialement, mais à la fin oscillera autour des valeurs optimales, ce qui au long terme diminue la qualité du résultat.

Une valeur de η intermédiaire (entre 0.2 et 0.8) devrait offrir un bon compromis entre vitesse de convergence et stabilité.

#### Influence de σ :