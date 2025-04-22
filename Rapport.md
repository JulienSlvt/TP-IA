Auteurs :
Albin Martin
Julien Salvat


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


- Si σ augmente, les neurones proches du neurone gagnant j∗ vont plus apprendre l'entrée courante.  
Car comme on le voit dans l'équation de mise à jour, les poids seront plus modifiés pour l'entrée, car l'influence des voisins est plus forte.
$$\Delta w_{ji} = \eta e^{-\frac{||j - j^*||_c^2}{2\sigma^2}} (x_i - w_{ji})$$  


- Si σ est plus grand à convergence, l’auto-organisation obtenue sera donc plus “resserrée” car au neuronne gagnant j∗, les neurones voisins vont plus apprendre de l’entrée courante.

- Pour quantifier l'influence de sigma, on peut utiliser la mesure de la moyenne des distances entre les vecteurs de poids des neurones voisins. 
$$M(\sigma) = \frac{1}{|\mathcal{C}|} \sum_{j \in \mathcal{C}} \frac{1}{|\mathcal{N}(j)|} \sum_{k \in \mathcal{N}(j)} ||W_j - W_k||$$



En conséquence, le poids des connexions entre les neurones sera plus important, c'est-à-dire que les neurones voisins auront une plus grande influence sur la mise à jour des poids.