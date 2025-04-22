Auteurs :
Albin Martin
Julien Salvat

### Influence de σ : 


- Si σ augmente, les neurones proches du neurone gagnant j∗ vont plus apprendre l'entrée courante.  
Car comme on le voit dans l'équation de mise à jour, les poids seront plus modifiés pour l'entrée, car l'influence des voisins est plus forte.
$$\Delta w_{ji} = \eta e^{-\frac{||j - j^*||_c^2}{2\sigma^2}} (x_i - w_{ji})$$  


- Si σ est plus grand à convergence, l’auto-organisation obtenue sera donc plus “resserrée” car au neuronne gagnant j∗, les neurones voisins vont plus apprendre de l’entrée courante.

- Pour quantifier l'influence de sigma, on peut utiliser la mesure de la moyenne des distances entre les vecteurs de poids des neurones voisins. 
$$M(\sigma) = \frac{1}{|\mathcal{C}|} \sum_{j \in \mathcal{C}} \frac{1}{|\mathcal{N}(j)|} \sum_{k \in \mathcal{N}(j)} ||W_j - W_k||



En conséquence, le poids des connexions entre les neurones sera plus important, c'est-à-dire que les neurones voisins auront une plus grande influence sur la mise à jour des poids.