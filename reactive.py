import numpy as np
import time
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

import psutil

process = psutil.Process()

# Avant l'exécution de votre code
start_memory = process.memory_info().rss

startTime = time.time()

start_t = time.process_time()

def powerset(fullset):  #fonction qui permet de definir powerset(N) = (2^N)
    listsub = list(fullset)
    subsets = []
    for i in range(2**len(listsub)):
        subset = []
        for k in range(len(listsub)):            
            if i & 1<<k:
                subset.append(listsub[k])
        subsets.append(subset)        
    return subsets

# attaque 1
subset1 = powerset(set(["prob1","prob2","prob3"]))
sub1 = ["prob1","prob2","prob3"]

subset11 = powerset(set(["prob1","prob2"]))
sub11 = ["prob1","prob2"]

subset10 = powerset(set(["prob1","prob2","prob3","prob4","prob5"]))
sub10 = ["prob1","prob2","prob3","prob4","prob5"]

# attaque 2
subset2 = powerset(set(["prob1","prob2","prob3"]))
sub2 = ["prob1","prob2","prob3"]

subset22 = powerset(set(["prob1","prob2"]))
sub22 = ["prob1","prob2"]

subset20 = powerset(set(["prob1","prob2","prob3","prob4","prob5"]))
sub20 = ["prob1","prob2","prob3","prob4","prob5"]

# attaque 3
subset3 = powerset(set(["prob1","prob2"]))
sub3 = ["prob1","prob2"]

subset33 = powerset(set(["prob1","prob2","prob3"]))
sub33 = ["prob1","prob2","prob3"]

subset333 = powerset(set(["prob1","prob2"]))
sub333 = ["prob1","prob2"]

subset30 = powerset(set(["prob1","prob2","prob3","prob4","prob5","prob6","prob7"]))
sub30 = ["prob1","prob2","prob3","prob4","prob5","prob6","prob7"]

# attaque 4
subset4 = powerset(set(["prob1","prob2","prob3"]))
sub4 = ["prob1","prob2","prob3"]

subset44 = powerset(set(["prob1","prob2","prob3","prob4","prob5"]))
sub44 = ["prob1","prob2","prob3","prob4","prob5"]

subset40 = powerset(set(["prob1","prob2","prob3","prob4","prob5","prob6","prob7","prob8"]))
sub40 = ["prob1","prob2","prob3","prob4","prob5","prob6","prob7","prob8"]

# definir les payoffs et les connaissances a priori sur l'attaque 1
G1 = [0.49, 0.49, 0.52] 
C1 = [0.20, 0.20, 0.20]
b1 = [0.20, 0.20, 0.20]
u1 = 2.20 

G10 = [0.49, 0.49, 0.52, 0.46, 0.49] 
C10 = [0.20, 0.20, 0.20, 0.20, 0.20]
b10 = [0.20, 0.20, 0.20, 0.20, 0.20]
u10 = 3.20 

G11 = [0.46, 0.49] 
C11 = [0.20, 0.20]
b11 = [0.20, 0.20]
u11 = 1.0 

# definir les gains et pertes et les connaissances a priori sur l'attaque 4
G2 = [0.46, 0.46, 0.46]
C2 = [0.20, 0.20, 0.20]
b2 = [0.2, 0.2, 0.2]
u2 = 2.40 

G20 = [0.46, 0.46, 0.46, 0.52, 0.49]
C20 = [0.20, 0.20, 0.20, 0.20, 0.20]
b20 = [0.20, 0.20, 0.20, 0.20, 0.20]
u20 = 4.0 

G22 = [0.52, 0.49]
C22 = [0.20, 0.20]
b22 = [0.2, 0.2]
u22 = 1.6 

# definir les gains et pertes et les connaissances a priori sur l'attaque 6
G3 = [0.49, 0.46]
C3 = [0.20, 0.20]
b3 = [0.143, 0.143]
u3 = 0.0 

G30 = [0.49, 0.46, 0.49, 0.52, 0.49, 0.46, 0.49]
C30 = [0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20]
b30 = [0.143, 0.143, 0.143, 0.143, 0.143, 0.143, 0.143]
u30 = 1.6 

G33 = [0.49, 0.52, 0.49]
C33 = [0.20, 0.20, 0.20]
b33 = [0.143, 0.143, 0.143]
u33 = 0.80

G333 = [0.46, 0.49]
C333 = [0.20, 0.20]
b333 = [0.143, 0.143]
u333 = 0.80 

# definir les gains et pertes et les connaissances a priori sur l'attaque 7
G4 = [0.49, 0.49, 0.46]
C4 = [0.20, 0.20, 0.20]
b4 = [0.125, 0.125, 0.125]
u4 = 1.0 

G40 = [0.49, 0.49, 0.46, 0.55, 0.52, 0.52, 0.46, 0.49]
C40 = [0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20]
b40 = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
u40 = 3.20 

G44 = [0.55, 0.52, 0.52, 0.46, 0.49]
C44 = [0.20, 0.20, 0.20, 0.20, 0.20]
b44 = [0.125,0.125,0.125,0.125,0.125]
u44 = 2.20 

def payoff1(subset,G,C,k):
    update = []
    for i,y in enumerate(subset):
        u = k
        for j,x in enumerate(y):
            if x == "prob1":
                u = u + (G[0]+C[0])
            elif x == "prob2":
                u = u + (G[1]+C[1])
            elif x == "prob3":
                u = u + (G[2]+C[2])
            elif x == "prob4":
                u = u + (G[3]+C[3])
            elif x == "prob5":
                u = u + (G[4]+C[4])  
            elif x == "prob6":
                u = u + (G[5]+C[5])
            elif x == "prob7":
                u = u + (G[6]+C[6])
            elif x == "prob8":
                u = u + (G[7]+C[7])    
        update.append(u)
        
    return update

payoff10 = payoff1(subset1,G1,C1,u1) 
payoff11 = payoff1(subset11,G11,C11,u11)
payoff100 = payoff1(subset10,G10,C10,u10)

payoff2 = payoff1(subset2,G2,C2,u2) 
payoff22 = payoff1(subset22,G22,C22,u22)
payoff20 = payoff1(subset20,G20,C20,u20)

payoff3 = payoff1(subset3,G3,C3,u3) 
payoff33 = payoff1(subset33,G33,C33,u33)
payoff30 = payoff1(subset30,G30,C30,u30)
payoff333 = payoff1(subset333,G333,C333,u333)

payoff4 = payoff1(subset4,G4,C4,u4) 
payoff40 = payoff1(subset40,G40,C40,u40) 
payoff44 = payoff1(subset44,G44,C44,u44)


# calcules des probabilités sur les actions du defenseur ou encore les éléments du powerset

def prob(subset,sub,p):
    update = []
    for i,y in enumerate(subset): 
        x = 1
        for j,k in enumerate(sub): 
            if k in y:
                x = x*p[j]             
        update.append(x)
  
    return update

def prob1(subset,sub,p):
    update = []
    for i,y in enumerate(subset): 
        x = 1
        for j,k in enumerate(sub): 
            if k in y:
                x = x*p[j]
            else:
                x = x*(1-p[j])
                
        update.append(x)
  
    return update

probability1 = prob(subset1,sub1,b1)
probability10 = prob1(subset10,sub10,b10)
probability11 = prob(subset11,sub11,b11)

probability2 = prob(subset2,sub2,b2)
probability20 = prob1(subset20,sub20,b20)
probability22 = prob(subset22,sub22,b22)

probability3 = prob(subset3,sub3,b3)
probability30 = prob1(subset30,sub30,b30)
probability33 = prob(subset33,sub33,b33)
probability333 = prob(subset333,sub333,b333)

probability4 = prob(subset4,sub4,b4)
probability40 = prob1(subset40,sub40,b40)
probability44 = prob(subset44,sub44,b44)

for i,x in enumerate(probability1): 
    y = sum(probability1) - 1
    if x == 1:
        probability1[i] = 1 - y
        
for i,x in enumerate(probability11): 
    y = sum(probability11) - 1
    if x == 1:
        probability11[i] = 1 - y
        
for i,x in enumerate(probability2): 
    y = sum(probability2) - 1
    if x == 1:
        probability2[i] = 1 - y
        
for i,x in enumerate(probability22): 
    y = sum(probability22) - 1
    if x == 1:
        probability22[i] = 1 - y
        
for i,x in enumerate(probability3): 
    y = sum(probability3) - 1
    if x == 1:
        probability3[i] = 1 - y
        
for i,x in enumerate(probability33): 
    y = sum(probability33) - 1
    if x == 1:
        probability33[i] = 1 - y
        
for i,x in enumerate(probability333): 
    y = sum(probability333) - 1
    if x == 1:
        probability333[i] = 1 - y
        
for i,x in enumerate(probability4): 
    y = sum(probability4) - 1
    if x == 1:
        probability4[i] = 1 - y
        
for i,x in enumerate(probability44): 
    y = sum(probability44) - 1
    if x == 1:
        probability44[i] = 1 - y

print(probability2)
print(probability22)

print(sum(probability1))
print(sum(probability11))


# fonction pour calculer le gain espéré

def expected(payoff,prob):
    u = []
    for i,y in enumerate(payoff):
        u.append(y*prob[i])        
    return u

e1 = expected(payoff10,probability1) # expected payoffs du domain 1 sur l'attaque 1
e10 = expected(payoff100,probability10)  # expected payoffs single domain sur l'attaque 1
e11 = expected(payoff11,probability11)  # expected payoffs du domain 2 sur l'attaque 1

e2 = expected(payoff2,probability2) # expected payoffs du domain 1 sur l'attaque 2
e20 = expected(payoff20,probability20) # expected payoffs single domain sur l'attaque 2
e22 = expected(payoff22,probability22)  # expected payoffs du domain 2 sur l'attaque 2

e3 = expected(payoff3,probability3)  # expected payoffs du domain 1 sur l'attaque 3
e30 = expected(payoff30,probability30) # expected payoffs single domain sur l'attaque 3
e33 = expected(payoff33,probability33)  # expected payoffs du domain 2 sur l'attaque 3
e333 = expected(payoff333,probability333) # expected payoffs du domain 3 sur l'attaque 3

e4 = expected(payoff4,probability4)  # expected payoffs du domain 1 sur l'attaque 4
e40 = expected(payoff40,probability40)  # expected payoffs single domain sur l'attaque 4
e44 = expected(payoff44,probability44)  # expected payoffs du domain 2 sur l'attaque 4


##########################################################################
# Définir vos vecteurs
vecteurs_1 = [
    np.array(e1),
    np.array(e11)
]

vecteurs_2 = [
    np.array(e2),
    np.array(e22)
]

vecteurs_3 = [
    np.array(e3),
    np.array(e33),
    np.array(e333)
]

vecteurs_4 = [
    np.array(e4),
    np.array(e44),
]

# Trouver la taille maximale parmi les vecteurs
taille_maximale_1 = max(len(v) for v in vecteurs_1)
taille_maximale_2 = max(len(v) for v in vecteurs_2)
taille_maximale_3 = max(len(v) for v in vecteurs_3)
taille_maximale_4 = max(len(v) for v in vecteurs_4)

# Créer une liste de vecteurs avec la même taille que la taille maximale
vecteurs_modifies_1 = [np.concatenate((v, np.zeros(taille_maximale_1 - len(v)))) for v in vecteurs_1]
vecteurs_modifies_2 = [np.concatenate((v, np.zeros(taille_maximale_2 - len(v)))) for v in vecteurs_2]
vecteurs_modifies_3 = [np.concatenate((v, np.zeros(taille_maximale_3 - len(v)))) for v in vecteurs_3]
vecteurs_modifies_4 = [np.concatenate((v, np.zeros(taille_maximale_4 - len(v)))) for v in vecteurs_4]

me1 = vecteurs_modifies_1[0]
me10 = vecteurs_modifies_1[1]

me11 = list(product(me1, me10))

me2 = vecteurs_modifies_2[0]
me20 = vecteurs_modifies_2[1]

me22 = list(product(me2, me20))

me3 = vecteurs_modifies_3[0]
me30 = vecteurs_modifies_3[1]
me31 = vecteurs_modifies_3[2]

me33 = list(product(me3, me30, me31))

me4 = vecteurs_modifies_4[0]
me40 = vecteurs_modifies_4[1]

me44 = list(product(me4, me40))


def sommer_tuple(tup):
    somme = 0
    for element in tup:
        somme += element
    return somme

def multiplier_liste(liste):
    resultat = 1
    for element in liste:
        resultat *= element
    return resultat

for i in range(len(me11)):
    me11[i] = sommer_tuple(me11[i])
    
for i in range(len(me11)):
    p0 = 0.25
    p1 = 0.75
    cout = 0.20
    alpha = me11[i]
     
    cout1 = alpha + cout
    me11[i] = (p0*cout1)/(p1*cout)
    
def sommer_tuple(tup):
    somme = 0
    for element in tup:
        somme += element
    return somme

def multiplier_liste(liste):
    resultat = 1
    for element in liste:
        resultat *= element
    return resultat

for i in range(len(me22)):
    me22[i] = sommer_tuple(me22[i])
    
for i in range(len(me22)):
    p0 = 0.5
    p1 = 0.5
    cout = 0.20
    alpha = me22[i]
     
    cout1 = alpha + cout
    me22[i] = (p0*cout1)/(p1*cout)
    
    
def sommer_tuple(tup):
    somme = 0
    for element in tup:
        somme += element
    return somme

def multiplier_liste(liste):
    resultat = 1
    for element in liste:
        resultat *= element
    return resultat

for i in range(len(me33)):
    me33[i] = sommer_tuple(me33[i])
    
for i in range(len(me33)):
    p0 = 0.25
    p1 = 0.75
    cout = 0.20
    alpha = me33[i]
     
    cout1 = alpha + cout
    me33[i] = (p0*cout1)/(p1*cout)


def sommer_tuple(tup):
    somme = 0
    for element in tup:
        somme += element
    return somme

def multiplier_liste(liste):
    resultat = 1
    for element in liste:
        resultat *= element
    return resultat

for i in range(len(me44)):
    me44[i] = sommer_tuple(me44[i])
    
for i in range(len(me44)):
    p0 = 0.25
    p1 = 0.75
    cout = 0.20
    alpha = me44[i]
     
    cout1 = alpha + cout
    me44[i] = (p0*cout1)/(p1*cout)
    

# Plotting memory usage over time
plt.plot(me44, marker='o',label='Multi-domain deception')

#plt.plot(me11,label='Multi-domain deception')

# Lignes horizontales de division
division_line1 = 1

# Mettre en surbrillance la région entre les deux lignes horizontales
#plt.axhspan(0.0, division_line1, facecolor='yellow', alpha=0.3, label='successful deception')
#plt.axhspan(division_line1, 2.8, facecolor='green', alpha=0.3, label='unsuccessful deception')

max_value = max(me44)
max_index = me44.index(max_value)

plt.annotate(f'Multi-objectif optimal response', xy=(max_index, max_value), xytext=(max_index + 0.1, max_value + 0.1),
             arrowprops=dict(facecolor='black', shrink=0.01))

# Ajouter des lignes horizontales de division
plt.axhline(division_line1, color='black', linestyle='--', label='deception threshold')

plt.xlabel('Attacks (H0 = 25, H1 = 75)')
plt.ylabel('Decision function of attacker')
plt.title('')

# Ajouter une légende
plt.legend()

# Afficher le graphique
plt.show()    
end_time = time.time()
###############################################################################################################

at1 = max(e1) + max(e11) # faire la somme des meilleurs stratégies ou reponses
at10 = max(e10) # obtenir le gains de la meilleur stratégie ou reponse

at2 = max(e2) + max(e22) 
at20 = max(e20) 

at3 = max(e3) + max(e33) + max(e333) 
at30 = max(e30) 

at4 = max(e4) + max(e44) 
at40 = max(e40) 

################################################################################################


import matplotlib.pyplot as plt
import numpy as np

# Vos données
data = [
    [at10, at20, at30, at40],
    [at1, at2, at3, at4]
]

# Convertir les données en un tableau NumPy pour une manipulation plus facile
data_array = np.array(data)

# Créer un graphique à barres
fig, ax = plt.subplots()

# Nombre de groupes de barres (égal au nombre de colonnes dans vos données)
num_groups = len(data[0])

# Largeur de chaque barre
bar_width = 0.2

# Positions des barres sur l'axe x
index = np.arange(num_groups)

# Couleurs des barres
colors = ['g', 'r', 'c', 'b']

labels = ['multi-domain','single-domain']
# Créer les barres pour chaque groupe de données
for i in range(len(data)):
    ax.bar(index + i * bar_width, data_array[i], bar_width, label=labels[i], color=colors[i])

# Ajouter des étiquettes et une légende
ax.set_xlabel('Attacks Multi-domaines')
ax.set_ylabel('reward of deception')
#ax.set_title('Execution times for honeypots response')
ax.set_xticks(index + bar_width * (len(data) - 1) / 2)
ax.set_xticklabels([f'Attack {j + 1}' for j in range(num_groups)])
ax.legend()

# Afficher le graphique
plt.savefig('mon_graphique.png')  # Enregistre le graphique sous le nom 'mon_graphique.png'
plt.show()








print(end_time - start_t, "Seconds")