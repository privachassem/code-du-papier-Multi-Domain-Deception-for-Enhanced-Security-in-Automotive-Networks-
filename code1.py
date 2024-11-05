from networkx.classes.function import degree, is_empty, number_of_nodes, subgraph
from networkx.generators import directed
from networkx.generators.random_graphs import erdos_renyi_graph, watts_strogatz_graph
from networkx.generators.trees import random_tree
from networkx.generators.directed import gn_graph, gnc_graph, gnr_graph
import networkx as nx
from networkx.utils.misc import PythonRandomInterface
#import pyvis
#from pyvis.network import Network
#from pyvis import network as net
from networkx.linalg.attrmatrix import _node_value
import numpy as np
import numpy
from numpy.core.fromnumeric import nonzero, shape, squeeze
from numpy.core.numeric import Inf
from numpy.lib.function_base import append
import scipy as sp
import matplotlib.pyplot as plt
import pickle
import random
from itertools import combinations_with_replacement,combinations,permutations
import operator as op
from functools import reduce
import nashpy as nash
import time
from threading import Timer
import subprocess
import matplotlib.pyplot as plt
from networkx.algorithms.coloring import greedy_color, strategy_random_sequential
import copy
import time
#from Allocate_i import get_dynamic_allocation

startTime = time.time()

# ee=[(0,2),(0,3),(1,3),(1,4),(2,5),(2,6),(3,5),(3,7),(4,6),(4,7)]
# tt= [tuple(i) for i in np.add(ee,5)]
# print(tt)
# g=nx.Graph()
# g.add_edges_from([(0,1),(0,2),(1,2),(2,3),(3,4),(2,4),(3,5),(4,5)]) 


zero_day = [((8,10),),((7,12),),((8,10), (7,12))]

  
ee = [(0,3),(0,4),(0,5),(0,6),(2,3),(2,4),(2,5),(2,6),(1,3),(1,4),(1,5),(1,6),(6,7),(6,8),(6,9),(6,11),(6,12),(7,10),(8,11),
      (9,12),(10,11)]
# graphe connue des deux joueurs sans zero-days vulnérabilités
    
       
ee1= [(0,3),(0,4),(0,5),(0,6),(2,3),(2,4),(2,5),(2,6),(1,3),(1,4),(1,5),(1,6),(6,7),(6,8),(6,9),(6,11),(6,12),(7,10),(8,10),(8,11),
      (9,12),(10,11)]
# (10,13) comme zero-day vulnérabilité

    
ee2= [(0,3),(0,4),(0,5),(0,6),(2,3),(2,4),(2,5),(2,6),(1,3),(1,4),(1,5),(1,6),(6,7),(6,8),(6,9),(6,11),(6,12),(7,10),(8,11),(7,12),
      (9,12),(10,11)]
# (12,13) comme zero-day vulnérabilité

      
ee3= [(0,3),(0,4),(0,5),(0,6),(2,3),(2,4),(2,5),(2,6),(1,3),(1,4),(1,5),(1,6),(6,7),(6,8),(6,9),(6,11),(6,12),(7,10),(8,10),(8,11),(7,12),
      (9,12),(10,11)]
# ((10,13),(12,13)) comme zero-day vulnérabilité

    
Target_nodes=[11, 12]
Target_nodes_values = [80,80]


g=nx.DiGraph()
g.add_edges_from(ee, size=24)

#########################################
g1=nx.DiGraph()
g1.add_edges_from(ee1, size=24)
   

#########################################
g2=nx.DiGraph()
g2.add_edges_from(ee2, size=24)

#########################################
g3=nx.DiGraph()
g3.add_edges_from(ee3, size=24)
    
print(g.nodes)



start_t = time.process_time()

numb_of_honeypots = int(input("Number of Honeypots: <1, 2, 3, 4, 5, 6, 7, 8, 9> "))

index = 0  

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom  # or / in Python 2


nodes = np.size(g.nodes)       #renvoie le nombre d'éléments du tableau.
#print("Num_nodes: ",nodes)

# Number of edges
number_of_nodes = nodes 
edges = np.shape(g.edges)[0]    #aide à trouver les dimensions d'un tableau sur forme de turple.
#print("Num_edges: ", edges)

initial_k_percent =  10

all_nodes=range(0,nodes)
# number_entry_nodes_set = [0, 1] # Overwrite the number of entry nodes.
number_entry_nodes_set = [0,1,2]

entry_nodes = int(input("Choose Entry Nodes: <0, 1 or 2> "))

#print(" --- Entry Node is : ", entry_nodes) 
roots = [ entry_nodes ] 
for i in number_entry_nodes_set:
    # g.add_node(i, size=20, title='Entry', group=1,x=x_l[number_entry_nodes_set.index(i)],y=y_l[number_entry_nodes_set.index(i)])
    g.add_node(i, size=30, title='Entry', group=1)

for jj in Target_nodes:
    g.add_node(jj, size=35, title='Target', group=3)

intermediate_nodes = list(set.difference(set(all_nodes),set.union(set(number_entry_nodes_set),set(Target_nodes))))

#intermediate_nodes = list(np.delete(intermediate_nodes, 4))

#print("intermed: ",intermediate_nodes)
#print("all_nodes: ",all_nodes)
#print("target nodes: ", Target_nodes)

for jj in intermediate_nodes:
    g.add_node(jj, size=24,title='Internode', group=2)
print("g.nodes: ", g.nodes)

#intermediate_nodes = [3, 4, 5, 6, 9, 10, 11, 12]
intermediate_nodes_values= [g3.degree(i) * 10 for i in intermediate_nodes]
#intermediate_nodes_values = [40, 40, 30, 30, 30, 60, 40, 40, 20, 10, 20,60,20,30,30]
nodes_set = number_entry_nodes_set+ intermediate_nodes + Target_nodes
values = number_entry_nodes_set+intermediate_nodes_values + Target_nodes_values

#print(intermediate_nodes_values)

leaves = Target_nodes.copy()

Cap = 9
Esc = 3
Cd = 5 
Ca = 2

all_paths_1 = []

g_1 = [g, g1, g2, g3]

c=0      

for gg in g_1:
    all_paths = []
    for root in roots:
        for leaf in leaves:
            paths = nx.all_simple_paths(gg, root, leaf)
            all_paths.extend(paths)
            # Vérification si les nœuds existent dans le graphe
    all_paths_1.append(all_paths)

reward = []
reward1 = []

number_of_allocations = 0 
v2 = g.edges # allocation 

List_combinations = []

for j in range(1,numb_of_honeypots+1):  #retourne une liste de nombres en prenant de 1 a 3 entiers : 
        #print(i)
    number_of_allocations = number_of_allocations + ncr(len(v2),j)
    List_combinations.extend(list(combinations(v2,j)))


for i in range(len(all_paths_1)):
    
    num_states=len(all_paths_1[i]) # total number of possible pathes taken by the attacker. 
    
    v1 = all_paths_1[i].copy() #state
    
    # Build the game matrix
    rew_def = np.zeros([number_of_allocations,num_states],dtype=float)

    index1 = 0
    
    for attacK_path in all_paths_1[i]:
       # print(attacK_path)
        if type(attacK_path) == int:
            attacK_path=[attacK_path]
        index2 = 0
        for act_def in List_combinations:
            # print(act_def)
            Escape_reward = 0
            Capture_reward = 0
            if attacK_path == list(np.zeros(len(attacK_path))):
                Escape_reward = 0
                Capture_reward = 0
                Cd = 0
                Ca = 0
            else:
                for i in range(0,len(attacK_path)-1):
                    vv = attacK_path[i]
                    uu = attacK_path[i+1]
                    protected_nodes =[]
                    compromised_nodes=[]
                    #print(vv,uu)
                    node_index= nodes_set.index(uu)
                    node_index_1 = nodes_set.index(vv)
                    # print("node_index: ",node_index)
                    if ((vv,uu) in act_def):
                        protected_nodes.append(uu)
                        Capture_reward = Capture_reward + (values[node_index] + values[node_index_1]) * Cap 
                        #print('Captured')
                    else:
                        # Escape
                        #print('Escaped') 
                        compromised_nodes.append(uu)
                        Escape_reward = Escape_reward + (values[node_index] + values[node_index_1]) * Esc 
            rew_def[index2,index1] = Capture_reward - Escape_reward - Cd * len(act_def) + Ca * len(attacK_path)
            index2 +=1
        index1 +=1
    
#    print("valeur de alpha", alpha2[i])
    
    reward1.append(rew_def)

alpha2 = [0.25, 0.25, 0.5]
#alpha2 = [0.25, 0.25, 0.25, 0.25]
#alpha2 = [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]
#alpha2 = [0.4285714285714286, 0.4285714285714286, 0.14285714285714288]

# Trouver la taille maximale des matrices
max_rows = max(matrix.shape[0] for matrix in reward1)
max_cols = max(matrix.shape[1] for matrix in reward1)

# Compléter chaque matrice avec des zéros
reward1_padded = []
for matrix in reward1:
    padded_matrix = np.zeros((max_rows, max_cols))
    padded_matrix[:matrix.shape[0], :matrix.shape[1]] = matrix
    reward1_padded.append(padded_matrix)
    
# Addition pondérée des matrices de même taille
#rew = alpha2[0]*reward1_padded[0] + alpha2[1]*reward1_padded[1] + alpha2[2]*reward1_padded[2] + alpha2[3]*reward1_padded[3] 
rew = alpha2[0]*reward1_padded[1] + alpha2[1]*reward1_padded[2] + alpha2[2]*reward1_padded[3] 

# jeu de nash pour l'objectif 1
game_g1 = nash.Game(reward1[1])
# jeu de nash pour l'objectif 2
game_g2 = nash.Game(reward1[2])
# jeu de nash pour l'objectif 3
game_g3 = nash.Game(reward1[3])

# pareto-nash 
game1 = nash.Game(rew)
game10 = nash.Game(rew) 
# iterations pour chaque objectifs pour leur jeu de nash 
iterations1=100

# iterations pour le pareto-nash
iterations10=100
np.random.seed(0)

# résolution sur chaque objectifs avec leur nash 
play_counts_g1 = tuple(game_g1.fictitious_play(iterations=iterations1))
play_counts_g2 = tuple(game_g2.fictitious_play(iterations=iterations1))
play_counts_g3 = tuple(game_g3.fictitious_play(iterations=iterations1))

equilibria_g1 = play_counts_g1[-1]
equilibria_g2 = play_counts_g2[-1]
equilibria_g3 = play_counts_g3[-1]

def_mixed_strategy_g1 = equilibria_g1[0].copy()/iterations1
att_mixed_strategy_g1 = equilibria_g1[1].copy()/iterations1

def_mixed_strategy_g2 = equilibria_g2[0].copy()/iterations1
att_mixed_strategy_g2 = equilibria_g2[1].copy()/iterations1

def_mixed_strategy_g3 = equilibria_g3[0].copy()/iterations1
att_mixed_strategy_g3 = equilibria_g3[1].copy()/iterations1


play_counts = tuple(game1.fictitious_play(iterations=iterations10))
play_counts10 = tuple(game10.fictitious_play(iterations=iterations10))
equilibria1 = play_counts[-1]

def_mixed_strategy = equilibria1[0].copy()/iterations10
att_mixed_strategy = equilibria1[1].copy()/iterations10

# Calcul des expected payoffs pour l'attaquant et le défenseur
reward_game_g1 = game_g1[def_mixed_strategy, att_mixed_strategy_g1]
reward_game_g2 = game_g2[def_mixed_strategy, att_mixed_strategy_g2]
reward_game_g3 = game_g3[def_mixed_strategy, att_mixed_strategy_g3]

reward_game_g11 = game_g1[def_mixed_strategy_g1, att_mixed_strategy_g1]
reward_game_g111 = game_g1[def_mixed_strategy_g2, att_mixed_strategy_g1]
reward_game_g1111 = game_g1[def_mixed_strategy_g3, att_mixed_strategy_g1]

reward_game_g22 = game_g2[def_mixed_strategy_g1, att_mixed_strategy_g2]
reward_game_g222 = game_g2[def_mixed_strategy_g2, att_mixed_strategy_g2]
reward_game_g2222 = game_g2[def_mixed_strategy_g3, att_mixed_strategy_g2]

reward_game_g33 = game_g3[def_mixed_strategy_g1, att_mixed_strategy_g3]
reward_game_g333 = game_g3[def_mixed_strategy_g2, att_mixed_strategy_g3]
reward_game_g3333 = game_g3[def_mixed_strategy_g3, att_mixed_strategy_g3]

reward_game1 = game1[def_mixed_strategy, att_mixed_strategy]

attacker_O1 = [reward_game_g1[1], reward_game_g11[1], reward_game_g111[1], reward_game_g1111[1]]
attacker_O2 = [reward_game_g2[1], reward_game_g22[1], reward_game_g222[1], reward_game_g2222[1]]
attacker_O3 = [reward_game_g3[1], reward_game_g33[1], reward_game_g333[1], reward_game_g3333[1]]


print('graphe1:', attacker_O1)
print('graphe2:', attacker_O2)
print('graphe3:', attacker_O3)

