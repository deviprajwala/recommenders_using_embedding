#implementation embedded recommender system
print("hi")
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Embedding 
import matplotlib.pyplot as plt

def plot_graph(output_array):
    print("oo")
    x = []
    y = []
    #initialisation of the list and the string

    for i in output_array:
        y.append( i[0][0] )
        #similarity values are appended to the list y
        x.append(i[0][1])
        
        '''the two items are taken as strings and added and these values are appended to the list x, the item values are incremented by 1 as the item
        number begins from zero in our program'''
        #a ='item'
        #reinitialisation of string
    print(x,y)
    ax = plt.gca()  # gca stands for 'get current axis'
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data',0))

    plt.scatter(x, y,color=['red','green','pink','purple','blue'])
    plt.show()
    #to display the graph

model = Sequential()
model.add(Embedding(5, 2, input_length=1))
rate = pd.read_csv("embed_input.csv")

rate.drop ( 'brands', axis='columns', inplace=True )
print(rate)
input_array = np.array(rate)
print(input_array)
model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)

plot_graph(output_array)