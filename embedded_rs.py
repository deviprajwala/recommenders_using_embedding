#implementation embedded recommender system
print("hi")
import numpy as np
import pandas as pd
import csv
from keras.models import Sequential
from keras.layers import Embedding 
import matplotlib.pyplot as plt



# Import writer class from csv module
from csv import writer

def write_new_user():
    # List
    print("Enter the choise of user")
    print("1 for food\n 2 for groceries\n 3 for cloth\n 4 for chocolate\n")
    val=input()
    row = ['user',val]


    # open the file in the write mode
    f = open('embed_input.csv', 'a')

    # create the csv writer
    writer = csv.writer(f)

    # write a row to the csv file
    writer.writerow(row)

    # close the file
    f.close()
	


def plot_graph(output_array):
    print("oo")
    x = []
    y = []
    n = ['brand1','b2','b3','b4','user']
    #initialisation of the list and the string

    for i in output_array:
        y.append( i[0][0] )
        x.append( i[0][1] )
        
    
    
    plt.grid()
    plt.legend()
    plt.scatter(x, y,color=['red','green','pink','purple','blue','black'],cmap=['a','b','c','d','e','user'])
    for i, txt in enumerate(n):
        plt.annotate(txt, (x[i], y[i]))
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

#plot_graph(output_array)
#print(output_array)
write_new_user()
print(rate)