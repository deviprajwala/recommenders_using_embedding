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
    print("Enter the choise of user\n")
    print("1 for food\n 2 for groceries\n 3 for cloth\n 4 for chocolate\n 5 for plants\n")
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
    n = ['app1','app2','app3','app4','app5','user']
    #initialisation of the list and the string

    for i in output_array:
        y.append( i[0][0] )
        x.append( i[0][1] )
        
    
    
    plt.grid()
    plt.scatter(x, y,color=['red','green','pink','purple','blue','yellow'],marker = "^")
    for i, txt in enumerate(n):
        plt.annotate(txt, (x[i], y[i]))
    plt.show()
    #to display the graph

def dot_product(a1,a2,b1,b2,index,simi):

  ans = (a1 * b1) + (a2 * b2)
  simi.append([ans,index])


def predict(embedings):
    simi = [ ]
    a1 = embedings[5][0][0]
    a2 = embedings[5][0][1]
    for count,val in enumerate(embedings):
        if count<5 :
            b1 = val[0][0]
            b2 = val[0][1]
            dot_product(a1,a2,b1,b2,count,simi)
    #print(simi)
    max_value = max( simi )
    #the maximum similarity is computed

    max_index = simi.index( max_value )
    #index of the maximum value is calculated
    
    print("the application ",max_index+1,"can be recommended")

#write_new_user()
model = Sequential()
model.add(Embedding(6, 2, input_length=1))
rate = pd.read_csv("embed_input.csv")

rate.drop ( 'application', axis='columns', inplace=True )
print(rate)
input_array = np.array(rate)
print(input_array)
model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)

plot_graph(output_array)
#print(output_array)

predict(output_array)