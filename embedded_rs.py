#implementation of embedded recommender system

import numpy as np
import pandas as pd
import csv
from keras.models import Sequential
from keras.layers import Embedding 
import matplotlib.pyplot as plt
from csv import writer
#importing the modules needed for performing the operations

def write_new_user():
    # function to get the value for the user and append it to the csv file
    
    print("Enter the choise of user\n")
    print("1 for food\n 2 for groceries\n 3 for cloth\n 4 for chocolate\n 5 for plants\n")
    
    val=input()
    row = ['user',val]
    #get the list for inserting it into csv file
  
    f = open('embed_input.csv', 'a')
    # open the file in the write mode
    
    writer = csv.writer(f)
    # create the csv writer
    
    writer.writerow(row)
    # write a row to the csv file
    
    f.close()
	# close the file

def plot_graph(output_array):
    #function to plot the graph. Here the values which are taken along the axis are the values which are embedded to the two dimensions
    
    x = []
    y = []
    n = ['app1','app2','app3','app4','app5','user']
    #initialisation of the list

    for i in output_array:
        y.append( i[0][0] )
        x.append( i[0][1] )
    #appending the values to the list x an y for plotting the graph    
    
    
    plt.grid()
    #for displaying the grid in the graph

    plt.scatter(x, y,color=['red','green','pink','purple','blue','yellow'],marker = "^")
    #to plot the points in the graph at particular location with the given color

    for i, txt in enumerate(n):
        plt.annotate(txt, (x[i], y[i]))
    #for displaying the string along with the points in the graph

    plt.xlabel('feature1')
    #for the label along the x axis

    plt.ylabel('feature2')
    #for the label along the y axis

    plt.title('graph showing similarity of applications and a user')
    #for the title of the graph

    plt.show()
    #to display the graph

def dot_product(a1,a2,b1,b2,index,simi):
    #function to calculate the dot product for the given values

    ans = (a1 * b1) + (a2 * b2)
    #dot product is calculated

    simi.append([ans,index])
    #the calculated dot product is appended to the list along with index

def predict(embedings):
    #function to predict which application has to be recommended to the user

    simi = [ ]
    #initialisation of the list

    a1 = embedings[5][0][0]
    a2 = embedings[5][0][1]
    #embeded values which are calculated for the user

    for count,val in enumerate(embedings):
        if count<5 :
            #since the last entry in embedings is of the user so we are not considering

            b1 = val[0][0]
            b2 = val[0][1]
            #embeded values of the applications

            dot_product(a1,a2,b1,b2,count,simi)
            #function call to calculate the dot product

    max_value = max( simi )
    #the maximum similarity is computed

    max_index = simi.index( max_value )
    #index of the maximum value is calculated
    
    print("the application ",max_index+1,"can be recommended")
    #to print the relevant application to the user based on the dot product

def main():
    write_new_user()
    # function call to get the value for the user and append it to the csv file

    model = Sequential()
    #sequential model

    model.add(Embedding(6, 2, input_length=1))
    #getting the embeded layer for maximum 6 inputs to the dimention of two for the one input

    rate = pd.read_csv("embed_input.csv")
    #reading the csv file

    rate.drop ( 'application', axis='columns', inplace=True )
    #droping the first column from the data frame

    input_array = np.array(rate)
    #getting the nupy array which is sent as argument to the predict function

    model.compile('rmsprop', 'mse')
    #compiling the model here rms is root mean square and mse is mean square error

    output_array = model.predict(input_array)
    #the output array containd the embeded values which is further used to plot graph and predict

    plot_graph(output_array)
    #function to plot the graph. Here the values which are taken along the axis are the values which are embedded to the two dimensions
    
    predict(output_array)
    #function to predict which application has to be recommended to the user

main()