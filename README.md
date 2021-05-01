# recommenders_using_embedding

The repositary consists of the program which the implimentation of the recommender systems using the embeddings.It also consists of a csv file from which the input is read, a graph 
which is obtained by plotting the feature along the two dimensions and a txt file which has the information regarding the input data.

The aim of the program is to recommend the application whih is relevant to the user by making use of the embeded values, the method used for making the prediction is the dot product.
Initially we get the value for the user and append it to the csv file, we make use of the sequential model and the embeding layes to generate the embeddings for the features of the
apllication.Once we get the embeded values we plot the graph by taking the values along the x axis and y axis. Later we calculate the dot product between the user and the applications
the application with the highest value is the one which has to be recommmended.In this way we recommend an application to the user.

References:
https://stats.stackexchange.com/questions/270546/how-does-keras-embedding-layer-work
https://drop.engineering/building-a-recommender-system-using-embeddings-de5a30e655aa
https://www.youtube.com/watch?v=sZGuyTLjsco
