# ABSTRACT
The model explores using machine learning technic to perform deep learning on natural gas industry trends and consumer demands, and predict natural gas future prices movement. It experiments the possibility to use advanced computer algorithm in structural modeling to predict human behavior, such as the pricing of a cyclical commodity. It will also compare the model result from different machine learning algorithms, along with statistical modeling. Please see the details in model document in model root directory. 

# How to use the model<br>
model.testing.py, run the file to use currently trained neural network to run testing data to predict price movement since 2020.

ng.nn.model.v1.py, run the file to retrain the model from scratch, a subdirectory \models will be needed in order not to overwrite the existing trained parameters.

ng.corr.testing.py, side script to test the relationship in the data set, not used by models.
