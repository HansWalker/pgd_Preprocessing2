# pgd_Preprocessing

Basic File structure is as follows:
Blur is the main directory and contains the main training and evaluation files.
train_sargan trains the auto-encoder with various different datasets

train pgd trains hte classifier
the nat or -n tag indicates that the classifier was naturally trained without a pgd attack

the eval files evaluate the classifier with the noise and autoencoder
-Note only the mnist dataset is currently up to date.

the sar_data folder contains the various data sets. They should download automattically

trained_models contains the trained auto=encoders
sargan_dep contains the files need to train and the the auto-encoder
pgd contains the files need to train and run the classifiers, as well as doing a pgd attack
-Of note is that models folder contains the classifiers
-Each ata set also has its own config file. This contain shte various parameters needed to run an dtrain the classifier (number of training examples, training batch size, etc...)

outputs contains sample images during traing of the auto-encoder

