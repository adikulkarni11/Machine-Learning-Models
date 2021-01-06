# Machine-Learning-Models
A collection of machine learning models implemented on Google Colaboratory using Python and libraries such as Tensorflow, Keras, and Matplotlib.


## Model Synopsis

- **Celeb_Face_Categorizer.ipynb:** Celebrity FaceCategorizer
  - Training a classifier to classify celebrity photos jointly using four 4 labels with 10000 photos to later make predictions on a specific set of labels.
  - Model was successful in predictions with over **85% accuracy**.
  - *Some Libraries Used:* Pandas, TensorFlow, Matplotlib, NumPy.
  
- **Celebrity_Face_Classification.ipynb:** Celebrity_Face_Classification (VGG16)
  - Using photos from Large-scale CelebFaces Attributes (CelebA) Dataset to train ConvNets that can classify photos to male celebrities and female celebrities accurately.
  - To make this problem more challenging, we only use 2,000 photos for training, 1,000 photos for validation, and 1,000 photos for testing.
  - Utilized Data Augmentation to fight Overfitting.
  - Observed a **94.6%** accuracy with the limited amount of training data samples
  - Implemented pre-trained Convolutional neural network **VGG16**.
  
- **ECG_Classification_w_LSTM_&_Conv1D.ipynb**: ECG Classification with LSTM and Conv1D
  - Classify ECG signals from two people.
  - Train and Test of classifier is done using 1-second segments of the ECG signals.
  - Model evaluates to 95.64% test accuracy.
  - We use both Conv1D and LSTM models and compare their performances.
  
- **Fashion_MNIST_MLP_Classifer.ipynb**: Fashion MNIST Classification using Multilayer perceptron
  - Training a custom MLP Classifier to identify if it outperforms other methods.
  - Also use PCA to reduce the dimensionality of the dataset while making sure to preserve 95% of the explained variance.
  - Observed an **88%** accuracy on the reduced test set of the final model with a 0.3425 loss.

- **Multi_Class_Classification_using_GloVe_(Reuters_Newswire).ipynb:** Classifying Newswires- a multi-class classification example
  - Build a network to classify Reuters newswires into 46 different mutually-exclusive topics. 
  - Working with Reuters Dataset, a set of short newswires and their topics that were published by Reuters in 1986.
  - Data used is the 10,000 most frequently occurring words found in set.
  - Ended with test accuracy of **83%**.
  
- **Word_Embeddings_w__IMDB_Reviews.ipynb:** Word Embeddings with IMDB Reviews
  - Classifying IMBD Movie Reviews into Postive or Negative Reviews.
  - We evaluate GloVe, dense, and LSTM embeddings in this model.
  - Improved versions of pre-trained GloVe word embeddings (Global Vectors for Word Representation developed by Stanford) are used to convert text into tensors.
  - Beat **90%** Final Accuracy with a relatively simple embedding model.
 
 
 ---
  Primary Resources: Hands-On Machine Learning with Scikit-Learn and TensorFlow by Aurélien Géron, https://www.kaggle.com/
 ---
 
 
  

