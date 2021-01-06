# Machine-Learning-Models
A collection of machine learning models implemented on Google Colaboratory using Python and libraries such as Tensorflow, Keras, and Matplotlib.

## Model Synopsis

- **Celeb_Face_Categorizer.ipynb:**
  - Training a classifier to classify celebrity photos jointly using four 4 labels with 10000 photos to later make predictions on a specific set of labels.
  - Model was successful in predictions with over **85% accuracy**.
  - *Some Libraries Used:* Pandas, TensorFlow, Matplotlib, NumPy.
  
- **Celebrity_Face_Classification (VGG16):**
  - Using photos from Large-scale CelebFaces Attributes (CelebA) Dataset to train ConvNets that can classify photos to male celebrities and female celebrities accurately.
  - To make this problem more challenging, we only use 2,000 photos for training, 1,000 photos for validation, and 1,000 photos for testing.
  - Utilized Data Augmentation to fight Overfitting.
  - Observed a **94.6%** accuracy with the limited amount of training data samples
  - Implemented pre-trained Convolutional neural network **VGG16**.
  

