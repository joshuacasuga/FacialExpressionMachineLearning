# FacialExpressionMachineLearning
Automatic human facial expression classification has lots of potential real life applications, such as customer service systems, personalized advertisement, and etertainment. 
## Task Description:
Build and train a deep neural network model for facial expression classification. The evaluation is based on categorization accuracy for ranking.
The training dataset consists of 16,175 examples to build the model. The test set contains 3,965 examples to determine the accuracy score. 
Overall performance of the model is determined by a Kaggle leaderboard. Submissions with a score >= 0.76696 receive full credit. 

## Abstract:
Human-computer interaction has become increasingly advanced in recent times. The ability for computers to recognize human facial expressions is a significant aspect pertaining to the human-computer interaction that has allowed for machines to understand and respond to human emotions. This task involves the classification of images of human faces into emotional categories such as happiness, anger, and sadness. Advancements in deep learning, particularly convolutional neural networks, have shown exceptional progress. The goal of this project is to leverage machine learning techniques to build and train a neural network model to achieve high accuracy in facial expression classification. Performance of the model is measured by a provided dataset and ranked on accuracy using a Kaggle leaderboard. The approach taken involves various neural network architectures and techniques resulting in an accuracy score that demonstrates the effectiveness of the chosen methods. 

## Objectives:
Design and train a deep learning convolutional neural network that accurately recognizes and classifies human facial expressions. It should implement concepts of neural networks that aim towards increasing accuracy. The output of the model is a comma separated value (CSV) file that stores target values (0-Angry, 1-Happy, 2-Neutral classifying images of human facial expressions. During testing, given a dataset of test points, the model should perform accurately, producing an accuracy score of 0.76696 or above. This score will be produced by Kaggle, given the model’s resulting CSV file. 

![image](https://github.com/user-attachments/assets/99e418ff-4ff9-4861-b4a9-ee2929a13296) ![image](https://github.com/user-attachments/assets/673ff9a9-2b23-4abc-b7e6-4b91bd10c235)

## Methods:
1. The first step in developing this classification model involves data preprocessing. The datasets are reshaped into the appropriate dimensions for the convolutional neural network using -1 for batch size, allowing NumPy to automatically calculate it, and reshaping into single-channel 48x48 images. The pixel values are normalized by dividing by 255, scaling them to a range of 0 to 1, which improves performance. The target training data is converted into one-hot encoding for multi-class classification. All datasets are then converted into PyTorch tensors and data loaders. 

2. The model includes 2D convolutional layers, pooling layers, normalization layers, dropout layers, and fully connected layers. Each convolutional layer is followed by a pooling layer to subsample the images, reducing the number of parameters. Dropout layers are used to improve training by forcing the network to learn redundant patterns. Fully connected layers pass parameters through the network, resulting in a classification. Normalization layers are included to normalize inputs between layers.

3. This model implements the cross entropy loss function, commonly used for classification tasks. The Adam optimization algorithm was chosen for its adaptive learning rate capabilities. The algorithm begins with a learning rate of 0.001 which is a standard baseline value. A learning rate scheduler is used to reduce the learning rate if there is no improvement in loss after 5 epochs. 

4. Training the model is straightforward as it follows the typical machine learning training loop. However, early stopping was implemented to stop training early if the model converges before all epochs are trained. The model is trained on 50 epochs, allowing enough time for the possibility of early convergence. 

## Results:
![image](https://github.com/user-attachments/assets/1b2fd9e2-b492-42a6-884f-e8dc8c09ab39)

## Conclusions:
 - Understanding the problem will allow for the correct decisions to be made when designing a neural network
 - Data preprocessing is especially important for image classification tasks due to the number of pixels being passed through the network for learning
 - Deep learning techniques such as pooling layers, normalization layers, and dropout layers help improve performance of neural networks
 - Choosing the correct loss function for your task is crucial for producing accurate loss scores
 - Fine tuning your model by changing the learning rate and number of epochs prevents overfitting/underfitting
 - Using an adaptive learning rate scheduler is an efficient way to modify the learning rate automatically
 - Early stopping saves time and processing power by stopping training once the model has converged
