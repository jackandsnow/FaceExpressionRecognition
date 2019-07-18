# FaceExpressionRecognition

1. Datasets: CK+ database: 
    You can go to the [official website](http://www.consortium.ri.cmu.edu/ckagree/) to download `Emotion_labels.zip` and `extend-cohn-kanade-images.zip`.
    You also can go to the [Baidu network disk](https://download.csdn.net/download/jackandsnow/11210946) to download datasets.

2. Environment needed: Python 3, Tensorflow-gpu, NumPy, opencv, PIL

3. This project is mainly implemented by CNN, which is set-up by myself through tensorflow. In this project, firstly I crop out the main part of face in each picture, and then I save all pictures' data and their labels to csv files.
 Lastly I load all images' data from csv files as the CNN input. What's more, I saved the weights, losses and accuracies by tensorboard in training process.
