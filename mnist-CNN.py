import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')


trainSet = pd.read_csv("mnist_train.csv")
testSet = pd.read_csv("mnist_test.csv")

Y_train = trainSet["label"]
X_train = trainSet.drop(labels=["label"] , axis = 1)

testSetLabel = testSet["label"]
testSet = testSet.drop(labels=["label"] , axis = 1)

#Visualize Some Data

plt.figure(figsize=(15,7))
valueList = Y_train.value_counts()
sns.countplot(Y_train , palette="icefire")
plt.show()

#Visualize Examples
img1 = X_train.loc[[0]].to_numpy()
img1 = img1.reshape((28,28))
plt.imshow(img1,cmap='gray')
plt.axis("off")
plt.show()

img2 = X_train.loc[[550]].to_numpy()
img2 = img2.reshape((28,28))
plt.imshow(img2,cmap='gray')
plt.axis("off")
plt.show()

#Normalization 
X_train = X_train / 255.0
testSet = testSet / 255.0

#Reshape
X_train = X_train.values.reshape(-1,28,28,1)
testSetData = testSet.values.reshape((-1,28,28,1))

#OneHotEncoding
from keras.utils.np_utils import to_categorical
Y_train = to_categorical(Y_train , num_classes=10)

#Train , Test 
X_train , X_val , Y_train , Y_val = train_test_split(X_train , Y_train , test_size = 0.05 , random_state = 25)


#Building a Model
from keras.models import Sequential
from keras.layers import Dense , Conv2D , MaxPooling2D , Dropout ,Flatten
from keras.preprocessing.image import ImageDataGenerator

#CNN - Step 1 
model = Sequential()
model.add(Conv2D(filters = 16 , kernel_size = (5,5) , padding = "same" , activation='relu', input_shape=(28,28,1) , name = "Conv-Input"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#Step 2
model.add(Conv2D(filters = 16 , kernel_size = (3,3) , padding = "same" , activation='relu', name = "Conv-Hidden-1"))
model.add(MaxPooling2D(pool_size=(2,2) , strides=(2,2)))
model.add(Dropout(0.25))

#Step 3 
model.add(Conv2D(filters = 16 , kernel_size = (3,3) , padding = "same" , activation='relu', name = "Conv-Hidden-2"))
model.add(MaxPooling2D(pool_size=(2,2) , strides=(2,2)))
model.add(Dropout(0.25))

#DL 
model.add(Flatten())
model.add(Dense(256, activation = "relu" , name = "AI-Neural-Input"))
model.add(Dropout(0.25))
model.add(Dense(10 , activation= "softmax" , name = "AI-Neural-Output"))


model.compile(optimizer = "adam" , loss="categorical_crossentropy" , metrics=["accuracy"])

EPOCHS = 20
BATCH_SIZE = 125

imgDataGenerator = ImageDataGenerator(
        featurewise_center=False,  
        samplewise_center=False,  
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
        zca_whitening=False,  
        rotation_range=5,  
        zoom_range = 0.1, 
        width_shift_range=0.1,  
        height_shift_range=0.1,  
        horizontal_flip=False,  
        vertical_flip=False)  


imgDataGenerator.fit(X_train)

hist = model.fit_generator(imgDataGenerator.flow(X_train , Y_train , batch_size=BATCH_SIZE) , validation_data = (X_val,Y_val) ,epochs=EPOCHS, steps_per_epoch=X_train.shape[0] // BATCH_SIZE)

#Confussion Matrix 
predictions = model.predict(X_val)
predictions_classes = np.argmax(predictions,axis = 1) 
truePred = np.argmax(Y_val,axis = 1) 
confusion_mtx = confusion_matrix(truePred , predictions_classes) 

f,ax = plt.subplots(figsize=(8, 8))

sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

plt.plot(hist.history['val_accuracy'], color='b', label="validation accuracy")
plt.title("Test Accuracy")
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

#Predict Test Data Set
predictions2 = model.predict(testSetData)
predictions2_classes = np.argmax(predictions2 , axis = 1)
predictions2List = np.argmax(predictions2 , axis = 1).tolist()


#Check Result
img1 = testSet.loc[[0]].to_numpy()
img1 = img1.reshape((28,28))
plt.imshow(img1,cmap='gray')
plt.title('Prediction : {}'.format(predictions2List[0]))
plt.axis("off")
plt.show()

img2 = testSet.loc[[456]].to_numpy()
img2 = img2.reshape((28,28))
plt.imshow(img2,cmap='gray')
plt.title('Prediction : {}'.format(predictions2List[456]))
plt.axis("off")
plt.show()











