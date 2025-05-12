from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import cv2
import random
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.applications import DenseNet169
from sklearn.metrics import accuracy_score
from keras.callbacks import ModelCheckpoint 
import pickle
import os
from keras.models import load_model
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from keras.models import load_model
from keras.applications import VGG19
from keras.applications import MobileNetV2

main = tkinter.Tk()
main.title("Remote Sensing Image Scene Classification Meets Deep Learning: Challenges, Methods, Benchmarks, and Opportunities")
main.geometry("1300x1200")

global filename
global classifier
global labels, X, Y, X_train, y_train, X_test, y_test, classifier

def readLabels(filename):
    global labels
    labels = []
    for root, dirs, directory in os.walk(filename):
        for j in range(len(directory)):
            name = os.path.basename(root)
            if name not in labels:
                labels.append(name)

def getID(name):
    index = 0
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index        

def uploadDataset():
    global filename
    global labels
    labels = []
    filename = filedialog.askdirectory(initialdir=".")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n");
    readLabels(filename)
    text.insert(END,"Remote Scenes found in dataset are\n\n")
    for i in range(len(labels)):
        text.insert(END,labels[i]+"\n")

def processDataset():
    text.delete('1.0', END)
    global filename, X, Y, X_train, y_train, X_test, y_test
    if os.path.exists("model/X.txt.npy"):
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
    else:
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):
                name = os.path.basename(root)
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root+"/"+directory[j])
                    img = cv2.resize(img, (32,32))
                    im2arr = np.array(img)
                    im2arr = im2arr.reshape(32,32,3)
                    X.append(im2arr)
                    label = getID(name)
                    Y.append(label)
                    print(name+" "+str(label))
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save('model/X.txt',X)
        np.save('model/Y.txt',Y)
    X = X.astype('float32')
    X = X/255
    text.insert(END,"Dataset Preprocessing Completed\n")
    text.insert(END,"Total images found in dataset : "+str(X.shape[0])+"\n\n")
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
    text.insert(END,"80% images are used to train VGG19 : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% images are used to test VGG19 : "+str(X_test.shape[0])+"\n")


def trainVGG19():
    text.delete('1.0', END)
    global filename, X, Y, X_train, y_train, X_test, y_test, classifier, labels
    vgg = VGG19(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), include_top=False, weights="imagenet")
    vgg.trainable = False
    classifier = Sequential()
    classifier.add(vgg)
    classifier.add(Convolution2D(32, 1, 1, input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (1, 1)))
    classifier.add(Convolution2D(32, 1, 1, activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (1, 1)))
    classifier.add(Flatten())
    classifier.add(Dense(output_dim = 256, activation = 'relu'))
    classifier.add(Dense(output_dim = y_train.shape[1], activation = 'softmax'))
    print(classifier.summary())
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/vgg_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/vgg_weights.hdf5', verbose = 1, save_best_only = True)
        hist = classifier.fit(X_train, y_train, batch_size = 32, epochs = 20, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/vgg_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        classifier = load_model("model/vgg_weights.hdf5")
    predict = classifier.predict(X_test)
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(y_test, axis=1)
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100
    text.insert(END,"VGG19 Transfer Learning Performance\n")
    text.insert(END,"VGG19 Accuracy  : "+str(a)+"\n")
    text.insert(END,"VGG19 Precision : "+str(p)+"\n")
    text.insert(END,"VGG19 Recall    : "+str(r)+"\n")
    text.insert(END,"VGG19 FSCORE    : "+str(f)+"\n\n")
    conf_matrix = confusion_matrix(testY, predict)
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title("VGG19 Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()        


def trainMobileNet():
    global filename, X, Y, X_train, y_train, X_test, y_test, classifier, labels
    #creating Pretrained MobileNetV2 object as mobilenet
    mobilenet = MobileNetV2(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), include_top=False, weights="imagenet")
    mobilenet.trainable = False
    #creating custom CNN model object as mobilenet_model
    mobilenet_model = Sequential()
    mobilenet_model.add(mobilenet)#now transfer learning mobilenet pretrained model to custome mobilenet_model object
    #now defining new training layers to transfer learning mobilenet object
    mobilenet_model.add(Convolution2D(32, 1, 1, input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    mobilenet_model.add(MaxPooling2D(pool_size = (1, 1)))
    mobilenet_model.add(Convolution2D(32, 1, 1, activation = 'relu'))
    mobilenet_model.add(MaxPooling2D(pool_size = (1, 1)))
    mobilenet_model.add(Flatten())
    mobilenet_model.add(Dense(output_dim = 256, activation = 'relu')) #define output layer
    mobilenet_model.add(Dense(output_dim = y_train.shape[1], activation = 'softmax'))
    print(mobilenet_model.summary())
    mobilenet_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    mobilenet_model = load_model("model/mobilenet_weights.hdf5")
    predict = mobilenet_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(y_test, axis=1)
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100
    text.insert(END,"MobileNetV2 Transfer Learning Performance\n")
    text.insert(END,"MobileNetV2 Accuracy  : "+str(a)+"\n")
    text.insert(END,"MobileNetV2 Precision : "+str(p)+"\n")
    text.insert(END,"MobileNetV2 Recall    : "+str(r)+"\n")
    text.insert(END,"MobileNetV2 FSCORE    : "+str(f)+"\n\n")
    conf_matrix = confusion_matrix(testY, predict)
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title("MobileNetV2 Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()
    

def graph():
    f = open('model/vgg_history.pckl', 'rb')
    graph = pickle.load(f)
    f.close()
    vgg_accuracy = graph['val_accuracy']
    vgg_error = graph['val_loss']

    f = open('model/mobilenet_history.pckl', 'rb')
    graph = pickle.load(f)
    f.close()
    mobilenet_accuracy = graph['val_accuracy']
    mobilenet_error = graph['val_loss']

    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('EPOCH')
    plt.ylabel('Accuracy/Loss')
    plt.plot(vgg_accuracy, 'ro-', color = 'green')
    plt.plot(vgg_error, 'ro-', color = 'red')
    plt.plot(mobilenet_accuracy, 'ro-', color = 'blue')
    plt.plot(mobilenet_error, 'ro-', color = 'yellow')
    plt.legend(['VGG19 Accuracy', 'VGG19 Loss', 'MobileNetV2 Accuracy', 'MobileNetV2 Loss'], loc='upper left')
    plt.title('VGG19 & MobileNetV2 Training Accuracy & Loss Graph')
    plt.show()
    

def classifyScenes():
    global classifier, labels
    filename = filedialog.askopenfilename(initialdir="testImages")
    image = cv2.imread(filename)
    img = cv2.resize(image, (32, 32))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,32,32,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255
    preds = classifier.predict(img)
    predict = np.argmax(preds)

    img = cv2.imread(filename)
    img = cv2.resize(img, (700,400))
    cv2.putText(img, 'Remote Scene Classified as : '+labels[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
    cv2.imshow('Remote Scene Classified as : '+labels[predict], img)
    cv2.waitKey(0)
    

def close():
    main.destroy()
    
    
font = ('times', 16, 'bold')
title = Label(main, text='Remote Sensing Image Scene Classification Meets Deep Learning: Challenges, Methods, Benchmarks, and Opportunities',anchor=W, justify=CENTER)
title.config(bg='DodgerBlue3', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)


font1 = ('times', 13, 'bold')
upload = Button(main, text="Upload NWPU Satellite Scenes Dataset", command=uploadDataset)
upload.place(x=50,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='yellow4', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=50,y=150)

processButton = Button(main, text="Preprocess Dataset", command=processDataset)
processButton.place(x=50,y=200)
processButton.config(font=font1)

trainButton = Button(main, text="Train Transfer Learning VGG19", command=trainVGG19)
trainButton.place(x=50,y=250)
trainButton.config(font=font1)

mobilenetButton = Button(main, text="Train Transfer Learning MobileNetV2", command=trainMobileNet)
mobilenetButton.place(x=50,y=300)
mobilenetButton.config(font=font1)

graphButton = Button(main, text="VGG19 & MobileNetV2 Training Graph", command=graph)
graphButton.place(x=50,y=350)
graphButton.config(font=font1)

classifyButton = Button(main, text="Remote Scenes Classification", command=classifyScenes)
classifyButton.place(x=50,y=400)
classifyButton.config(font=font1)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=50,y=450)
exitButton.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=25,width=78)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=370,y=100)
text.config(font=font1)


main.config(bg='LightPink1')
main.mainloop()
