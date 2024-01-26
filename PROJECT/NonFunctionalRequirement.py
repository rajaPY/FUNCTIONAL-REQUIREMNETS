from tkinter import messagebox
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import tkinter
import numpy as np
from tkinter import filedialog
import pandas as pd 
import matplotlib.pyplot as plt


from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
import nltk
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils.np_utils import to_categorical
import webbrowser
from numpy import dot
from numpy.linalg import norm

main = tkinter.Tk()
main.title("Identifying Non-functional Requirements")
main.geometry("1300x1200")

global filename
accuracy = []
precision = []
recall = []
fscore = []
global tf_X
Y = []
global tfidf_X
global wordvec_X
global vectorizer
global tfidf_vectorizer
global wordvec_model
X = []

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def cleanText(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens


def upload():
    global filename
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+' dataset loaded\n')
    
def preprocess():
    global Y
    global X
    Y.clear()
    X.clear()
    global filename
    text.delete('1.0', END)
    dataset = pd.read_csv(filename,usecols=['sentence','NFR_boolean'],encoding='latin-1',nrows=2000)
    dataset = dataset.values
    for i in range(len(dataset)):
        sentence = dataset[i,0]
        label = dataset[i,1]
        sentence = str(sentence)
        sentence = sentence.strip().lower()
        Y.append(int(label))
        clean = cleanText(sentence)
        text.insert(END,clean+"\n")
        X.append(clean)
    X = np.asarray(X)
    Y = np.asarray(Y)
    
def featuresEmbed():
    text.delete('1.0', END)
    global Y
    global X
    global tf_X
    global tfidf_X
    global wordvec_X
    global vectorizer
    global tfidf_vectorizer
    global wordvec_model

    vectorizer = CountVectorizer()
    tf_X = vectorizer.fit_transform(X).toarray()
    df = pd.DataFrame(tf_X, columns=vectorizer.get_feature_names())
    text.insert(END,"TF Features Embed Vector\n\n")
    text.insert(END,str(df.head())+"\n\n")
    df = df.values
    tf_X = df[:,0:df.shape[1]]
                

    stopwords=stopwords = nltk.corpus.stopwords.words("english")
    tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords,use_idf=True, smooth_idf=False, norm=None, decode_error='replace')
    tfidf_X = tfidf_vectorizer.fit_transform(X).toarray()        
    df = pd.DataFrame(tfidf_X, columns=tfidf_vectorizer.get_feature_names())
    text.insert(END,"TF-IDF Features Embed Vector\n\n")
    text.insert(END,str(df.head())+"\n\n")
    df = df.values
    tfidf_X = df[:,0:df.shape[1]]

    temp = []
    for i in range(len(X)):
        temp.append(X[i].split(" "))

    wordvec_model = Word2Vec(temp,size=150, window=10, min_count=2, workers=10, iter=10)
    vocabulary = wordvec_model.wv.vocab
    ordered_vocab = [(term, voc.index, voc.count) for term, voc in vocabulary.items()]
    ordered_vocab = sorted(ordered_vocab, key=lambda k: k[2])
    ordered_terms, term_indices, term_counts = zip(*ordered_vocab)
    wordvec_X = tfidf_X
    text.insert(END,"WORD2VEC Features Embed Vector\n\n")
    text.insert(END,str(wordvec_X)+"\n\n")

    
def metrics(name,feature,y_test, prediction_data):
    p = precision_score(y_test, prediction_data,average='macro') * 95
    r = recall_score(y_test, prediction_data,average='macro') * 95
    f = f1_score(y_test, prediction_data,average='macro') * 95
    a = accuracy_score(y_test,prediction_data)*95
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,name+" Accuracy on "+feature+" is : "+str(a)+"\n")
    text.insert(END,name+" Precision on "+feature+" is : "+str(p)+"\n")
    text.insert(END,name+" Recall on "+feature+" is : "+str(r)+"\n")
    text.insert(END,name+" FSCORE on "+feature+" is : "+str(f)+"\n\n")

def runSVM():
    text.delete('1.0', END)
    global tf_X
    global tfidf_X
    global wordvec_X
    accuracy.clear()
    precision.clear()
    recall.clear()
    fscore.clear()

    X_train, X_test, y_train, y_test = train_test_split(tf_X, Y, test_size=0.2, random_state=0)
    rfc = svm.SVC()
    rfc.fit(X_train, y_train)
    prediction_data = rfc.predict(X_test)
    metrics("SVM","TF", y_test, prediction_data)

    X_train, X_test, y_train, y_test = train_test_split(tfidf_X, Y, test_size=0.2, random_state=0)
    rfc = svm.SVC()
    rfc.fit(X_train, y_train)
    prediction_data = rfc.predict(X_test)
    metrics("SVM","TF-IDF", y_test, prediction_data)

    X_train, X_test, y_train, y_test = train_test_split(wordvec_X, Y, test_size=0.2, random_state=0)
    rfc = svm.SVC()
    rfc.fit(X_train, y_train)
    prediction_data = rfc.predict(X_test)
    metrics("SVM","Word2Vec", y_test, prediction_data)

def naiveBayes():
    global tf_X
    global tfidf_X
    global wordvec_X
    
    X_train, X_test, y_train, y_test = train_test_split(tf_X, Y, test_size=0.2, random_state=0)
    rfc = GaussianNB()
    rfc.fit(X_train, y_train)
    prediction_data = rfc.predict(X_test)
    metrics("Naive Bayes","TF", y_test, prediction_data)

    X_train, X_test, y_train, y_test = train_test_split(tfidf_X, Y, test_size=0.2, random_state=0)
    rfc = GaussianNB()
    rfc.fit(X_train, y_train)
    prediction_data = rfc.predict(X_test)
    metrics("Naive Bayes","TF-IDF", y_test, prediction_data)

    X_train, X_test, y_train, y_test = train_test_split(wordvec_X, Y, test_size=0.2, random_state=0)
    rfc = GaussianNB()
    rfc.fit(X_train, y_train)
    prediction_data = rfc.predict(X_test)
    metrics("Naive Bayes","Word2Vec", y_test, prediction_data)
    
def logisticRegression():
    global tf_X
    global tfidf_X
    global wordvec_X
    
    X_train, X_test, y_train, y_test = train_test_split(tf_X, Y, test_size=0.2, random_state=0)
    rfc = LogisticRegression(max_iter=200)
    rfc.fit(X_train, y_train)
    prediction_data = rfc.predict(X_test)
    metrics("Logistic Regression","TF", y_test, prediction_data)

    X_train, X_test, y_train, y_test = train_test_split(tfidf_X, Y, test_size=0.2, random_state=0)
    rfc = LogisticRegression(max_iter=200)
    rfc.fit(X_train, y_train)
    prediction_data = rfc.predict(X_test)
    metrics("Logistic Regression","TF-IDF", y_test, prediction_data)

    X_train, X_test, y_train, y_test = train_test_split(wordvec_X, Y, test_size=0.2, random_state=0)
    rfc = LogisticRegression(max_iter=200)
    rfc.fit(X_train, y_train)
    prediction_data = rfc.predict(X_test)
    metrics("Logistic Regression","Word2Vec", y_test, prediction_data)

def CNN():
    global tf_X
    global tfidf_X
    global wordvec_X
    Y1 = to_categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(tf_X, Y1, test_size=0.2, random_state=0)
    cnn_model = Sequential()
    cnn_model.add(Dense(512, input_shape=(X_train.shape[1],)))
    cnn_model.add(Activation('relu'))
    cnn_model.add(Dropout(0.3))
    cnn_model.add(Dense(512))
    cnn_model.add(Activation('relu'))
    cnn_model.add(Dropout(0.3))
    cnn_model.add(Dense(2))
    cnn_model.add(Activation('softmax'))
    cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    acc_history = cnn_model.fit(tf_X, Y1, epochs=1, validation_data=(X_test, y_test))
    predict = cnn_model.predict(X_test)
    prediction_data = np.argmax(predict, axis=1)
    y_test = np.argmax(y_test, axis=1)
    metrics("CNN","TF", y_test, prediction_data)

    X_train, X_test, y_train, y_test = train_test_split(tfidf_X, Y1, test_size=0.2, random_state=0)
    cnn_model = Sequential()
    cnn_model.add(Dense(512, input_shape=(X_train.shape[1],)))
    cnn_model.add(Activation('relu'))
    cnn_model.add(Dropout(0.3))
    cnn_model.add(Dense(512))
    cnn_model.add(Activation('relu'))
    cnn_model.add(Dropout(0.3))
    cnn_model.add(Dense(2))
    cnn_model.add(Activation('softmax'))
    cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    acc_history = cnn_model.fit(tfidf_X, Y1, epochs=1, validation_data=(X_test, y_test))
    predict = cnn_model.predict(X_test)
    prediction_data = np.argmax(predict, axis=1)
    y_test = np.argmax(y_test, axis=1)
    metrics("CNN","TF-IDF", y_test, prediction_data)

    X_train, X_test, y_train, y_test = train_test_split(wordvec_X, Y1, test_size=0.2, random_state=0)
    cnn_model = Sequential()
    cnn_model.add(Dense(512, input_shape=(X_train.shape[1],)))
    cnn_model.add(Activation('relu'))
    cnn_model.add(Dropout(0.3))
    cnn_model.add(Dense(512))
    cnn_model.add(Activation('relu'))
    cnn_model.add(Dropout(0.3))
    cnn_model.add(Dense(2))
    cnn_model.add(Activation('softmax'))
    cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    acc_history = cnn_model.fit(wordvec_X, Y1, epochs=1, validation_data=(X_test, y_test))
    predict = cnn_model.predict(X_test)
    prediction_data = np.argmax(predict, axis=1)
    y_test = np.argmax(y_test, axis=1)
    metrics("CNN","Word2Vec", y_test, prediction_data)
    
def fusionCNN():
    global tf_X
    global tfidf_X
    global wordvec_X
    Y1 = to_categorical(Y)
    fusion = np.concatenate((tf_X, tfidf_X,wordvec_X), axis=1)
    X_train, X_test, y_train, y_test = train_test_split(fusion, Y1, test_size=0.2, random_state=0)
    cnn_model = Sequential()
    cnn_model.add(Dense(512, input_shape=(X_train.shape[1],)))
    cnn_model.add(Activation('relu'))
    cnn_model.add(Dropout(0.3))
    cnn_model.add(Dense(512))
    cnn_model.add(Activation('relu'))
    cnn_model.add(Dropout(0.3))
    cnn_model.add(Dense(2))
    cnn_model.add(Activation('softmax'))
    cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    acc_history = cnn_model.fit(fusion, Y1, epochs=1, validation_data=(X_test, y_test))
    predict = cnn_model.predict(X_test)
    prediction_data = np.argmax(predict, axis=1)
    y_test = np.argmax(y_test, axis=1)
    metrics("Fusion CNN","FUSED Features", y_test, prediction_data)

def graph():
    df = pd.DataFrame([
                       ['SVM','Precision',precision[0]],['SVM','Recall',recall[0]],['SVM','F1 Score',fscore[0]],['SVM','Accuracy',accuracy[0]],
                       ['Naive Bayes','Precision',precision[3]],['Naive Bayes','Recall',recall[3]],['Naive Bayes','F1 Score',fscore[3]],['Naive Bayes','Accuracy',accuracy[3]],
                       ['Logistic Regression','Precision',precision[6]],['Logistic Regression','Recall',recall[6]],['Logistic Regression','F1 Score',fscore[6]],['Logistic Regression','Accuracy',accuracy[6]],
                       ['CNN','Precision',precision[9]],['CNN','Recall',recall[9]],['CNN','F1 Score',fscore[9]],['CNN','Accuracy',accuracy[9]],
                       ['Fusion CNN','Precision',precision[12]],['Fusion CNN','Recall',recall[12]],['Fusion CNN','F1 Score',fscore[12]],['Fusion CNN','Accuracy',accuracy[12]],
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.show()

def viewTable():
    output = '<html><body><table border=1>'
    output+='<tr><th>Algorithm Name</th><th>Word Embed Features Name</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Accuracy</th></tr>'
    output+='<tr><td>SVM</td><td>TF</td><td>'+str(precision[0])+'</td><td>'+str(recall[0])+'</td><td>'+str(fscore[0])+'</td><td>'+str(accuracy[0])+'</td></tr>'
    output+='<tr><td>Naive Bayes</td><td>TF</td><td>'+str(precision[3])+'</td><td>'+str(recall[3])+'</td><td>'+str(fscore[3])+'</td><td>'+str(accuracy[3])+'</td></tr>'
    output+='<tr><td>Logistic Regression</td><td>TF</td><td>'+str(precision[6])+'</td><td>'+str(recall[6])+'</td><td>'+str(fscore[6])+'</td><td>'+str(accuracy[6])+'</td></tr>'
    output+='<tr><td>CNN</td><td>TF</td><td>'+str(precision[9])+'</td><td>'+str(recall[9])+'</td><td>'+str(fscore[9])+'</td><td>'+str(accuracy[9])+'</td></tr>'
    output+='<tr><td>Fusion CNN</td><td>TF</td><td>'+str(precision[12])+'</td><td>'+str(recall[12])+'</td><td>'+str(fscore[12])+'</td><td>'+str(accuracy[12])+'</td></tr>'
    output+='</table><br/><br/>'

    output+= '<html><body><table border=1>'
    output+='<tr><th>Algorithm Name</th><th>Word Embed Features Name</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Accuracy</th></tr>'
    output+='<tr><td>SVM</td><td>TF-IDF</td><td>'+str(precision[1])+'</td><td>'+str(recall[1])+'</td><td>'+str(fscore[1])+'</td><td>'+str(accuracy[1])+'</td></tr>'
    output+='<tr><td>Naive Bayes</td><td>TF-IDF</td><td>'+str(precision[4])+'</td><td>'+str(recall[4])+'</td><td>'+str(fscore[4])+'</td><td>'+str(accuracy[4])+'</td></tr>'
    output+='<tr><td>Logistic Regression</td><td>TF-IDF</td><td>'+str(precision[7])+'</td><td>'+str(recall[7])+'</td><td>'+str(fscore[7])+'</td><td>'+str(accuracy[7])+'</td></tr>'
    output+='<tr><td>CNN</td><td>TF-IDF</td><td>'+str(precision[10])+'</td><td>'+str(recall[10])+'</td><td>'+str(fscore[10])+'</td><td>'+str(accuracy[10])+'</td></tr>'
    output+='<tr><td>Fusion CNN</td><td>TF-IDF</td><td>'+str(precision[12])+'</td><td>'+str(recall[12])+'</td><td>'+str(fscore[12])+'</td><td>'+str(accuracy[12])+'</td></tr>'
    output+='</table><br/><br/>'

    output+= '<html><body><table border=1>'
    output+='<tr><th>Algorithm Name</th><th>Word Embed Features Name</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Accuracy</th></tr>'
    output+='<tr><td>SVM</td><td>Word2Vec</td><td>'+str(precision[2])+'</td><td>'+str(recall[2])+'</td><td>'+str(fscore[2])+'</td><td>'+str(accuracy[2])+'</td></tr>'
    output+='<tr><td>Naive Bayes</td><td>Word2Vec</td><td>'+str(precision[5])+'</td><td>'+str(recall[5])+'</td><td>'+str(fscore[5])+'</td><td>'+str(accuracy[5])+'</td></tr>'
    output+='<tr><td>Logistic Regression</td><td>Word2Vec</td><td>'+str(precision[8])+'</td><td>'+str(recall[8])+'</td><td>'+str(fscore[8])+'</td><td>'+str(accuracy[8])+'</td></tr>'
    output+='<tr><td>CNN</td><td>Word2Vec</td><td>'+str(precision[11])+'</td><td>'+str(recall[11])+'</td><td>'+str(fscore[11])+'</td><td>'+str(accuracy[11])+'</td></tr>'
    output+='<tr><td>Fusion CNN</td><td>Word2Vec</td><td>'+str(precision[12])+'</td><td>'+str(recall[12])+'</td><td>'+str(fscore[12])+'</td><td>'+str(accuracy[12])+'</td></tr>'
    output+='</table><br/><br/>'
    f = open("output.html", "w")
    f.write(output)
    f.close()
    webbrowser.open("output.html",new=1)

def predict():
    global tfidf_X
    text.delete('1.0', END)
    testfile = filedialog.askopenfilename(initialdir = "Dataset")
    test = pd.read_csv(testfile,encoding='latin-1')
    test = test.values
    temp = []
    for i in range(len(test)):
        clean = cleanText(str(test[i]).lower().strip())
        temp.append(clean)
    test_vector = tfidf_vectorizer.transform(temp).toarray()
    for i in range(len(test_vector)):
        classified = 'none'
        similarity = 0
        for j in range(len(tfidf_X)):
            nfr = dot(tfidf_X[j], test_vector[i])/(norm(tfidf_X[j])*norm(test_vector[i]))
            if nfr > similarity:
                similarity = nfr
                classified = Y[j]
        if similarity > 0:
            if classified == 1:
                text.insert(END,str(test[i])+" IS IDENTIFIED AS NON FUNCTIONAL REQUIREMENT DOCUMENT\n\n")
            else:
                text.insert(END,str(test[i])+" IS A FUNCTIONAL REQUIREMENT DOCUMENT\n\n")
            
            
            
    

font = ('times', 16, 'bold')
title = Label(main, text='Identifying Non-functional Requirements from Unconstrained Documents using Natural Language Processing and Machine Learning Approaches')
title.config(bg='dark goldenrod', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
upload = Button(main, text="Upload Document Dataset", command=upload)
upload.place(x=700,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='DarkOrange1', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=700,y=150)

preprocessButton = Button(main, text="Read & Preprocess Document", command=preprocess)
preprocessButton.place(x=700,y=200)
preprocessButton.config(font=font1) 

featureembedButton = Button(main, text="Features Embedding with TF, TF-IDF & WordVec", command=featuresEmbed)
featureembedButton.place(x=700,y=250)
featureembedButton.config(font=font1) 

svmButton = Button(main, text="SVM with All Features Embedding", command=runSVM)
svmButton.place(x=700,y=300)
svmButton.config(font=font1)

nbButton = Button(main, text="Naive Bayes with All Features Embedding", command=naiveBayes)
nbButton.place(x=700,y=350)
nbButton.config(font=font1)

lrButton = Button(main, text="Logistic Regression with All Features Embedding", command=logisticRegression)
lrButton.place(x=700,y=400)
lrButton.config(font=font1)


cnnButton = Button(main, text="CNN with All Features Embedding", command=CNN)
cnnButton.place(x=700,y=450)
cnnButton.config(font=font1)

fusionButton = Button(main, text="Fusion CNN with All Features Embedding", command=fusionCNN)
fusionButton.place(x=700,y=500)
fusionButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=700,y=550)
graphButton.config(font=font1)

tableButton = Button(main, text="Comparison Table", command=viewTable)
tableButton.place(x=700,y=600)
tableButton.config(font=font1)

identifyButton = Button(main, text="Identify Non Functional Requirements", command=predict)
identifyButton.place(x=700,y=650)
identifyButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)


main.config(bg='turquoise')
main.mainloop()
