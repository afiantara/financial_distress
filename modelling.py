import numpy
import pandas as pd
import keras
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense,Input,Dropout
from keras.utils import to_categorical
from scikeras.wrappers import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE=10
NB_START_EPOCHS=150

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

def training_kfold(df):
    df=df.drop(['CIU','flag'], axis=1)
    print(df)
    X, y = df.iloc[:, 2:-1].values, df.iloc[:, -1].values

    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_y = encoder.transform(y)
    #convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = to_categorical(encoded_y)
    
    # define 10-fold cross validation test harness
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    cvscores = []
    for train, test in kfold.split(X,  dummy_y):
        # create model
        model = baseline_model()
        # Fit the model
        model.fit(X[train], dummy_y[train], epochs=150, batch_size=10, verbose=2)
        # evaluate the model
        scores = model.evaluate(X[test], dummy_y[test], verbose=2)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
        print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    
def training_automatic_verification(df):
    df=df.drop(['CIU','flag'], axis=1)
    print(df)
    X, y = df.iloc[:, 2:-1].values, df.iloc[:, -1].values
    
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_y = encoder.transform(y)
    #convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = to_categorical(encoded_y)
    model = baseline_model()
    # Fit the model
    history = model.fit(X, dummy_y, validation_split=0.2, epochs=150, batch_size=10)
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
def training(df):
    df=df.drop(['CIU','flag'], axis=1)
    print(df)
    X, y = df.iloc[:, 2:-1].values, df.iloc[:, -1].values
    #print(*y, sep=', ')
    X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3,
    stratify=y,
    random_state=0)
    
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_y = encoder.transform(y_train)
    #convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = to_categorical(encoded_y)
    
    encoder = LabelEncoder()
    encoder.fit(y_test)
    encoded_y = encoder.transform(y_test)
    #convert integers to dummy variables (i.e. one hot encoded)
    dummy_y_test = to_categorical(encoded_y)
    
    '''
    estimator = KerasClassifier(model=baseline_model,epochs=100, batch_size=5, verbose=1)
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

    #I print the mean accuracy and its standard deviation
    results = cross_val_score(estimator, X_train, dummy_y, cv=kfold)
    print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100,
    results.std()*100))    
    '''
    model = baseline_model()
    # Fit the model
    
    #history = model.fit(X_train, dummy_y, validation_split=0.33, epochs=150, batch_size=10, shuffle=True,verbose=2)
    history = model.fit(X_train, dummy_y, validation_data=(X_test, dummy_y_test), epochs=NB_START_EPOCHS, batch_size=BATCH_SIZE, verbose=2)
    #eval_metric(model,history,'accuracy')
    #eval_metric(model,history,'loss')
    
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def estimatorKerasClassifier(df):
    df=df.drop(['CIU','flag'], axis=1)
    print(df)
    X, y = df.iloc[:, 2:-1].values, df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3,
    stratify=y,
    random_state=0)
    
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_y = encoder.transform(y_train)
    #convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = to_categorical(encoded_y)
    
    estimator = KerasClassifier(build_fn=baseline_model,
    epochs=100, batch_size=5, verbose=0)
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    results = cross_val_score(estimator, X_train, dummy_y, cv=kfold)

    print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100,
    results.std()*100))

def baseline_model():
    # create model
    model = Sequential()
    model.add(Input(shape=(15,)))
    model.add(Dropout(0.2))
    model.add(Dense(30, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(4, activation='softmax'))
    model.summary()
    
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam',
    metrics=['accuracy'])
    dot_img_file = 'model_1.png'
    keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
    
    return model


def eval_metric(model, history, metric_name):
    '''
    Function to evaluate a trained model on a chosen metric. 
    Training and validation metric are plotted in a
    line chart for each epoch.
    
    Parameters:
        history : model training history
        metric_name : loss or accuracy
    Output:
        line chart with epochs of x-axis and metric on
        y-axis
    '''
    metric = history.history[metric_name]
    val_metric = history.history['val_' + metric_name]
    e = range(1, NB_START_EPOCHS + 1)
    plt.plot(e, metric, 'bo', label='Train ' + metric_name)
    plt.plot(e, val_metric, 'b', label='Validation ' + metric_name)
    plt.xlabel('Epoch number')
    plt.ylabel(metric_name)
    plt.title('Comparing training and validation ' + metric_name + ' for ' + model.name)
    plt.legend()
    plt.show()
    
def test_model(model, X_train, y_train, X_test, y_test, epoch_stop):
    '''
    Function to test the model on new data after training it
    on the full training data with the optimal number of epochs.
    
    Parameters:
        model : trained model
        X_train : training features
        y_train : training target
        X_test : test features
        y_test : test target
        epochs : optimal number of epochs
    Output:
        test accuracy and test loss
    '''
    model.fit(X_train
              , y_train
              , epochs=epoch_stop
              , batch_size=BATCH_SIZE
              , verbose=0)
    results = model.evaluate(X_test, y_test)
    print('Test accuracy: {0:.2f}%'.format(results[1]*100))
    return results

def predict(model, x):
    predictions = model.predict(x)
    print([round(x[0]) for x in predictions])
    
def save_model(model):
    # save model and architecture to single file
    model.save("model.h5")
    print("Saved model to disk")
    
def load_model(df):
    from tensorflow.keras.models import load_model
    # load model
    model = load_model('model.h5')
    # summarize model.
    model.summary()
    
    df=df.drop(['CIU','flag'], axis=1)
    print(df)
    X, y = df.iloc[:, 2:-1].values, df.iloc[:, -1].values
    
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_y = encoder.transform(y)
    #convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = to_categorical(encoded_y)
    
    # evaluate the model
    score = model.evaluate(X, dummy_y, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
