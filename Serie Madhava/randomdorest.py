# Import the necessary dependencies
import multiprocess
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# We create a list with the parameters to be evaluated 
hyperparameters = []
for criterion in ['gini','entropy']:
    for trees in range(10, 210):
        hyperparameters.append([trees, criterion])

def evaluate_set(hyperparameter_set, p_id, lock):
    """
    Evaluate a set of hyperparameters
    Args:
    hyperparameter_set: a list with the set of hyperparameters to be evaluated
    """
    import datetime
    print('Yo soy el proceso', p_id, 'Comence a las',datetime.datetime.now())
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    # We load the dataset, here we use 80-20 for training and testing splits
    iris=datasets.load_iris()
    X=iris.data
    y=iris.target
    # se particiona el conjunto en 80-20 para la evaluación
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        stratify=y, 
                                                        test_size=0.20)
    for s in hyperparameter_set:
        clf=RandomForestClassifier(n_estimators=int(s[0]), criterion=s[1])
        clf.fit(X_train, y_train)
        y_pred=clf.predict(X_test)
        lock.acquire()
        print('Accuracy en el proceso',p_id,':',accuracy_score(y_test,y_pred))
        lock.release()
        
        
# Now we will evaluated with more threads
if __name__ == '__main__':   
    threads=[]
    N_THREADS=8
    splits=np.split(np.array(hyperparameters), N_THREADS)
    lock=multiprocess.Lock()
    for i in range(N_THREADS-1):
        # Se generan los hilos de procesamiento
        threads.append(multiprocess.Process(target=evaluate_set, args=[splits[i],i, lock]))


    start_time = time.perf_counter()
    # Se lanzan a ejecución
    for thread in threads:
        thread.start()

    # y se espera a que todos terminen
    for thread in threads:
        thread.join()
                
    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time-start_time} seconds")