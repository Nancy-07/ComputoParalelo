import multiprocess
import time
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
# penalización  = list(np.arange(0.1, 1, 0.1)) + list(range(1, 51)) + list(range(100, 501, 50))
penalización  = list(np.arange(0.1, 1, 0.1)) + list(range(1, 51))

# tols =  [1e-3, 1e-4, 1e-5]
tols =  [1e-3]

gammas = ['scale', 'auto']

hyperparameters = []
for kernel in kernels:
    for c in penalización :
        for tol in tols :
            hyperparameters.append([kernel, c, tol])
            if hyperparameters[0] != "linear":
                for gamma in gammas:
                    hyperparameters.append([kernel, c, tol,gamma])

iris=datasets.load_wine()
X=iris.data
y=iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y, 
                                                    test_size=0.20)

def evaluate_set( X_train, X_test, y_train, y_test, parametros, p_id, lock):
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    import datetime
    print('\n Yo soy el proceso:', p_id, 'Comence a las:',datetime.datetime.now())
    for s in parametros:
        clf=SVC(kernel=(s[0]), C=float(s[1]), tol=float(s[2]))
        clf.fit(X_train, y_train)
        y_pred=clf.predict(X_test)
        lock.acquire()
        print(f'Proceso {p_id} con parametros {s} y accuracy {accuracy_score(y_test,y_pred)}')
        lock.release()

if __name__ == '__main__':   
    threads=[]
    N_THREADS=4
    splits=np.split(np.array(hyperparameters), N_THREADS)
    lock=multiprocess.Lock()
    for i in range(N_THREADS-1):
        # Se generan los hilos de procesamiento
        threads.append(multiprocess.Process(target=evaluate_set, args=[X_train,X_test,y_train,y_test, splits[i],i, lock]))

    start_time = time.perf_counter()
    # Se lanzan a ejecución
    for thread in threads:
        thread.start()
    # y se espera a que todos terminen
    for thread in threads:
        thread.join()
    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time-start_time} seconds")


    