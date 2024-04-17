import multiprocess
import time
import numpy as np
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO

def load_data(data_flag):
    data_flag = data_flag
    # data_flag = 'breastmnist'
    download = True
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    # preprocessing
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    # load the data
    train_dataset = DataClass(split='train', transform=data_transform, download=download)
    test_dataset = DataClass(split='test', transform=data_transform, download=download)

    #  Los datos se cargan en la forma de un DataLoader de PyTorch
    # Se convierte los DataLoaders a listas de imágenes y etiquetas
    train_images = [image for image, label in train_dataset]
    train_labels = [label for image, label in train_dataset]
    test_images = [image for image, label in test_dataset]
    test_labels = [label for image, label in test_dataset]

    # Se convierte las listas a arrays de numpy 
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    # Se aplana las imágenes ya que los clasificadores no aceptan imágenes en 2D
    # entonces se convierten a vectores de 1D
    train_images = train_images.reshape((train_images.shape[0], -1))
    test_images = test_images.reshape((test_images.shape[0], -1))

    return train_images, test_images, train_labels, test_labels

def hyperparametros():
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    # penalización  = list(np.arange(0.1, 1, 0.1)) + list(range(1, 51)) + list(range(100, 501, 50))
    penalización  = list(np.arange(0.1, 0.3, 0.1)) + list(range(1, 51)) 

    # tols =  [1e-3, 1e-4, 1e-5]
    tols =  [1e-3, 1e-4]

    gammas = ['scale', 'auto']

    hyperparameters = []
    for kernel in kernels:
        for c in penalización :
            for tol in tols :
                for gamma in gammas:
                    hyperparameters.append([kernel, c, tol,gamma])
    return hyperparameters

def evaluate_set( X_train, X_test, y_train, y_test, parametros, p_id, lock):
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    import datetime
    print('\n Yo soy el proceso:', p_id, 'Comence a las:',datetime.datetime.now())
    for s in parametros:
        clf=SVC(kernel=(s[0]), C=float(s[1]), tol=float(s[2]), gamma=(s[3]))
        clf.fit(X_train, y_train.ravel())
        y_pred=clf.predict(X_test)
        lock.acquire()
        print(f'Proceso {p_id} con parametros {s} y accuracy {accuracy_score(y_test,y_pred)}')
        lock.release()

train_images, test_images, train_labels, test_labels = load_data('breastmnist')
hyperparameters = hyperparametros()
if __name__ == '__main__': 
    threads=[]
    N_THREADS=4
    splits=np.split(np.array(hyperparameters), N_THREADS)
    lock=multiprocess.Lock()
    for i in range(N_THREADS):
        # Se generan los hilos de procesamiento
        threads.append(multiprocess.Process(
                target=evaluate_set, 
                args=[train_images,test_images,train_labels,test_labels, splits[i],i, lock]))

    start_time = time.perf_counter()
    # Se lanzan a ejecución
    for thread in threads:
        thread.start()
    # y se espera a que todos terminen
    for thread in threads:
        thread.join()
    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time-start_time} seconds")