import multiprocessing 
import numpy as np
import time

def sum_parcial(arr, lock, results):
    suma = 0
    for n in arr:
        denominador = 2*n + 1
        suma += 1/denominador
    results.append(suma)
    
    lock.acquire()
    print(f"Resultado total del set : -> {suma}")
    lock.release()
    
N_THREADS=5
splits = np.array_split(np.arange(1000), N_THREADS)

if __name__ == '__main__':   
    lock = multiprocessing.Lock()
    threads = []
    results = multiprocessing.Manager().list()

    for i in range(N_THREADS):
        thread = multiprocessing.Process(target=sum_parcial, args=(splits[i], lock, results))
        threads.append(thread)

    start_time = time.perf_counter()

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    finish_time = time.perf_counter()

    print(f"Program finished in {finish_time-start_time} seconds")
    print(f"Suma total: {sum(results)}")