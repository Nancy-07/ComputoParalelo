# Import the necessary dependencies
import multiprocessing
import time
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# We create a list with the parameters to be evaluated
hyperparameters = []
for criterion in ["gini", "entropy"]:
    for trees in range(10, 100):
        hyperparameters.append([trees, criterion])


def evaluate_set(hyperparameter_set, results, lock):
    """
    Evaluate a set of hyperparameters
    Args:
    hyperparameter_set: a list with the set of hyperparameters to be evaluated
    results: a shared list to store the evaluation results
    """
    for s in hyperparameter_set:
        clf = RandomForestClassifier(n_estimators=int(s[0]), criterion=s[1])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        with lock:
            results.append(accuracy_score(y_test, y_pred))


def main():
    # Load the dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    # Split the dataset into 80-20 for training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.20
    )

    # Initialize shared data structures
    manager = multiprocessing.Manager()
    results = manager.list()

    # Now we will evaluate with multiple processes
    processes = []
    N_PROCESSES = 4
    splits = np.array_split(hyperparameters, N_PROCESSES)
    lock = multiprocessing.Lock()

    start_time = time.perf_counter()

    for i in range(N_PROCESSES):
        # Generate the processing threads
        p = multiprocessing.Process(
            target=evaluate_set, args=(splits[i], results, lock)
        )
        processes.append(p)
        p.start()

    # Wait for all processes to finish
    for p in processes:
        p.join()

    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time-start_time} seconds")

    # Process the results
    print("Results:")
    for i, acc in enumerate(results):
        print(f"Accuracy for process {i}: {acc}")


if __name__ == "__main__":
    main()
