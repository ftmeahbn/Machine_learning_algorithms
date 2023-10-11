from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import time




def evaluate_model(X, y, model):
    time_start = time.time()
    model.fit(X, y)
    predict = model.predict(X)
    time_end = time.time()
    accuracy = model.score(X, y)
    execution_time = time_end - time_start
    return accuracy, execution_time



std_values = [1, 2, 4]
num_features_values = [10, 20, 30]
results = []




for std in std_values:
    for num_features in n_features_values:
        X, y = make_blobs(n_samples=1000, num_features=num_features, cluster_std=std, random_state=30)

        logistic_model = LogisticRegression()
        svm_model = SVC()
        knn_model = KNeighborsClassifier(n_neighbors=3)

        models = [logistic_model, svm_model, knn_model]

        model_results = {"std": std, "features": num_features, "results": []}

        for model in models:
            accuracy, execution_time = evaluate_model(X, y, model)
            model_results["results"].append(
                {"model": type(model).__name__, "accuracy": accuracy, "execution_time": execution_time})

        results.append(model_results)




results_final = ""




for std in std_values:
    for num_features in n_features_values:
        model_results = next(
            result for result in results if result["std"] == std and result["features"] == num_features)

        results_final += f"STD: {std}, N_FEATURES: {num_features}\n"

        for res in model_results["results"]:
            results_final += f"{res['model']}: Accuracy={res['accuracy'] * 100:.2f}%, Execution Time={res['execution_time']:.4f} seconds\n"

        results_final += "\n"

        

print(results_final)
