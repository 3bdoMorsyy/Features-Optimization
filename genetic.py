import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


def get_best_features(path, col_name, pop_size=100, num_gen=20):
    scaler = StandardScaler()
    data = pd.read_csv(path)
    data = data.drop(columns="Index")
    data.dropna(inplace=True)
    cols = data.columns
    label_encoder = LabelEncoder()
    for col in cols:
        data[col] = label_encoder.fit_transform(data[col])

    X = data.drop(columns=col_name).values
    X = scaler.fit_transform(X)
    y = data[col_name].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=21
    )

    def fitness_function(selected_features):
        selected_indices = []

        for i, bit in enumerate(selected_features):
            if bit == 1:
                selected_indices.append(i)
        if len(selected_indices) == 0:
            return float("inf")

        X_train_selected = X_train[:, selected_indices]
        X_test_selected = X_test[:, selected_indices]
        # PCA
        pca_clf = PCA(n_components=0.95, random_state=42)
        x_train_pca = pca_clf.fit_transform(X_train_selected)
        x_test_pca = pca_clf.transform(X_test_selected)
        # MLP with pca
        mlp_clf_pca = MLPClassifier(
            solver="adam", max_iter=500, activation="relu", random_state=42
        )
        mlp_clf_pca.fit(x_train_pca, y_train)
        y_pred_mlp_pca = mlp_clf_pca.predict(x_test_pca)
        accuracy = accuracy_score(y_test, y_pred_mlp_pca)
        return accuracy

    def genetic_algorithm(population_size, num_generations, num_features):
        population = np.random.randint(0, 2, size=(population_size, num_features))

        for generation in range(num_generations):
            fitness_scores = []

            for individual in population:
                fitness_scores.append(fitness_function(individual))
            elite_indices = np.argsort(fitness_scores)[
                ::-1
            ]  # Reverse Order (High to low)
            print(fitness_scores)
            print(elite_indices)
            elite_indices = elite_indices[
                : int(population_size * 0.3)
            ]  # we choose Top 30% for cross over and keep them for next generation

            elite_population = []
            for i in elite_indices:
                elite_population.append(population[i])

            offspring = []
            while len(offspring) < population_size - len(elite_population):
                parent1, parent2 = random.choices(elite_population, k=2)
                crossover_point = random.randint(1, num_features - 1)
                child = np.concatenate(
                    (parent1[:crossover_point], parent2[crossover_point:])
                )
                # mutation
                if random.randint(1, 10) >= 5:  # Make Mutaion ratio = 50%
                    mutation_bit = random.randint(0, num_features - 1)
                    child[mutation_bit] = (
                        1 - child[mutation_bit]
                    )  # Inverse bit (if bit = 0 -> 1 and vice versa)
                offspring.append(child)

            # new population
            population = elite_population + offspring
            best_chromosome = population[np.argmax(fitness_scores)]
            print(
                f"Generation {generation + 1}: Best Individual = {best_chromosome}, Best Accuracy = {max(fitness_scores)}"
            )

        # best individual in of each generation
        return population[np.argmax(fitness_scores)]

    num_features = X_train.shape[1]  # number of features
    best_features = genetic_algorithm(
        population_size=pop_size, num_generations=num_gen, num_features=num_features
    )
    print(f"Best features found:{best_features}")

    selected_feature_indices = []
    for i in range(len(best_features)):
        if best_features[i] == 1:
            selected_feature_indices.append(i)
    selected_feature_names = data.drop(columns=col_name).columns[
        selected_feature_indices
    ]
    print(data.iloc[:, 2:])
    return selected_feature_names.values
