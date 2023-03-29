import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = load_breast_cancer()

X = data.data
y = data.target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


class GreyWolfOptimizer:
    def __init__(self, X_train, y_train, population_size=10, iterations=50):
        self.X_train = X_train
        self.y_train = y_train
        self.population_size = population_size
        self.iterations = iterations
        
        self.dim = X_train.shape[1]
        self.upper_bound = np.ones(self.dim)
        self.lower_bound = np.zeros(self.dim)
        
    def fitness(self, individual):
        # Evaluate fitness using accuracy score
        selected_features = np.where(individual > 0.5)[0]
        if len(selected_features) == 0:
            return 0
        X = self.X_train[:, selected_features]
        y = self.y_train
        model = RandomForestClassifier()
        model.fit(X, y)
        y_pred = model.predict(X)
        return accuracy_score(y, y_pred)
    
    def optimize(self):
        # Initialize population
        population = np.random.uniform(low=self.lower_bound, high=self.upper_bound, size=(self.population_size, self.dim))
        
        # Main loop
        for i in range(self.iterations):
            # Sort population by fitness
            fitness_values = np.array([self.fitness(individual) for individual in population])
            sorted_indices = np.argsort(fitness_values)[::-1]
            population = population[sorted_indices]
            
            # Update alpha, beta, and delta values
            alpha = population[0]
            beta = population[1]
            delta = population[2]
            
            # Update positions of other individuals
            a = 2 - 2 * i / (self.iterations - 1)  # parameter for spiral update
            for j in range(3, self.population_size):
                r1 = np.random.random(size=self.dim)
                r2 = np.random.random(size=self.dim)
                A1 = a * (2 * r1 - 1)
                C1 = 2 * r2
                D_alpha = np.abs(C1 * alpha - population[j])
                X1 = alpha - A1 * D_alpha
                
                r1 = np.random.random(size=self.dim)
                r2 = np.random.random(size=self.dim)
                A2 = a * (2 * r1 - 1)
                C2 = 2 * r2
                D_beta = np.abs(C2 * beta - population[j])
                X2 = beta - A2 * D_beta
                
                r1 = np.random.random(size=self.dim)
                r2 = np.random.random(size=self.dim)
                A3 = a * (2 * r1 - 1)
                C3 = 2 * r2
                D_delta = np.abs(C3 * delta - population[j])
                X3 = delta - A3 * D_delta
                
                new_individual = (X1 + X2 + X3) / 3
                new_individual = np.clip(new_individual, self.lower_bound, self.upper_bound)
                
                # Update individual in population
                if self.fitness(new_individual) > fitness_values[j]:
                    population[j] = new_individual
            
            self.population = population


# Create instance of GreyWolfOptimizer class
gwo = GreyWolfOptimizer(X_train, y_train)

# Run optimization algorithm
gwo.optimize()

# Get selected features
selected_features = np.where(gwo.population[0] > 0.5)[0]
print("Selected features:", selected_features)

# Evaluate accuracy on test set using selected features
X_train_selected = X_train[:, selected_features]
X_test_selected = X_test[:, selected_features]

model = RandomForestClassifier()
model.fit(X_train_selected, y_train)
y_pred = model.predict(X_test_selected)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)