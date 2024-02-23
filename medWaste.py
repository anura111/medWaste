import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

# Read in the csv file with hospital waste stats as a dataframe
hospitalWasteData = pd.read_csv("Hospital Waste 5.csv")

# Assign first 5 columns (0 to 6 not including 6) as X - independent variables/input
# Columns: staffedBeds, patientDays, violations, totalWaste, properlyDisposedWaste, recycledWaste (these were values we researched as common statistics that hospitals will report 
X = hospitalWasteData.iloc[:, :6]
# Assign the final column (6) as y - dependent variable/output
# Columns: class
y = hospitalWasteData.iloc[:, 6]

# Scale the data, ie: 95000 tons of waste and 10 violations hold the same weight/influence
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# We want to split the data into 60% for training, 20% for validation and 25% for testing
# First we split test data (X_test, y_test) as 0.2 of data
# Then we split the remaining 80% of data into validation(X_trainB, y_trainB) which is 0.25 of 0.8, and what's left becomes training data (X_trainA, y_trainB)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.20,
                                                    random_state=5,
                                                    stratify=y)
X_trainA, X_trainB, y_trainA, y_trainB = train_test_split(X_train, y_train,
                                                          test_size=0.25,
                                                          random_state=5)


# Initialize variables to develop a KNN model using 1 to 30 neighbors and arrays to record each k's accuracy to
# later choose the k with the best accuracy

neighbours = range(1, 31)
trainA_accuracy = np.empty(30)
trainB_accuracy = np.empty(30)

# Model KNN for all amounts of neighbors from 1 to 30 neighbors recording accuracy with each loop
for k in neighbors:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_trainA, y_trainA)
    trainA_accuracy[k - 1] = model.score(X_trainA, y_trainA)
    trainB_accuracy[k - 1] = model.score(X_trainB, y_trainB)

# Plot a graph of training A and training B accuracy and determine the best value of k.
# We choose the highest accuracy for both lines after the initial fall-off
plt.plot(neighbours, trainA_accuracy, label="Training A Accuracy")
plt.plot(neighbours, trainB_accuracy, label="Training B Accuracy")
plt.legend()
plt.show()

# From the graph we find that having 5 neighbors results in the most accurate model
bestK = 5

# Model KNN using bestK and display the confusion matrix (as a figure). (4)
best_knn = KNeighborsClassifier(n_neighbors=bestK)
best_knn.fit(X_trainA, y_trainA)

# Use the model to predict the class of all the test data
y_pred = best_knn.predict(X_test)

# Initialize and print out the confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cnf_matrix, display_labels=best_knn.classes_).plot()
plt.show()

# Print accuracy of confusion matrix
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
