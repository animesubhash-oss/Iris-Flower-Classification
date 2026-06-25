import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#Load Dataset
iris= load_iris()

X=iris.data
Y=iris.target

#Split data
X_train,X_test,Y_train,Y_test=train_test_split(
    X,Y,
    test_size=0.2,
    random_state=42
)

#Create Model
model=KNeighborsClassifier(n_neighbors=3)

#Train
model.fit(X_train,Y_train)

#Predict
Y_pred=model.predict(X_test)

#Accuracy
accuracy=accuracy_score(Y_test,Y_pred)
print("Accuracy:", accuracy)


#Test flower 
sample=[[5.1,3.5,1.4,0.2]]
prediction=model.predict(sample)
print("Predicted Species:", iris.target_names[prediction[0]])


#visulaization
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"]=iris.target
for species in range(3):
    subset = df[df["species"] == species]
    plt.scatter(
        subset["sepal length (cm)"],
        subset["sepal width (cm)"],
        label=iris.target_names[species]
    )

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Iris Dataset")
plt.legend()
plt.show()



