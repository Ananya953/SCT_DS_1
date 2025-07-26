import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load dataset
DATADIR = "PetImages"
CATEGORIES = ["Cat", "Dog"]
IMG_SIZE = 64

data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DATADIR, category + "\\train")
    class_num = CATEGORIES.index(category)

    for img in os.listdir(path): 
        try:
            img_array = cv2.imread(os.path.join(path, img))
            if img_array is None:
                continue
            resized_img = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            data.append(resized_img.flatten())
            labels.append(class_num)
        except Exception as e:
            print("Skipped:", img, "Reason:", e)
            continue
            
# Convert to NumPy arrays
X = np.array(data) 
y = np.array(labels)
print("Data length:", len(data))
print("Labels length:", len(labels))

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Predictions
y_pred = svm.predict(X_test)

# Accuracy and report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CATEGORIES, yticklabels=CATEGORIES)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("output.png")  # Save the graph
plt.show()