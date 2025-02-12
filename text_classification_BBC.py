#                        loading data from different folders
import os

def load_data(base_path):
    texts = []
    labels = []
    categories = ['athletics', 'cricket', 'football', 'rugby', 'tennis']
    
    for category in categories:
        category_path = os.path.join(base_path, category)
        
        if os.path.isdir(category_path):  
            for file_name in os.listdir(category_path):
                file_path = os.path.join(category_path, file_name)
                
                if file_name.endswith('.txt'):  
                    with open(file_path, 'r', encoding='utf-8') as file:
                        texts.append(file.read())
                        labels.append(category)  
        else:
            print(f"Category folder not found: {category}")

    return texts, labels
#           change this path to the folder that contains the data in your computer 
base_path = r"d:\University\TERM7 14031\DataScience_Algs\bbcsport-fulltext\bbcsport"
texts, labels = load_data(base_path)


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()


term_doc_matrix = vectorizer.fit_transform(texts)


############################ ***********************************************############################ ***********************************************

from sklearn.decomposition import TruncatedSVD

# d = 5
# svd = TruncatedSVD(n_components=d, random_state=42)
# reduced_matrix = svd.fit_transform(term_doc_matrix)

############################ ***********************************************############################ ***********************************************


from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(reduced_matrix, labels, test_size=0.3, random_state=42)


############################ ***********************************************############################ ***********************************************

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# knn = KNeighborsClassifier(metric='cosine')

# knn.fit(X_train, y_train)

# y_pred = knn.predict(X_test)

# # Calculaing accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)


############################ ***********************************************############################ ***********************************************
#                            Accuracy with dimensionality reduction for D=2,3,5,10,20,50,100
results = {}
for d in [2, 3, 5, 10, 20, 50, 100]:
    svd = TruncatedSVD(n_components=d, random_state=42)
    reduced_matrix = svd.fit_transform(term_doc_matrix)
    X_train, X_test, y_train, y_test = train_test_split(reduced_matrix, labels, test_size=0.3)
    knn = KNeighborsClassifier(metric='cosine')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[d] = accuracy
    print(f"d={d}, Accuracy={accuracy}")


############################ ***********************************************############################ ***********************************************

#                               Accuracy without dimensionality reduction


X_train, X_test, y_train, y_test = train_test_split(term_doc_matrix, labels, test_size=0.3, random_state=42)
knn = KNeighborsClassifier(metric='cosine')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy without dimensionality reduction:", accuracy)
