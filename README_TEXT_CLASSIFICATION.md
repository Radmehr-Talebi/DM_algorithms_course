BBC News Text Classification

Overview

This project performs text classification on the BBC News dataset using Singular Value Decomposition (SVD) and Latent Semantic Indexing (LSI). The goal is to convert text into lower-dimensional vectors and compare classification accuracy before and after dimensionality reduction.

Dataset

The dataset used is the BBC News dataset, which consists of approximately 2200 news articles categorized into five classes:

Athletics

Rugby

Football

Tennis

Cricket

Download the dataset from:

BBC News Dataset

Dataset Direct Download

Methodology

Text Preprocessing & Feature Extraction:

Convert text data into a term-document matrix using CountVectorizer from sklearn.feature_extraction.text.

Dimensionality Reduction using SVD:

Apply TruncatedSVD from sklearn.decomposition to reduce dimensions.

Perform experiments with different values of dimensions: d = [2, 3, 5, 10, 20, 50, 100].

Text Classification:

Use the cosine similarity metric and K-Nearest Neighbors (KNN) to classify the texts.

Split data into 70% training and 30% testing.

Measure classification accuracy for each reduced dimension.

Comparison:

Perform classification using the original (non-reduced) feature vectors.

Compare the accuracy of classification before and after dimensionality reduction.

Dependencies

Make sure you have the following Python libraries installed:

pip install numpy pandas scikit-learn

Running the Code

Download and extract the dataset.

Run the Python script to preprocess the data, reduce dimensionality, classify, and evaluate results.

Results

The classification accuracy is recorded for different values of d.

A comparison is made between the original high-dimensional data and the reduced feature space.

The trade-off between dimensionality reduction and classification performance is analyzed.

References

BBC News Dataset

Scikit-learn Documentation
