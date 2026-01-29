# AI-BeginnerProjects
A collection of end to end Machine Learning projects covering Regression, Classification, Unsupervised Learning, and Deep Learning.

This repository documents my progression through the AI domain, transitioning from classical statistical modeling to advanced Deep Learning architectures. Each project represents a specific challenge in data science, with a focus on Empirical Testing and Model Robustness.

*Project 1:* Agricultural Yield Predictor (Regression)
Core Challenge: Engineering a model to handle dirty real world environmental data.
Technical Focus: Data cleaning, handling missing values, and feature encoding.
Key Insight: Mastered the use of Random Forest Regressors to capture non-linear relationships between climate variables and crop output.
Impact: Demonstrates the ability to apply AI to sustainability and global food security challenges.

*Project 2:* Credit Card Fraud Detector (Classification)
Core Challenge: Solving the Imbalanced Class problem where 99% of data is normal.
Technical Focus: Precision-Recall trade-offs and cost-sensitive learning.

**While standard practice suggests class_weight='balanced', my testing showed that on this specific small minority sample, balancing amplified noise. I successfully optimized a 10-tree baseline that achieved a robust 80% recall, prioritizing model stability over complexity.**

*Project 3:* Movie Recommender System (Unsupervised Learning)
Core Challenge: Converting unstructured text (plots and genres) into mathematical vectors.
Technical Focus: Natural Language Processing (NLP), CountVectorizer, and Cosine Similarity.
Key Insight: Implemented feature engineering by merging overviews and metadata, then utilized the angle between 5,000 dimensional vectors to find similarities.
Impact: Shows an understanding of how modern platforms (Netflix/Spotify) utilize item similarity to personalize user experiences.

*Project 4:* Handwritten Digit Recognizer (Deep Learning)
Core Challenge: Building an artificial brain to recognize visual patterns.
Technical Focus: Neural Network architecture using TensorFlow(or)Keras.
Key Insight: Designed a Multi-Layer Perceptron featuring a ReLU hidden layer for non-linearity and a Softmax output layer for multi-class probability.
Optimization: Applied Normalization and Dropout (20%) to ensure the model generalizes to new handwriting styles rather than just memorizing the training set.
