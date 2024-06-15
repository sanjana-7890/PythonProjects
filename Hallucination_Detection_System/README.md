# Hallucination-Detection-Model-Hackathon-
This repository contains the Hallucination Detection Model in Contextual Question Answering that I made as a part of Hackathon.

# Hallucination Detection in Contextual Question Answering
Welcome to the Hallucination Detection model repository for Contextual Question Answering! This repository contains code and resources for building a sequential RNN model, specifically a subtype of LSTM, designed to detect hallucinations in contextual question answering scenarios.

# Dataset Overview
The dataset used for training and evaluation contains 400 samples, each comprising the following features:

• Context: The context or passage in which the question is asked.  

• Question: The question asked based on the given context.  

• Answer: The model's predicted answer for the given question.  

• Hallucination Score: A binary score indicating whether the model's answer is considered hallucinated or not, with values of either 0 or 1.

# Model Architecture
Before model architecture, proper NLP techniques are applied to create a vocabulary. The model architecture is based on a sequential RNN, specifically a subtype of Long Short-Term Memory (LSTM) network. LSTM networks are chosen for their ability to capture long-term dependencies in sequential data, which is essential for understanding contextual information in question answering tasks. Once created, on giving path of any csv file it will create a new column and store the predicted Hallucination score and return the final updated csv file.

# Evaluation Metrics
To assess the performance of the model, the following evaluation metrics are utilized:

• Accuracy (92.3%): Measures the overall correctness of the model's predictions.  

• AUC-ROC (Area Under the Receiver Operating Characteristic Curve): Evaluates the model's ability to discriminate between hallucinated and non-hallucinated answers across different thresholds.

• F1 Score (0.95): Provides a balance between precision and recall, particularly useful for imbalanced datasets.  

# Contributing
Contributions to the project are welcome! If you have any ideas for improvements or new features, feel free to open an issue or submit a pull request.

# Acknowledgments
Special thanks to the creators of the dataset used in this project, as well as the open-source community for providing valuable resources and inspiration.
