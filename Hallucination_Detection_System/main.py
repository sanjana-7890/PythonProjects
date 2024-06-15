import subprocess

# Function to install dependencies from requirements.txt
def install_dependencies():
    subprocess.call(['pip', 'install', '-r', 'requirements.txt'])

# Call the function to install dependencies
install_dependencies()

# URL DEFAULT : https://drive.google.com/file/d/1igd3Zx9Vy8xgQElRxgqI3kvDd7ephrql/view

import os
import logging
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout

import pandas as pd

current_directory = os.getcwd()

# Specify the absolute path for the log file
log_file_path = os.path.join(current_directory, "logfile.log")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', filename=log_file_path)

try:

    url= input("Please Enter the URL of csv file : ")
    file_id=url.split('/')[-2]
    dwn_url='https://drive.google.com/uc?id=' + file_id
    df = pd.read_csv(dwn_url)

    logging.info("##########File received successfully##########")
    print("##########File received successfully##########")

    print(df.head())

    df = df.dropna(subset=['Question'])
# df.info()

# prompt: delete the complete row if all the 3 columns context,question,answers are same after preprocessing

    df = df.drop_duplicates(subset=['Context', 'Question', 'Answer'], keep='first')

    texts = df['Context'] + ' ' + df['Question'] + ' ' + df['Answer']
    labels = df['Hallucination']

except:

    print("Error : Please Enter correct URL")

    raise SystemExit

# prompt: delete the row having null value inquestion column

logging.info("##########Model Training Started##########")
print("##########Model Training Started##########")

# Encode the labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Tokenization
max_words = 10000  # assuming you want to use the top 10,000 words
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Padding sequences
max_sequence_length = max(len(sequence) for sequence in sequences)
data = pad_sequences(sequences, maxlen=max_sequence_length)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)


# Define the model architecture
embedding_dim = 100  # dimension of word embeddings
hidden_units = 64  # number of units in LSTM layer
vocab_size = min(max_words, len(tokenizer.word_index) + 1)

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_sequence_length))
model.add(LSTM(hidden_units))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
batch_size = 32
epochs = 15
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))

logging.info("##########Model Training Completed##########")
print("##########Model Training Started##########")

logging.info("##########Model Evaluation Started##########")
print("##########Model Training Started##########")

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

print(y_train)

# Predict hallucination values for all rows in the dataset
predictions_complete = model.predict(data)

# Decode predictions if necessary
predictions_decoded_complete = label_encoder.inverse_transform(predictions_complete.flatten().round().astype(int))

# Calculate accuracy
accuracy_complete = np.mean(predictions_decoded_complete == labels)

print("Accuracy on complete dataset:", accuracy_complete)

from sklearn.metrics import roc_auc_score, f1_score

# Calculate AUC-ROC
auc_roc_complete = roc_auc_score(labels, predictions_complete)

# Calculate F1 score
f1_score_complete = f1_score(labels, predictions_decoded_complete, average='binary')

print("AUC-ROC on complete dataset:", auc_roc_complete)
print("F1 score on complete dataset:", f1_score_complete)


from sklearn.metrics import roc_auc_score, f1_score

# Make predictions on the test set
predictions_test = model.predict(X_test)

# Decode predictions if necessary
predictions_decoded_test = label_encoder.inverse_transform(predictions_test.flatten().round().astype(int))

# Calculate AUC-ROC for the test set
auc_roc_test = roc_auc_score(y_test, predictions_test)

# Calculate F1 score for the test set
f1_score_test = f1_score(y_test, predictions_decoded_test, average='binary')

print("AUC-ROC on test set:", auc_roc_test)
print("F1 score on test set:", f1_score_test)

logging.info("##########Model Evaluation Started##########")
print("##########Model Evaluation Started##########")

# Predict hallucination values for the complete dataset
predictions_complete = model.predict(data)

# Decode predictions if necessary
predictions_decoded_complete = label_encoder.inverse_transform(predictions_complete.flatten().round().astype(int))

# Store predicted values in the complete dataset dataframe
df['Prediction'] = predictions_decoded_complete

# Save the modified complete dataset dataframe to a new CSV file
df.to_csv("complete_with_predictions.csv", index=False)

# Read the CSV file into a DataFrame
df_new = pd.read_csv("complete_with_predictions.csv")
df_new.shape
df_new

print(df_new)

# prompt: find number of rows having different values correspondingly in hallucination and prediction column

df_new['Match'] = df_new['Prediction'] == df_new['Hallucination']
df_new['Match'].value_counts()


#Download the model
model.save("/hallucination_model.h5")

from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# Compute precision-recall curve
precision, recall, _ = precision_recall_curve(y_test, predictions_test)

# Compute average precision score
average_precision = average_precision_score(y_test, predictions_test)

# Plot precision-recall curve
plt.figure()
plt.plot(recall, precision, color='darkorange', lw=2, label='Precision-Recall curve (AP = %0.2f)' % average_precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower right")
plt.show()

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Compute ROC curve and ROC area for each class
fpr, tpr, _ = roc_curve(y_test, predictions_test)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()