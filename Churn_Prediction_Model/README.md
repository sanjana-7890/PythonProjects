# CustomerChurnPrediction_SpeakX

Dataset - <a href="https://www.kaggle.com/datasets/blastchar/telco-customer-churn"> Link to Dataset</a><br><br>
Customer churn or the loss of clients or subscribers is a critical issue for businesses across various sectors especially for service-oriented industries like telecommunications. Understanding and predicting customer churn can provide valuable insights for retaining customers and enhancing overall business performance.<BR>

I conducted a comprehensive analysis and implementation of machine learning models for predicting customer churn in the telecom industry.The primary objective was to evaluate various machine learning algorithms to accurately predict churn and identify key factors influencing customer retention.The models examined included traditional algorithms like Logistic Regression, Gradient Boosting, Random Forest and Random Forest post hyperparameter tuning. At last,all the four models were evaluated on the different classification metrics as the problem statement involves the binary classification task.<br>

## Four machine learning models were chosen for churn prediction:<br>
1.Logistic Regression:<br>
o Explanation: Logistic Regression is a widely used statistical model that is simple, interpretable, and effective for binary classification problems. It models the probability of the default class (churn) using a logistic function.<br>
o Performance: Without hyperparameter tuning, the logistic regression model achieved an accuracy of 81.69%. The precision was 67.69%, recall was 58.98%, and the F1 score was 63.04%.<br><br>
2.Gradient Boosting:<br>
o Explanation: Gradient Boosting is an ensemble technique that builds models sequentially, each new model correcting the errors made by the previous ones. It is known for its high performance with imbalanced datasets.<br>
o Performance: This model achieved an accuracy of 80.70%. The precision was 66.45%, recall was 54.69%, and the F1 score was 60.00%.<br><br>
3. Random Forest:<br>
o Explanation: Random Forest is another ensemble technique that builds multiple decision trees and merges them together to get a more accurate and stable prediction. It is robust and effective for large datasets.<br>
o Performance: Initially, the Random Forest model achieved an accuracy of 78.99%. The precision was 64.00%, recall was 47.18%, and the F1 score was 54.32%.<br><br>
4. Random Forest with Hyperparameter Tuning:<br>
o Explanation: After hyperparameter tuning using GridSearchCV, which exhaustively searches over a specified parameter grid to find the optimal model parameters, the Random Forest model's performance improved significantly.<br>
o Performance: The best model achieved an accuracy of 80.98%. The precision increased to 68.04%, recall to 53.08%, and the F1 score to 59.64%.<br>
