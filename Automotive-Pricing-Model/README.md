# Problem Statement
A Chinese automobile company, Geely Auto, aspires to enter the US market by setting up their manufacturing unit there and producing cars locally to give competition to their US and European counterparts. 

 
They have contracted an automobile consulting company to understand the factors on which the pricing of cars depends. Specifically, they want to understand the factors affecting car pricing in the American market, as they may differ from the Chinese market. 

The company wants to know the following things:

   • Which variables are significant in predicting the price of a car?
   • How well do those variables describe the price of a car?  

Based on various market surveys, the consulting firm has gathered a large data set of different types of cars across the American market. 

 

# Business Goals

You are required to model the price of cars with the available independent variables. The management will use be using this model to understand exactly how the prices vary with the independent variables. Accordingly, they can change the design of the cars, the business strategy, etc., to meet certain price levels. Further, the model will allow the management to understand the pricing dynamics of a new market.

 

# Data Preparation

There is a variable named CarName that comprises two parts: the first word is the name of the car company, and the second is the car model. For example, Chevrolet Impala has ‘Chevrolet’ as the car company name and ‘Impala’ as the car model name. You need to consider only the company name as the independent variable for model building.

 

# Model Evaluation

When you are done with model building and residual analysis and have made predictions on the test set, ensure that you use the following two lines of code to calculate the R-squared score on the test set:

from sklearn.metrics import r2_score <br>
r2_score(y_test, y_pred)

Where y_test is the test dataset for the target variable, and y_pred is the variable containing the predicted values of the target variable on the test set.
