# EarthquakeDamageModels

![Header Image](EarthquakeDamaeModels.png)

This dataset is part of a practical exercise using the "Richter's Predictor: Modeling Earthquake Damage" competition on DrivenData. It's designed to test various tools and train artificial intelligence models to predict the level of damage to buildings caused by the 2015 Gorkha earthquake in Nepal. The data includes various structural and geo-location features of buildings, which are used to anticipate the severity of damage, aiding in disaster management and preparedness strategies. For more information, visit DrivenData's competition page.

https://www.drivendata.org/competitions/57/nepal-earthquake/data/

## Badges

![GitHub license](https://img.shields.io/github/license/JMartinArocha/DenguePredictionModels.svg)
![Python version](https://img.shields.io/badge/python-3.x-blue.svg)
![last-commit](https://img.shields.io/github/last-commit/JMartinArocha/DenguePredictionModels)
![issues](https://img.shields.io/github/issues/JMartinArocha/DenguePredictionModels)
![commit-activity](https://img.shields.io/github/commit-activity/m/JMartinArocha/DenguePredictionModels)
![repo-size](https://img.shields.io/github/repo-size/JMartinArocha/DenguePredictionModels)


### Prerequisites

Before running the scripts, it's essential to install the necessary Python packages. The project has a `requirements.txt` file listing all the dependencies. You can install these using `pip`. 

Note: The commands below are intended to be run in a Jupyter notebook environment, where the `!` prefix executes shell commands. If you're setting up the project in a different environment, you may omit the `!` and run the commands directly in your terminal.

```bash
!pip3 install --upgrade pip
!pip3 install -r requirements.txt
```

## Importing Shared Utilities

The project utilizes a shared Python utility script hosted on GitHub Gist. This script, `ml_utilities.py`, contains common functions and helpers used across various parts of the project. To ensure you have the latest version of this utility script, the project includes a step to download it directly from GitHub Gist before importing and using its functions.

Below is the procedure to fetch and save the `ml_utilities.py` script programmatically:


## Data Loading and Preprocessing

Here's how we programmatically load these datasets directly into our project using pandas, a powerful data manipulation library in Python. Additionally, we use custom utility functions to quickly inspect the loaded data.

## Data clean and normalization

This section of the code involves the preprocessing of categorical data within the dataset. We utilize the LabelEncoder from sklearn.preprocessing to convert each categorical column into a format that can be easily used by machine learning algorithms. This step is crucial as it translates categorical labels into a numeric format where each unique label in a column is assigned a corresponding integer. The included columns such as 'land_surface_condition', 'foundation_type', among others, are transformed to enhance the model's ability to learn from these features effectively.

Missing values are filled using the forward fill method to maintain data continuity. The dataset is then normalized using a custom utility function with MinMaxScaler, adjusting feature scales to enhance model performance. Furthermore, the dataset is merged with damage_grade labels for direct analysis. Data type conversions ensure consistent handling of features. To address potential class imbalance, the dataset is resampled to equalize the number of samples across different damage grades. This balancing act is visualized in a bar chart, showcasing the distribution of buildings across damage grades post-resampling.

## Graphical Methods and Feature Selection

This section of the code is dedicated to graphical methods for feature selection which are crucial for understanding the relationships and importance of different variables in the dataset. The heatmap visualizes correlations between variables, providing insights into which features are most related to each other. The SelectKBest method with ANOVA F-statistic is used to select the most significant features, helping to refine the model by focusing only on relevant inputs. Additionally, a dendrogram is generated to visualize the hierarchical clustering of features, which can help in understanding how features are grouped together based on similarity, aiding in further reduction or understanding of the feature space.

## Data spliting

This section of the code handles the data splitting process essential for model training and evaluation. First, a subset of features is selected for modeling. The dataset is then sampled down to 10% of its original size to manage computational resources more effectively. This sample is split into distinct training and testing datasets, with 20% of the data reserved for final evaluation to avoid overfitting. The training set is further divided, setting aside 25% as a validation set used for tuning model parameters. This step ensures that the model can be trained and validated effectively, leading to more reliable predictions.

## Automated Model Comparison with LazyPredict

In this section, the code demonstrates the use of LazyPredict to automate the process of fitting multiple models and comparing their performance. LazyPredict is a library that provides tools to quickly compare a wide range of machine learning models, making it easier to identify which models perform best for your specific dataset without the need for detailed configuration for each model initially. The code is structured to handle both classification and regression tasks, with output that includes the performance metrics of each model. This allows for a broad comparison across different types of algorithms in a concise manner.

## Automated Model comparison with PyCaret


PyCaret is an open-source, low-code machine learning library in Python that allows for building and comparing multiple machine learning models with just a few lines of code. It includes a wide range of algorithms for regression, classification, and clustering, and also supports deep learning models. PyCaret includes tools for data preprocessing, feature engineering, hyperparameter tuning, and model interpretation.

## Training - CrossValidation, RandomSearch y GridSearch

This section of the code illustrates advanced model optimization techniques using the AdaBoost algorithm. It employs RandomizedSearchCV to explore a wide range of parameter combinations quickly and GridSearchCV to refine the search within a more targeted parameter space. Both methods aim to find the optimal settings for the AdaBoost regressor. The final step involves evaluating the best model using cross-validation, which provides an unbiased estimation of the model's performance on unseen data. These methods collectively enhance the model's ability to generalize, ensuring robust predictions.

This section showcases how the model's performance metrics are calculated and visualized. Metrics such as the Mean Squared Error (MSE), R-squared (R²), and F1 Score are computed to evaluate the accuracy and reliability of the predictions. Additionally, graphical representations are provided:

Actual vs Predicted Values Plot: This scatter plot helps in visualizing the accuracy of predictions against actual values. The closer the points lie along the diagonal, the more accurate the predictions.

Residuals Plot: By plotting residuals against predicted values, this graph helps in identifying patterns in the prediction errors. Ideally, residuals should be randomly distributed around the horizontal line at zero, indicating that the model does not suffer from heteroscedasticity.