import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

data = pd.read_csv("data.csv")




def clean_and_split_data(x, y):

    from sklearn.model_selection import train_test_split

    encoder = OneHotEncoder(sparse=False, sparse_output=False)

    label_encoder = LabelEncoder()
    x["Class"] = label_encoder.fit_transform(x["Class"])
    x["Type of Waiting List"] = label_encoder.fit_transform(x["Type of Waiting List"])

    encoded_cols = encoder.fit_transform(x[["Class", "Type of Waiting List"]])
    encoded_cols_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(["Class", "Type of Waiting List"]))
    x_encoded = pd.concat([x.drop(["Class", "Type of Waiting List"], axis=1), encoded_cols_df], axis=1)
    print(x_encoded.head())

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12)

    return x_train, x_test, y_train, y_test


def build_gradientboost_model(x_train, y_train):
    # build the model
    from sklearn.ensemble import GradientBoostingClassifier
    classifier = GradientBoostingClassifier()
    classifier = classifier.fit(x_train, y_train)
    return classifier


def build_adaboost_model(x_train, y_train):
    # build the model
    from sklearn.ensemble import AdaBoostClassifier
    classifier = AdaBoostClassifier()
    classifier = classifier.fit(x_train, y_train)
    return classifier

def build_randomforest_model(x_train, y_train):
    # build the model
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier()
    classifier = classifier.fit(x_train, y_train)
    return classifier

def build_xgboost_model(x_train, y_train):
    # build the model
    from xgboost import XGBClassifier
    classifier = XGBClassifier()
    classifier = classifier.fit(x_train, y_train)
    return classifier


def build_decisiontree_model(x_train, y_train):
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier()
    classifier = classifier.fit(x_train, y_train)
    return  classifier


def build_naive_bayes_model(x_train, y_train):
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier = classifier.fit(x_train, y_train)
    return  classifier



def build_svc_model(x_train, y_train):
    from sklearn.svm import SVC
    classifier = SVC()
    classifier = classifier.fit(x_train, y_train)
    return  classifier

from tabulate import tabulate

def predict_probability(classifier, new_data):
    if hasattr(classifier, "decision_function"):
        decision_values = classifier.decision_function(new_data)  # Get the decision values
        probabilities = 1 / (1 + np.exp(-decision_values))  # Convert decision values to probabilities using sigmoid function
    elif hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(new_data)  # Get the class probabilities
        probabilities = probabilities[:, 1]  # Use the probabilities of the positive class
    else:
        raise AttributeError("Classifier does not have decision_function() or predict_proba() method")

    return probabilities


def cross_validation(algorithm, classifier, x_test, y_test):
    # evaluate the model
    predictions = classifier.predict(x_test)
    if hasattr(classifier, "decision_function"):
        decision_values = classifier.decision_function(x_test)  # Get the decision values
    elif hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(x_test)  # Get the class probabilities
        decision_values = probabilities[:, 1]  # Use the probabilities of the positive class
    else:
        raise AttributeError("Classifier does not have decision_function() or predict_proba() method")
    

    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.metrics import recall_score, precision_score, f1_score

    cm = confusion_matrix(y_test, predictions)
    accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
    recall = recall_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    error_rate = 1 - accuracy

    metrics_table = [["Metric", "Value"],
                     ["Accuracy", accuracy],
                     ["Recall", recall],
                     ["Precision", precision],
                     ["F1 Score", f1],
                     ["Error Rate", error_rate]]

    print(f"Evaluation Metrics for {algorithm}:")
    print(tabulate(metrics_table, headers="firstrow", tablefmt="fancy_grid"))

    print("\nConfusion Matrix:")
    print(tabulate(cm, headers=["Predicted Negative", "Predicted Positive"], showindex=["Actual Negative", "Actual Positive"], tablefmt="fancy_grid"))

    return accuracy, recall, precision, f1, error_rate, decision_values








def probability_generate(algorithm,x_validate):
    # print(x_validate)
    # print(len(x_validate[0]))
    # print(algo)

    y=data.iloc[:,17:18]
    
    x= data.iloc[:,5:5+len(x_validate[0])]

    print(x.head())


    from sklearn.utils import column_or_1d  # Import column_or_1d function
    y = column_or_1d(y, warn=False)

    x_train, x_test, y_train, y_test = clean_and_split_data(x, y)
    
    if algorithm == 0:
        classifier_gb = build_gradientboost_model(x_train, y_train)
        accuracy_gb = cross_validation('Gradient Boost', classifier_gb, x_test, y_test)
        probability_gb = predict_probability(classifier_gb, x_validate)
        return probability_gb,accuracy_gb[0]

    elif algorithm == 1:
        classifier_ada = build_adaboost_model(x_train, y_train)
        accuracy_ada = cross_validation('Ada Boost', classifier_ada, x_test, y_test)
        probability_ada=predict_probability(classifier_ada,x_validate)
        return probability_ada,accuracy_ada[0]
    
    elif algorithm == 2:
        classifier_dt = build_decisiontree_model(x_train, y_train)
        accuracy_dt = cross_validation('Decision Tree', classifier_dt, x_test, y_test)
        probability_dt=predict_probability(classifier_dt,x_validate)
        return probability_dt,accuracy_dt[0]
    
    elif algorithm == 3:
        classifier_xg = build_xgboost_model(x_train, y_train)
        accuracy_xg = cross_validation('XGBoost', classifier_xg, x_test, y_test)
        probability_xg=predict_probability(classifier_xg,x_validate)
        return probability_xg,accuracy_xg[0]
    
    elif algorithm == 4:
        classifier_rf = build_randomforest_model(x_train, y_train)
        accuracy_rf = cross_validation('Random Forest', classifier_rf, x_test, y_test) 
        probability_ra = predict_probability(classifier_rf, x_validate)
        return probability_ra,accuracy_rf[0]

    elif algorithm == 5:
        classifier_svc = build_svc_model(x_train, y_train)
        accuracy_svc = cross_validation('SVM', classifier_svc, x_test, y_test)
        probability_svc = predict_probability(classifier_svc,x_validate)
        return probability_svc,accuracy_svc[0]

    elif algorithm == 6:
        classifier_nb = build_naive_bayes_model(x_train, y_train)
        accuracy_nb = cross_validation('Naive Bayes', classifier_nb, x_test, y_test)
        probability_nb= predict_probability(classifier_nb,x_validate)
        return probability_nb,accuracy_nb[0]

    
