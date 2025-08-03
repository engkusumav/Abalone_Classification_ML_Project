import numpy as np   
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, roc_curve, accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import precision_recall_curve,confusion_matrix, ConfusionMatrixDisplay

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

#Target_column
Target_column = 'Rings'
col = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 
           'Viscera_weight', 'Shell_weight', 'Rings']

# Import Data
def import_data(filename):
    df = pd.read_csv(filename, header=None, names=col)
    return df

# # Convert categorical data to numerical
def data_conversion(data):
    data['Sex'].replace({'M': 0, 'F': 1, 'I': 2}, inplace=True)


def log_transform(data):
    columns_for_log_transform = ['Shell_weight']
    for col in columns_for_log_transform:
        #using log1p in case the values is 0
        data[col] = np.log1p(data[col]) 
    return data


# Correlation Heatmap
def correlation_heatmap(data):
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap of Abalone Dataset')
    plt.savefig("Correlation_Map.png", bbox_inches='tight')
    plt.close()

# Scatter Plots
def scatter_plots(data):
    plt.figure(figsize=(7, 6))
    plt.scatter(data['Diameter'], data['Rings'], alpha=0.6, color='skyblue')
    plt.title('Diameter vs Rings')
    plt.xlabel('Diameter')
    plt.ylabel('Rings')
    plt.savefig('Scatter_Diameter_vs_Rings.png', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.scatter(data['Shell_weight'], data['Rings'], alpha=0.6, color='pink')
    plt.title('Shell Weight vs Rings')
    plt.xlabel('Shell Weight')
    plt.ylabel('Rings')
    plt.savefig('Scatter_ShellWeight_vs_Rings.png', bbox_inches='tight')
    plt.close()

# Histograms
def histograms(data):
    plt.figure(figsize=(7, 6))
    plt.hist(data['Rings'], bins=20, color='darksalmon', alpha=0.85, edgecolor='black')
    plt.title('Distribution of Rings')
    plt.xlabel('Rings')
    plt.ylabel('Frequency')
    plt.savefig('Histogram_Rings.png', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.hist(data['Diameter'], bins=20, color='deepskyblue', alpha=0.85, edgecolor='black')
    plt.title('Distribution of Diameter')
    plt.xlabel('Diameter')
    plt.ylabel('Frequency')
    plt.savefig('Histogram_Diameter.png', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.hist(data['Shell_weight'], bins=20, color='hotpink', alpha=0.85, edgecolor='black')
    plt.title('Distribution of Shell Weight')
    plt.xlabel('Shell Weight')
    plt.ylabel('Frequency')
    plt.savefig('Histogram_ShellWeight.png', bbox_inches='tight')
    plt.close()


def check_stats(data):
    dicts = {'Column' : [],'Max' : [], 'Min' : [], 'Mean' : [], 'Median' : [], 'Std' : []}
    for i in data:
        dicts['Column'].append(i)
        dicts['Max'].append(data[i].max())
        dicts['Min'].append(data[i].min())
        dicts['Mean'].append(data[i].mean())
        dicts['Median'].append(data[i].median())
        dicts['Std'].append(round(data[i].std(), 2))
    stats_table = pd.DataFrame(dicts)
    return stats_table


# Check for missing values
def check_missing(data):
    Null_column_list = []
    for i in data.columns:
        if data[i].isnull().any():
            Null_column_list.append(i)
    return Null_column_list


def features_selection(data, target_column):
    data_corr = data.corr()
    data_corr.reset_index(inplace = True)
    data_corr.rename(columns={'index': 'Feature'}, inplace=True)
    data_corr = data_corr[data_corr['Feature'] != target_column]

    data_final = data_corr.sort_values(by = [target_column], ascending  = False).reset_index()
    #Extend two columns with highest correlation on both sides
    columns = []
    columns.extend([data_final.iloc[0, 1], data_final.iloc[1, 1]])
    
    return columns


# Train/Test Split
def Create_TrainTest_Split(data, seed, features_selection = False):
    if features_selection == False:
        #For whole datasets
        y = data['Rings']
        X = data.drop(columns = ['Rings'])
    else:
        #For selected dataset
        X = data[['Diameter', 'Shell_weight']]  # Selecting 'Diameter' and 'Shell_weight'
        y = data['Rings']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=seed)
    return X_train, X_test, y_train, y_test


# Normalize the data using MinMaxScaler
def normalize_data(X_train, X_test):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


# Linear Regression
def train_linear_regression(X_train, y_train):
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    return lr_model

# Evaluate Linear Regression
def evaluate_model(lr_model, X_test, y_test, figure_name = "Final"):
    y_pred = lr_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolor='black')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.title(f'Actual vs Predicted Ring Age (Linear Regression)\n {figure_name}')
    plt.xlabel('Actual Ring Age')
    plt.ylabel('Predicted Ring Age')
    plt.grid(True)
    plt.savefig(f'LinearRegression_Actual_Vs_Predicted_Age_{figure_name}.png', dpi=300)
    plt.close()
    return rmse, r2

# Evaluate Linear Regression (for both train and test)
def evaluate_model_train_test_regression(lr_model, X_train, y_train, X_test, y_test):
    #Train Accuracy
    y_train_pred = lr_model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)

    #Test dataset Accuracy
    y_test_pred = lr_model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)

    return train_rmse, train_r2, test_rmse, test_r2


# Comparison Function for Linear Regression (With and Without Normalization)
def compare_linear_regression(X_train, X_test, y_train, y_test, feature_selection = False, train_or_test = False):
    print("\nComparing Linear Regression Models:\n")

    if feature_selection == False:
        if train_or_test == False:
            # Without Normalization and without feature selection
            lr_model = train_linear_regression(X_train, y_train)
            rmse, r2 = evaluate_model(lr_model, X_test, y_test, figure_name = 'Without Normalization and without feature selection')
            print(f"Without Normalization and without feature selection -> RMSE: {rmse}, R-squared: {r2}")

            # With Normalization and without feature selection
            X_train_scaled, X_test_scaled = normalize_data(X_train, X_test)
            lr_model_scaled = train_linear_regression(X_train_scaled, y_train)
            rmse_scaled, r2_scaled = evaluate_model(lr_model_scaled, X_test_scaled, y_test, figure_name = 'With Normalization and without feature selection ')
            print(f"With Normalization and without feature selection -> RMSE: {rmse_scaled}, R-squared: {r2_scaled}")

        elif train_or_test == True:
            # Without Normalization and without feature selection
            lr_model = train_linear_regression(X_train, y_train)
            train_rmse, train_r2, test_rmse, test_r2 = evaluate_model_train_test_regression(lr_model, X_train, y_train, X_test, y_test, figure_name = "Training and Testing without Normalization and without feature selection" )
            print(f"Training without Normalization and without feature selection -> RMSE: {train_rmse}, R-squared: {train_r2}")
            print(f"Testing without Normalization and without feature selection -> RMSE: {test_rmse}, R-squared: {test_r2}")

            # With Normalization and without feature selection
            X_train_scaled, X_test_scaled = normalize_data(X_train, X_test)
            lr_model = train_linear_regression(X_train_scaled, y_train)
            train_rmse_scaled, train_r2_scaled, test_rmse_scaled, test_r2_scaled = evaluate_model_train_test_regression(lr_model, X_train_scaled, y_train, X_test_scaled, y_test, figure_name = "Training without Normalization and without feature selection")
            print(f"Training with Normalization and without feature selection -> RMSE: {train_rmse_scaled}, R-squared: {train_r2_scaled}")
            print(f"Testing with Normalization and without feature selection -> RMSE: {train_rmse_scaled}, R-squared: {train_r2_scaled}")

    
    else:
        if train_or_test == False:
            # Feature selection Without Normalization
            lr_model = train_linear_regression(X_train, y_train)
            rmse, r2 = evaluate_model(lr_model, X_test, y_test, figure_name = "After feature selection and without normalization")
            print(f"After feature selection and without normalization -> RMSE: {rmse}, R-squared: {r2}")

            # Feature selection with normalization
            X_train_scaled, X_test_scaled = normalize_data(X_train, X_test)
            lr_model_scaled = train_linear_regression(X_train_scaled, y_train)
            rmse_scaled, r2_scaled = evaluate_model(lr_model_scaled, X_test_scaled, y_test, figure_name = "After feature selection and with normalization")
            print(f"After feature selection and with normalization -> RMSE: {rmse_scaled}, R-squared: {r2_scaled}")
        
        elif train_or_test == True:
            # Without Normalization and without feature selection
            lr_model = train_linear_regression(X_train, y_train)
            train_rmse, train_r2, test_rmse, test_r2 = evaluate_model_train_test_regression(lr_model, X_train, y_train, X_test, y_test, figure_name = "Training and Testing without Normalization and without feature selection")
            print(f"Training after feature selection and without normalization -> RMSE: {train_rmse}, R-squared: {train_r2}")
            print(f"Testing after feature selection and without normalization -> RMSE: {test_rmse}, R-squared: {test_r2}")

            # With Normalization and without feature selection
            X_train_scaled, X_test_scaled = normalize_data(X_train, X_test)
            lr_model = train_linear_regression(X_train_scaled, y_train)
            train_rmse_scaled, train_r2_scaled, test_rmse_scaled, test_r2_scaled = evaluate_model_train_test_regression(lr_model, X_train_scaled, y_train, X_test_scaled, y_test, "Training and Testing after feature selection and with normalization")
            print(f"Training after feature selection and with normalization -> RMSE: {train_rmse_scaled}, R-squared: {train_r2_scaled}")
            print(f"Testing after feature selection and with normalization -> RMSE: {test_rmse_scaled}, R-squared: {test_r2_scaled}")


# Experiment for Linear Regression
def LinReg_exper(X_train, X_test, y_train, y_test, feature_selection = False, experiment_no = "Final"):
    if feature_selection == False:
        # Without Normalization and without feature selection
        lr_model = train_linear_regression(X_train, y_train)
        rmse, r2 = evaluate_model(lr_model, X_test, y_test, figure_name = experiment_no)

        return rmse, r2
    
    else:
        # Feature selection with normalization
        X_train_scaled, X_test_scaled = normalize_data(X_train, X_test)
        lr_model_scaled = train_linear_regression(X_train_scaled, y_train)
        rmse_scaled, r2_scaled = evaluate_model(lr_model_scaled, X_test_scaled, y_test, figure_name =  experiment_no)

        return rmse_scaled, r2_scaled


# Logistic Regression Training
def train_logistic_regression(X_train, y_train):
    logreg_model = LogisticRegression(max_iter=1000)
    logreg_model.fit(X_train, y_train)
    return logreg_model

# Logistic Regression Evaluation
def evaluate_logistic_regression(logreg_model, X_test, y_test, figure_name = "Final"):
    y_pred_prob = logreg_model.predict_proba(X_test)[:, 1]
    y_pred = logreg_model.predict(X_test)
    auc = roc_auc_score(y_test, y_pred_prob)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - Logistic Regression \n{figure_name}')
    plt.savefig(f'Logistic Regression Confusion_Matrix {figure_name}.png', dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}', color='blue')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.title(f'ROC Curve (Logistic Regression) \n{figure_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig(f'Actual_Vs_Predicted_Classification_{figure_name}.png', dpi=300)
    plt.close()
    
    return auc, accuracy

# Logistic Regression Evaluation
def evaluate_model_train_test_logistic_regression(logreg_model, X_train, X_test, y_train, y_test, plot_roc=False, figure_name = "Final"):
    #Train Accuracy
    y_train_pred_prob = logreg_model.predict_proba(X_train)[:, 1]
    y_train_pred = logreg_model.predict(X_train)
    auc_train = roc_auc_score(y_train, y_train_pred)
    accuracy_train = accuracy_score(y_train, y_train_pred)

    #Test dataset Accuracy
    y_test_pred_prob = logreg_model.predict_proba(X_test)[:, 1]
    y_test_pred = logreg_model.predict(X_test)
    auc_test = roc_auc_score(y_test, y_test_pred)
    accuracy_test = accuracy_score(y_test, y_test_pred)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - LogisticRegression Testing set \n{figure_name}')
    plt.savefig(f'LogisticRegression_ConfusionMatrix_Training_{figure_name}.png', dpi=300)
    plt.close()

    if plot_roc:
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}', color='blue')
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')
        plt.title(f'ROC Curve (Logistic Regression) for testing set \n{figure_name}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.savefig(f'Testingset_Logistic_Regression_ROC_{figure_name}.png', dpi=300)
        plt.show()
    

    # Confusion Matrix
    cm = confusion_matrix(y_train, y_train_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - LogisticRegression Training set \n{figure_name}')
    plt.savefig(f'LogisticRegression_ConfusionMatrix_Testing_{figure_name}.png', dpi=300)
    plt.close()

    if plot_roc:
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}', color='blue')
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')
        plt.title(f'ROC Curve (Logistic Regression) for testing set \n{figure_name}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.savefig(f'Trainingset_Logistic_Regression_ROC_{figure_name}.png', dpi=300)
        plt.show()
    
    return auc_train, accuracy_train, auc_test, accuracy_test


# Comparison Function for Logistic Regression (With and Without Normalization)
def compare_logistic_regression(X_train, X_test, y_train, y_test, feature_selection = False, train_or_test = False):
    print("\nComparing Logistic Regression Models:\n")

    if feature_selection == False:
        if train_or_test == False:
            # Without Normalization and without feature selection
            logreg_model = train_logistic_regression(X_train, y_train)
            auc, accuracy = evaluate_logistic_regression(logreg_model, X_test, y_test, "Without Normalization and Without feature selection")
            print(f"Without Normalization and Without feature selection -> AUC: {auc}, Accuracy: {accuracy}")

            # With Normalization and without feature selection
            X_train_scaled, X_test_scaled = normalize_data(X_train, X_test)
            logreg_model_scaled = train_logistic_regression(X_train_scaled, y_train)
            auc_scaled, accuracy_scaled = evaluate_logistic_regression(logreg_model_scaled, X_test_scaled, y_test, "With Normalization and without feature selection")
            print(f"With Normalization and without feature selection -> AUC: {auc_scaled}, Accuracy: {accuracy_scaled}")

        elif train_or_test == True:
            # Without Normalization and without feature selection
            logreg_model = train_logistic_regression(X_train, y_train)
            auc_train, accuracy_train, auc_test, accuracy_test = evaluate_model_train_test_logistic_regression(logreg_model, X_train, X_test, y_train, y_test, plot_roc =True, figure_name = "Training and Testing accuracy without Normalization and Without feature selection")
            print(f"Training accuracy without Normalization and Without feature selection -> AUC: {auc_train}, Accuracy: {accuracy_train}")
            print(f"Testing accuracy without Normalization and Without feature selection -> AUC: {auc_test}, Accuracy: {accuracy_test}")

            # With Normalization and without feature selection
            X_train_scaled, X_test_scaled = normalize_data(X_train, X_test)
            logreg_model = train_logistic_regression(X_train_scaled, y_train)
            auc_scaled_train, accuracy_scaled_train, auc_scaled_test, accuracy_scaled_test = evaluate_model_train_test_logistic_regression(logreg_model, X_train_scaled, X_test_scaled, y_train, y_test, figure_name = "Training accuracy with Normalization and without feature selection")
            print(f"Training accuracy with Normalization and without feature selection -> AUC: {auc_scaled_train}, Accuracy: {accuracy_scaled_train}")
            print(f"Testing accuracy with Normalization and without feature selection -> AUC: {auc_scaled_test}, Accuracy: {accuracy_scaled_test}")

    else:
        if train_or_test == False:
            # Feature selection without normalization
            logreg_model = train_logistic_regression(X_train, y_train)
            auc, accuracy = evaluate_logistic_regression(logreg_model, X_test, y_test, "Feature selection without normalization")
            print(f"Feature selection without normalization -> AUC: {auc}, Accuracy: {accuracy}")

            # feature selection with normalization
            X_train_scaled, X_test_scaled = normalize_data(X_train, X_test)
            logreg_model_scaled = train_logistic_regression(X_train_scaled, y_train)
            auc_scaled, accuracy_scaled = evaluate_logistic_regression(logreg_model_scaled, X_test_scaled, y_test, "Feature selection with normalization")
            print(f"Feature selection with normalization -> AUC: {auc_scaled}, Accuracy: {accuracy_scaled}")

        elif train_or_test == True:
            # Without Normalization and without feature selection
            logreg_model = train_logistic_regression(X_train, y_train)
            auc_train, accuracy_train, auc_test, accuracy_test = evaluate_model_train_test_logistic_regression(logreg_model, X_train, X_test, y_train, y_test, figure_name = "Training and Testing accuracy Feature selection without normalization")
            print(f"Training accuracy Feature selection without normalization -> AUC: {auc_train}, Accuracy: {accuracy_train}")
            print(f"Testing accuracy Feature selection without normalization -> AUC: {auc_test}, Accuracy: {accuracy_test}")

            # With Normalization and without feature selection
            X_train_scaled, X_test_scaled = normalize_data(X_train, X_test)
            logreg_model = train_logistic_regression(X_train_scaled, y_train)
            auc_scaled_train, accuracy_scaled_train, auc_scaled_test, accuracy_scaled_test = evaluate_model_train_test_logistic_regression(logreg_model, X_train_scaled, X_test_scaled, y_train, y_test, figure_name = "Training and Testing accuracy feature selection with normalization")
            print(f"Training accuracy feature selection with normalization -> AUC: {auc_scaled_train}, Accuracy: {accuracy_scaled_train}")
            print(f"Testing accuracy feature selection with normalization -> AUC: {auc_scaled_test}, Accuracy: {accuracy_scaled_test}")


#Experiment for Logistic Regression
def LogReg_exper(X_train, X_test, y_train, y_test, feature_selection = False, experi_number = 0):
    if feature_selection == False:
        # Without Normalization and without feature selection
        logreg_model = train_logistic_regression(X_train, y_train)
        auc, accuracy = evaluate_logistic_regression(logreg_model, X_test, y_test, experi_number)

        return auc, accuracy
    
    else:
        X_train_scaled, X_test_scaled = normalize_data(X_train, X_test)
        logreg_model_scaled = train_logistic_regression(X_train_scaled, y_train)
        auc_scaled, accuracy_scaled = evaluate_logistic_regression(logreg_model_scaled, X_test_scaled, y_test,  experi_number)
        
        return auc_scaled, accuracy_scaled


# Neural Network for Regression
def train_neural_network_regression(X_train, y_train, hidden_layers=(100,), learning_rate=0.001, epoch = 100):
    nn_model = MLPRegressor(hidden_layer_sizes=hidden_layers, learning_rate_init=learning_rate, max_iter= epoch, solver='sgd')
    nn_model.fit(X_train, y_train)
    return nn_model

# Neural Network for Classification
def train_neural_network_classification(X_train, y_train, hidden_layers=(100,), learning_rate=0.001, epoch = 100):
    nn_model = MLPClassifier(hidden_layer_sizes=hidden_layers, learning_rate_init=learning_rate, max_iter= epoch, solver='sgd')
    nn_model.fit(X_train, y_train)
    return nn_model

# Evaluation for Neural Network Regression
def evaluate_nn_regression(nn_model, X_test, y_test):
    y_pred = nn_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"Neural Network Regression -> RMSE: {rmse}, R-squared: {r2}")
    return rmse, r2


# Evaluation for Neural Network Classification with ROC Curve plotting
def evaluate_nn_classification(nn_model, X_test, y_test, plot_roc=False, figure_name = "Final"):
    y_pred_prob = nn_model.predict_proba(X_test)[:, 1]
    y_pred = nn_model.predict(X_test)
    auc = roc_auc_score(y_test, y_pred_prob)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Neural Network Classification -> AUC: {auc}, Accuracy: {accuracy}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - Neural Network \n{figure_name}')
    plt.savefig(f'Neural_Network_Confusion_Matrix_{figure_name}.png', dpi=300)
    plt.close()

    if plot_roc:
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}', color='blue')
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')
        plt.title(f'ROC Curve (Neural Network) \n {figure_name}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.savefig(f'Best_Neural_Network_ROC_{figure_name}.png', dpi=300)
        plt.show()
    return auc, accuracy


# Hyperparameter trial function for Neural Networks
def run_trial_neural_networks(X_train, X_test, y_train, y_test, mode='regression'):
    hidden_layers = [(50,), (100,), (100, 50), (100, 50, 20) ]
    learning_rates = [0.01, 0.001, 0.0001]
    epochs = [100, 300,  500, 700, 1000]
    
    best_rmse, best_r2, best_auc, best_accuracy = None, None, None, None
    best_hidden_layer, best_learning_rate, best_epoch = None, None, None  # Track best hidden layer and learning rate

    for hidden_layer in hidden_layers:
        for lr in learning_rates:
            for epoch in epochs:
                print(f"\nTrial with hidden layers {hidden_layer}, learning rate {lr} and epoch {epoch}")
                
                if mode == 'regression':
                    nn_model = train_neural_network_regression(X_train, y_train, hidden_layers=hidden_layer, learning_rate=lr, epoch = epoch)
                    rmse, r2 = evaluate_nn_regression(nn_model, X_test, y_test)
                    if best_rmse is None or rmse < best_rmse:
                        best_rmse, best_r2 = rmse, r2
                        best_hidden_layer, best_learning_rate, best_epoch = hidden_layer, lr, epoch  # Store best configuration

                elif mode == 'classification':
                    nn_model = train_neural_network_classification(X_train, y_train, hidden_layers=hidden_layer, learning_rate=lr, epoch = epoch)
                    auc, accuracy = evaluate_nn_classification(nn_model, X_test, y_test)
                    if best_auc is None or auc > best_auc:
                        best_auc, best_accuracy = auc, accuracy
                        best_hidden_layer, best_learning_rate, best_epoch = hidden_layer, lr, epoch  # Store best configuration
                        best_model = nn_model  # Store the best model for ROC plot

    if mode == 'regression':
        print(f"\nBest Neural Network Regression -> RMSE: {best_rmse}, R-squared: {best_r2}")
        print(f"Best Configuration -> Hidden Layers: {best_hidden_layer}, Learning Rate: {best_learning_rate}, epoch: {best_epoch}")
        return best_hidden_layer, best_learning_rate, best_epoch
        
    elif mode == 'classification':
        print(f"\nBest Neural Network Classification -> AUC: {best_auc}, Accuracy: {best_accuracy}")
        print(f"Best Configuration -> Hidden Layers: {best_hidden_layer}, Learning Rate: {best_learning_rate}, epoch: {best_epoch}")
        print("Plotting ROC curve for the best model...")
        evaluate_nn_classification(best_model, X_test, y_test, plot_roc=True)
        return best_hidden_layer, best_learning_rate, best_epoch

# Function to run the entire experiment
def run_experiment(features_selected=False, n_exper=3):
    # Load the data
    Data = import_data(filename)

    # Clean and preprocess data
    data_conversion(Data)

    # Generate Correlation Heatmap
    correlation_heatmap(Data)

    # Display Statistical Properties
    print(f"Statistical Property: \n {check_stats(Data)}")

    # Scatter Plots and Histograms
    scatter_plots(Data)
    histograms(Data)

    # Log transformation
    Data = log_transform(Data)

    # Apply feature selection if set to True
    if features_selected:
        columns = features_selection(Data, Target_column)
        columns.append(Target_column)
        Data = Data[columns]
        print(f"\n Features selected: {columns}")

    # Train/Test Split
    Experi_No = 7
    X_train, X_test, y_train, y_test = Create_TrainTest_Split(Data, Experi_No, features_selection=features_selected)

    # Run Linear Regression comparison with and without normalization
    compare_linear_regression(X_train, X_test, y_train, y_test, feature_selection=features_selected, train_or_test=False)

    # Binary Classification
    Data['Rings'] = (Data['Rings'] >= 7).astype(int)  # Binary classification: 0 if below 7, 1 if 7 and above
    X_train_class, X_test_class, y_train_class, y_test_class = Create_TrainTest_Split(Data, Experi_No, features_selection=features_selected)

    # Run Logistic Regression comparison with and without normalization
    compare_logistic_regression(X_train_class, X_test_class, y_train_class, y_test_class, feature_selection=features_selected, train_or_test=False)

    # Neural Network for regression
    print("\nRunning Neural Network Trials for Regression:")
    best_hidden_layer, best_learning_rate, best_epoch = run_trial_neural_networks(X_train, X_test, y_train, y_test, mode='regression')

    # Final Neural Network Regression Model
    print("\nFinal model for Neural Network Regression: ")
    nn_model_reg = train_neural_network_regression(X_train, y_train, hidden_layers= best_hidden_layer, learning_rate=best_learning_rate, epoch=best_epoch)
    rmse, r2 = evaluate_nn_regression(nn_model_reg, X_train, y_train )
    print(f"Training set RMSE: {rmse}, Training set R2: {r2}")
    rmse, r2 = evaluate_nn_regression(nn_model_reg, X_test, y_test)
    print(f"Testing set Final RMSE: {rmse}, Testing set Final R2: {r2}")


    # Neural Network for classification
    print("\nRunning Neural Network Trials for Classification:")
    best_hidden_layer_class, best_learning_rate_class, best_epoch_class = run_trial_neural_networks(X_train_class, X_test_class, y_train_class, y_test_class, mode='classification')


    # Final Neural Network Classification Model
    print("\nFinal model for Neural Network Classification: ")
    nn_model_class = train_neural_network_classification(X_train_class, y_train_class, hidden_layers=best_hidden_layer_class, learning_rate=best_learning_rate_class, epoch=best_epoch_class)
    auc, accuracy = evaluate_nn_classification(nn_model_class, X_train_class, y_train_class, str("Final model:"))
    print(f"Training set AUC: {auc}, Training set Accuracy: {accuracy}")
    auc, accuracy = evaluate_nn_classification(nn_model_class, X_test_class, y_test_class, str("Final model:"))
    print(f"Testing set Final AUC: {auc}, Testing set Final Accuracy: {accuracy}")

    # Run the experiments multiple times
    Rmse_val_lin, R2_val_lin = [], []
    AUC_val_log, Accuracy_val_log = [], []
    Rmse_val_nn, R2_val_nn = [], []
    AUC_val_nn, Accuracy_val_nn = [], []

    for i in range(n_exper):
        # Reload the data
        Data = import_data(filename)
        data_conversion(Data)

        #spliting with same seed 
        X_train, X_test, y_train, y_test = Create_TrainTest_Split(Data, Experi_No, features_selection= features_selected)

        # Experiment for linear regression model
        Rmse, R2 = LinReg_exper(X_train, X_test, y_train, y_test, feature_selection=features_selected, experiment_no = f' Exper {str(Experi_No)}')
        Rmse_val_lin.append(Rmse)
        R2_val_lin.append(R2)

        # Neural Network model - Regression
        nn_model_reg = train_neural_network_regression(X_train, y_train, hidden_layers=best_hidden_layer, learning_rate=best_learning_rate, epoch=best_epoch)
        rmse, r2 = evaluate_nn_regression(nn_model_reg, X_test, y_test)
        Rmse_val_nn.append(rmse)
        R2_val_nn.append(r2)

        # Binary Classification for Logistic Regression
        Data['Rings'] = (Data['Rings'] >= 7).astype(int)
        X_train_class, X_test_class, y_train_class, y_test_class = Create_TrainTest_Split(Data, Experi_No, features_selection=features_selected)

        # Logistic Regression model
        auc, accuracy = LogReg_exper(X_train_class, X_test_class, y_train_class, y_test_class, feature_selection=features_selected, experi_number = f' Exper {str(Experi_No)}')
        AUC_val_log.append(auc)
        Accuracy_val_log.append(accuracy)

        # Neural Network Classification
        nn_model_class = train_neural_network_classification(X_train_class, y_train_class, hidden_layers=best_hidden_layer_class, learning_rate=best_learning_rate_class, epoch=best_epoch_class)
        auc, accuracy = evaluate_nn_classification(nn_model_class, X_test_class, y_test_class)
        AUC_val_nn.append(auc)
        Accuracy_val_nn.append(accuracy)

    # Summary of results
    print(f"Mean_Rmse_LinReg = {np.mean(Rmse_val_lin)}")
    print(f"Mean_R2_LinReg = {np.mean(R2_val_lin)}")
    print(f"Std_Rmse_LinReg = {np.std(Rmse_val_lin)}")
    print(f"Std_R2_LinReg = {np.std(R2_val_lin)}")

    print(f"Mean_auc_LogReg = {np.mean(AUC_val_log)}")
    print(f"Mean_accuracy_LogReg = {np.mean(Accuracy_val_log)}")
    print(f"Std_auc_LogReg = {np.std(AUC_val_log)}")
    print(f"Std_accuracy_LogReg = {np.std(Accuracy_val_log)}")

    print("\n")
    print(f"Mean_Rmse_nn = {np.mean(Rmse_val_nn)}")
    print(f"Mean_R2_nn = {np.mean(R2_val_nn)}")
    print(f"Std_Rmse_nn = {np.std(Rmse_val_nn)}")
    print(f"Std_R2_nn = {np.std(R2_val_nn)}")

    print(f"Mean_auc_nn = {np.mean(AUC_val_nn)}")
    print(f"Mean_accuracy_nn= {np.mean(Accuracy_val_nn)}")
    print(f"Std_auc_nn = {np.std(AUC_val_nn)}")
    print(f"Std_accuracy_nn = {np.std(Accuracy_val_nn)}")

if __name__ == "__main__":
    # Importing Data
    filename = "abalone.data"

    # Running the experiment without feature selection 
    print("Running the experiment without feature selection")
    run_experiment(features_selected=False)

    # Running the experiment with feature selection
    print("Running the experiment with feature selection")
    run_experiment(features_selected=True)

