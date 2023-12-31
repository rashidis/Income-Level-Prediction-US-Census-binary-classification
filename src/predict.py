import json
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import classification_report
from preprocessing import preprocessing
from preprocessing import encoding


if __name__ == "__main__":
    
    with open("./config.json", 'r') as json_file:
        config = json.load(json_file)
    
    val_file_path = config['train_file_path']
    numerical_features = config['numerical_features']
    cols2drop = config['cols2drop']
    unknown_values = config['unknown_values']

    # Read data
    df = pd.read_csv(val_file_path)
    
    # Preprocess data
    print('preprocessing started---------------------------------------------')
    df_val = preprocessing(df, numerical_features, cols2drop, unknown_values)
    
    # Encode data
    print('encoding started--------------------------------------------')
    categorical_features = list(set(df_val.columns.tolist()) - set(numerical_features) - set(['income']))
    df_val = encoding(df_val, categorical_features)
    
    print(f'shape of validation dataset is {df_val.shape}')
    
    # Setup validation data
    if 'income' in df_val.columns:  # if the dataset is a validation dataset
        X = df_val.drop(columns=["income"])
        y = df_val["income"]
    else:
        X = df_val

    # Load model
    with open('../results/model.pkl', 'rb') as file:
        model = pickle.load(file)
        
    # Predict
    y_pred_val = model.predict(X)
        
    # Exaplainability
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    waterfall = shap.summary_plot(shap_values, X, feature_names=X.columns, show=False)
    plt.savefig('../results/validation_waterfall.png')

    class_index = 0 #below 50k
    data_index = 0
    force_plot = shap.force_plot(explainer.expected_value[class_index], 
                shap_values[class_index][data_index, :], X.iloc[data_index, :], 
                matplotlib=matplotlib, show=False)
    plt.savefig(f'../results/val_force_plot_class{class_index}_index{data_index}.png')
    
    # Save results
    if 'incom' in df_val.columns:  # if the dataset is a validation dataset
        print("\nClassification Report:\n", classification_report(y, y_pred_val))
        with open('../results/val_classification_report.txt', 'w') as output_file:
            output_file.write(str(classification_report(y, y_pred_val)))
        X['income'] = y
        
    X['preds'] = y_pred_val
    X.to_csv('../results/prediction_results.csv', index=False)
    

