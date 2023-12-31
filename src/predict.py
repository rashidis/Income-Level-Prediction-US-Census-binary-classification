import json
import pandas as pd
import pickle
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
    X = df_val.drop(columns=["income"])
    y = df_val["income"]

    # Load model
    with open('../results/model.pkl', 'rb') as file:
        model = pickle.load(file)
        
    # Predict
    y_pred_val = model.predict(X)
    print("\nClassification Report:\n", classification_report(y, y_pred_val))
    
    # Save results
    X['income'] = y
    X['preds'] = y_pred_val
    X.to_csv('../results/validation_preds.csv', index=False)
    
    with open('../results/val_classification_report.txt', 'w') as output_file:
        output_file.write(str(classification_report(y, y_pred_val)))