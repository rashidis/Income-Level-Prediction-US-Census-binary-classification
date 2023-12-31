import json
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessing import preprocessing
from imblearn.over_sampling import SMOTENC
from lightgbm import LGBMClassifier
from preprocessing import encoding


if __name__ == "__main__":
    
    with open("./config.json", 'r') as json_file:
        config = json.load(json_file)
    
    train_file_path = config['train_file_path']
    numerical_features = config['numerical_features']
    cols2drop = config['cols2drop']
    unknown_values = config['unknown_values']

    # Read data
    df = pd.read_csv(train_file_path)
    
    # Preprocess data
    print('preprocessing started---------------------------------------------')
    df_train = preprocessing(df, numerical_features, cols2drop, unknown_values)
    
    # Encode data
    print('encoding started--------------------------------------------')
    categorical_features = list(set(df_train.columns.tolist()) - set(numerical_features) - set(['income']))
    df_train = encoding(df_train, categorical_features)
    
    print(f'shape of train dataset is {df_train.shape}')
    
    # Test and train setup data
    X = df_train.drop(columns=["income"])
    y = df_train["income"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

    smote_nc = SMOTENC(categorical_features=categorical_features, random_state=42)
    X_resampled, y_resampled = smote_nc.fit_resample(X_train, y_train)
    print('dataset balanced')
    print('value counts after oversampling', pd.DataFrame(y_resampled).value_counts())

    # Train model
    model = LGBMClassifier(random_state=42)
    model.fit(X_resampled, y_resampled)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # Save model and results   
    with open('../results/model.pkl', 'wb') as file:
        pickle.dump(model, file)
        
    X_test['income'] = y_test
    X_test['preds'] = y_pred
    X.to_csv('../results/test_preds.csv', index=False)
    
    with open('../results/test_classification_report.txt', 'w') as output_file:
        output_file.write(str(classification_report(y_test, y_pred)))