from sklearn.datasets import fetch_openml, load_wine, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import pandas as pd

def load_dataset(dataset_name: str):
    """Load inbuilt sklearn datasets."""
    if dataset_name == 'wine':
        data = load_wine()
    elif dataset_name == 'iris':
        data = load_iris()
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")
    
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df

def analyze_features(dataset_name, nam_model_class, config_class, num_clients=3):
    """Analyze features using a federated learning setup with inbuilt sklearn datasets."""
    import numpy as np
    df = load_dataset(dataset_name)  # Load the dataset using the helper function
    target_column = 'target'
    
    # Split dataset for federated learning simulation
    n_clients = num_clients
    client_data = [df.iloc[i * len(df) // n_clients: (i + 1) * len(df) // n_clients] for i in range(n_clients)]
    feature_columns = df.columns.drop(target_column)

    clients_high_contrib = {}
    clients_low_contrib = {}

    for i, client_df in enumerate(client_data):
        X = client_df[feature_columns].values
        y = client_df[target_column].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normalize the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        config = config_class()
        num_inputs = len(feature_columns)
        nam_model = nam_model_class(config=config, name=f"Client_{i}_Model", num_inputs=num_inputs, num_units=10)

        trained_model = train(nam_model, X_train_tensor, y_train_tensor, config)
        accuracy = evaluate(trained_model, X_test_tensor, y_test_tensor)
        print(f"Client {i} Accuracy: {accuracy * 100:.2f}%")

        high_contrib, low_contrib = trained_model.print_model_equation(feature_columns)
        clients_high_contrib[i] = high_contrib
        clients_low_contrib[i] = low_contrib

    return clients_high_contrib, clients_low_contrib
