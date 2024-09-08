import fedint
from fedint import analyze_features, NAM, Config

# Run the feature analysis using the built-in sklearn 'wine' dataset
if __name__ == "__main__":
    # Analyze the features for the wine dataset
    high_contrib, low_contrib = analyze_features(dataset_name='iris', nam_model_class=NAM, config_class=Config)

    # Print the most and least contributing features
    print("High contributing features:", high_contrib)
    print("Low contributing features:", low_contrib)
