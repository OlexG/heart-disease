# Heart Disease Prediction Project

This project implements a Random Forest classifier from scratch in Java for predicting heart disease based on clinical and lifestyle data.

## Project Structure

```
heart-disease/
├── data/                           # Raw dataset
│   └── heart_disease.csv
├── python-analysis/                # Data preprocessing and EDA
│   ├── heart_disease_eda.ipynb    # Jupyter notebook for analysis
│   └── artifacts/
│       ├── features_model_ready.csv      # Preprocessed data (ready for ML)
│       └── heart_disease_cleaned.csv     # Cleaned data
└── model-java/                     # Java Random Forest implementation
    ├── pom.xml
    └── src/main/java/com/heartdisease/model/
        ├── RandomForest.java       # Random Forest classifier
        ├── DecisionTree.java       # Decision tree implementation
        ├── Dataset.java            # Data container
        ├── DataLoader.java         # CSV loading utilities
        └── Main.java               # Example usage
```

## Prerequisites

- **Java 11** or higher
- **Maven 3.6+**
- **Python 3.8+** (for data preprocessing)

## Setup and Running

### 1. Data Preprocessing (Python)

First, prepare the data using the Jupyter notebook:

```bash
cd python-analysis
jupyter notebook heart_disease_eda.ipynb
```

Run all cells to generate the preprocessed data files in `python-analysis/artifacts/`.

### 2. Build and Run Java Model

Navigate to the Java project directory and build:

```bash
cd model-java
mvn clean compile
```

Run the Random Forest model:

```bash
mvn exec:java -Dexec.mainClass="com.heartdisease.model.Main"
```

### 3. Expected Output

The program will:
- Load the preprocessed dataset (10,000 samples, 20 features)
- Split into 80% training, 20% test
- Train a Random Forest with 100 trees
- Report training and test accuracy
- Show an example prediction

Example output:
```
Loading dataset...
Dataset loaded:
  Samples: 10000
  Features: 20

Splitting dataset (80% train, 20% test)...
  Train samples: 8000
  Test samples: 2000

Training Random Forest...
  Number of trees: 100
  Max depth: 10
  Min samples split: 2
  Max features: 4
  Training completed in XXXXms

Training accuracy: 0.XXXX
Test accuracy: 0.XXXX

Example prediction on first test sample:
  Predicted class: X
  Probability: 0.XXXX
  Actual class: X
  Correct: true/false
```

## Model Configuration

The Random Forest model can be configured in [model-java/src/main/java/com/heartdisease/model/Main.java](model-java/src/main/java/com/heartdisease/model/Main.java):

- `numTrees`: Number of decision trees (default: 100)
- `maxDepth`: Maximum depth of each tree (default: 10)
- `minSamplesSplit`: Minimum samples required to split a node (default: 2)
- `maxFeatures`: Number of features to consider for each split (default: sqrt(num_features))

## Dataset Features

The model uses 20 numeric features:
- **Demographics**: age, gender
- **Medical measurements**: blood_pressure, cholesterol_level, bmi, triglyceride_level, fasting_blood_sugar, crp_level, homocysteine_level
- **Lifestyle factors**: exercise_habits, smoking, alcohol_consumption, stress_level, sleep_hours, sugar_consumption
- **Medical history**: family_heart_disease, diabetes, high_blood_pressure, low_hdl_cholesterol, high_ldl_cholesterol

**Target**: heart_disease_status (0 = No, 1 = Yes)

## Running Tests

```bash
cd model-java
mvn test
```

## Dependencies

Java dependencies (managed by Maven):
- Apache Commons CSV 1.10.0 - CSV file parsing
- Apache Commons Math3 3.6.1 - Statistical functions
- JUnit 5.9.3 - Testing framework

## Implementation Details

The Random Forest implementation includes:
- **Bootstrap Aggregating (Bagging)**: Each tree trained on random sample with replacement
- **Random Feature Selection**: Each split considers random subset of features
- **Gini Impurity**: Criterion for selecting best splits
- **Majority Voting**: Final prediction based on votes from all trees
- **Probability Estimation**: Based on proportion of positive votes

## License

This is an educational project.
