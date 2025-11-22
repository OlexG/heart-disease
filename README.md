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
└── src/                            # Java Random Forest implementation
    ├── RandomForest.java           # Random Forest classifier
    ├── DecisionTree.java           # Decision tree implementation
    ├── Dataset.java                # Data container
    ├── DataLoader.java             # CSV loading utilities
    ├── TreeMatrix.java             # Matrix operations for tree building
    └── Main.java                   # Example usage
```

## Prerequisites

- **Java 11** or higher
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

Run the project directly from the command line (no external build tools required).

**From the project root directory:**

```bash
# Compile
javac src/*.java

# Run (default: visualizes tree #0)
java -cp src Main

# Run (optional: visualize specific tree index, e.g., tree #5)
java -cp src Main 5
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

=== Detailed Predictions on First 5 Test Samples ===

Sample #1
  Prediction: No Heart Disease (XX.X% confidence)
  Actual:     No Heart Disease
  Result:     CORRECT
```

### 4. Visualizing the Decision Tree

After training, the program generates a `.dot` file for the visualized tree in the project root (default `tree_viz_0.dot`). This file represents the structure of the specific decision tree from the forest.

To view this visualization:

1.  **Install Graphviz** (if not already installed).
2.  **Convert the .dot file to an image** using the command line:
    # Example for tree #0
    dot -Tpng tree_viz_0.dot -o tree_viz_0.png
    3.  Open `tree_viz_0.png` to see the decision nodes and splits.

*Note: You can also copy the contents of the `.dot` file and paste them into online viewers like [Viz-js.com](http://viz-js.com/) or [GraphvizOnline](https://dreampuf.github.io/GraphvizOnline/).*

## Model Configuration

The Random Forest model can be configured in [`src/Main.java`](src/Main.java):

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

## Dependencies

- **None** (Standard Java Standard Library only)

## Implementation Details

The Random Forest implementation includes:
- **Bootstrap Aggregating (Bagging)**: Each tree trained on random sample with replacement
- **Random Feature Selection**: Each split considers random subset of features
- **Information Gain Ratio (Entropy)**: Criterion for selecting best splits
- **Majority Voting**: Final prediction based on votes from all trees
- **Probability Estimation**: Based on proportion of positive votes

## License

This is an educational project.
