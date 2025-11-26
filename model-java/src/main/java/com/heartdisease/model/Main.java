package com.heartdisease.model;

import java.io.IOException;

/**
 * Main class to train and evaluate the Random Forest model
 */
public class Main {
    public static void main(String[] args) {
        try {
            // Load dataset
            System.out.println("Loading dataset...");
            String dataPath = "../python-analysis/artifacts/features_model_ready.csv";
            Dataset dataset = DataLoader.loadFromCSV(dataPath);

            System.out.println("Dataset loaded:");
            System.out.println("  Samples: " + dataset.getNumSamples());
            System.out.println("  Features: " + dataset.getNumFeatures());

            // Split into train and test
            System.out.println("\nSplitting dataset (80% train, 20% test)...");
            DataLoader.DataSplit split = DataLoader.trainTestSplit(dataset, 0.2, 42L);
            Dataset trainSet = split.getTrainSet();
            Dataset testSet = split.getTestSet();

            System.out.println("  Train samples: " + trainSet.getNumSamples());
            System.out.println("  Test samples: " + testSet.getNumSamples());

            // Train Random Forest
            System.out.println("\nTraining Random Forest...");
            int numTrees = 100;
            int maxDepth = 10;
            int minSamplesSplit = 2;
            int maxFeatures = (int) Math.sqrt(dataset.getNumFeatures());

            RandomForest rf = new RandomForest(numTrees, maxDepth, minSamplesSplit, maxFeatures, 42L);

            System.out.println("  Number of trees: " + numTrees);
            System.out.println("  Max depth: " + maxDepth);
            System.out.println("  Min samples split: " + minSamplesSplit);
            System.out.println("  Max features: " + maxFeatures);

            long startTime = System.currentTimeMillis();
            rf.fit(trainSet);
            long endTime = System.currentTimeMillis();

            System.out.println("  Training completed in " + (endTime - startTime) + "ms");

            // Evaluate on training set
            double trainAccuracy = rf.score(trainSet);
            System.out.println("\nTraining accuracy: " + String.format("%.4f", trainAccuracy));

            // Evaluate on test set
            double testAccuracy = rf.score(testSet);
            System.out.println("Test accuracy: " + String.format("%.4f", testAccuracy));

            // Example prediction
            System.out.println("\nExample prediction on first test sample:");
            double[] sample = testSet.getSample(0);
            int prediction = rf.predict(sample);
            double probability = rf.predictProbability(sample);
            int actualLabel = testSet.getLabel(0);

            System.out.println("  Predicted class: " + prediction);
            System.out.println("  Probability: " + String.format("%.4f", probability));
            System.out.println("  Actual class: " + actualLabel);
            System.out.println("  Correct: " + (prediction == actualLabel));

        } catch (IOException e) {
            System.err.println("Error loading dataset: " + e.getMessage());
            e.printStackTrace();
        } catch (Exception e) {
            System.err.println("Error during training: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
