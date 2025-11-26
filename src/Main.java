import java.io.IOException;

/**
 * Main class to train and evaluate the Random Forest model
 */
public class Main {
    public static void main(String[] args) {
        // Parse command line argument for tree index (default to 0)
        int treeIndexToVisualize = 0;
        if (args.length > 0) {
            try {
                treeIndexToVisualize = Integer.parseInt(args[0]);
            } catch (NumberFormatException e) {
                System.err.println("Invalid argument for tree index. Using default: 0");
            }
        }

        try {
            // Load dataset
            System.out.println("Loading dataset...");
            String dataPath = "python-analysis/artifacts/features_model_ready.csv";
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

            // Save the specified tree to a DOT file
            if (rf.getNumTrees() > 0) {
                if (treeIndexToVisualize >= 0 && treeIndexToVisualize < rf.getNumTrees()) {
                    String filename = "tree_viz_" + treeIndexToVisualize + ".dot";
                    System.out.println("\nSaving visualization of tree " + treeIndexToVisualize + " to '" + filename + "'...");
                    String dotContent = rf.getTrees().get(treeIndexToVisualize).toDotString(dataset.getFeatureNames());
                    
                    try (java.io.PrintWriter out = new java.io.PrintWriter(filename)) {
                        out.println(dotContent);
                        System.out.println("Done.");
                    } catch (java.io.IOException e) {
                        System.err.println("Failed to write DOT file: " + e.getMessage());
                    }
                } else {
                    System.err.println("\nTree index " + treeIndexToVisualize + " is out of bounds (0-" + (rf.getNumTrees() - 1) + "). Skipping visualization.");
                }
            }

            // Example prediction with better formatting
            System.out.println("\n=== Detailed Predictions on First 5 Test Samples ===");
            String[] classNames = {"No Heart Disease", "Heart Disease"};

            for (int i = 0; i < 5; i++) {
                double[] sample = testSet.getSample(i);
                int actualLabel = testSet.getLabel(i);
                
                // Get probability of Class 1 (Heart Disease)
                double probHeartDisease = rf.predictProba(sample);
                int prediction = rf.predict(sample);
                
                // Calculate confidence for the specific prediction made
                double confidence = (prediction == 1) ? probHeartDisease : (1.0 - probHeartDisease);

                System.out.println("\nSample #" + (i + 1));
                System.out.println("  Prediction: " + classNames[prediction] + 
                                 String.format(" (%.1f%% confidence)", confidence * 100));
                System.out.println("  Actual:     " + classNames[actualLabel]);
                System.out.println("  Result:     " + (prediction == actualLabel ? "CORRECT" : "INCORRECT"));
            }

        } catch (IOException e) {
            System.err.println("Error loading dataset: " + e.getMessage());
            e.printStackTrace();
        } catch (Exception e) {
            System.err.println("Error during training: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
