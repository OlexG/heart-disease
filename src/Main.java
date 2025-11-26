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
            String[] categoricalNames = {
                "gender", "exercise_habits", "smoking", "family_heart_disease", 
                "diabetes", "high_blood_pressure", "low_hdl_cholesterol", 
                "high_ldl_cholesterol", "alcohol_consumption", "stress_level", 
                "sugar_consumption"
            };
            String dataPath = "python-analysis/artifacts/features_model_ready.csv";
//            Dataset dataset = DataLoader.loadFromCSV(dataPath);
            Dataset dataset = DataLoader.loadFromCSV(dataPath, categoricalNames);

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

            // Define hyperparameter grid (industry standard ranges)
            System.out.println("\n=== Hyperparameter Grid Definition ===");
            int[] numTreesValues = {50, 100, 200, 300};
            Integer[] maxDepthValues = {5, 10, 15, 20, null}; // null = unlimited
            int[] minSamplesSplitValues = {2, 5, 10, 20};
            int numFeatures = dataset.getNumFeatures();
            int[] maxFeaturesValues = {
                (int) Math.sqrt(numFeatures),
                (int) (Math.log(numFeatures) / Math.log(2)),
                numFeatures / 3,
                numFeatures / 2
            };
            
            HyperparameterTuner.ParameterGrid grid = new HyperparameterTuner.ParameterGrid(
                numTreesValues, maxDepthValues, minSamplesSplitValues, maxFeaturesValues
            );
            
            System.out.println("  numTrees: " + java.util.Arrays.toString(numTreesValues));
            System.out.println("  maxDepth: " + java.util.Arrays.toString(maxDepthValues));
            System.out.println("  minSamplesSplit: " + java.util.Arrays.toString(minSamplesSplitValues));
            System.out.println("  maxFeatures: " + java.util.Arrays.toString(maxFeaturesValues));
            System.out.println("  Total combinations: " + grid.getTotalCombinations());

            // Perform hyperparameter tuning
            HyperparameterTuner tuner = new HyperparameterTuner(5, 42L, HyperparameterTuner.Metric.ACCURACY, true);
            HyperparameterTuner.TuningResult bestParams = tuner.tune(trainSet, grid);
            
            System.out.println("\n" + bestParams);

            // Train final model with best hyperparameters
            System.out.println("\n=== Training Final Model with Best Hyperparameters ===");
            int bestNumTrees = bestParams.getNumTrees();
            Integer bestMaxDepth = bestParams.getMaxDepth();
            int bestMinSamplesSplit = bestParams.getMinSamplesSplit();
            int bestMaxFeatures = bestParams.getMaxFeatures();

            RandomForest rf = new RandomForest(
                bestNumTrees,
                bestMaxDepth == null ? Integer.MAX_VALUE : bestMaxDepth,
                bestMinSamplesSplit,
                bestMaxFeatures,
                42L
            );

            System.out.println("  Number of trees: " + bestNumTrees);
            System.out.println("  Max depth: " + (bestMaxDepth == null ? "unlimited" : bestMaxDepth));
            System.out.println("  Min samples split: " + bestMinSamplesSplit);
            System.out.println("  Max features: " + bestMaxFeatures);

            long startTime = System.currentTimeMillis();
            rf.fit(trainSet);
            long endTime = System.currentTimeMillis();

            System.out.println("  Training completed in " + (endTime - startTime) + "ms");

            // Comprehensive evaluation on test set
            System.out.println("\n=== Comprehensive Model Evaluation on Test Set ===");
            int[] testPredictions = rf.predict(testSet.getFeatures());
            int[] testActual = new int[testSet.getNumSamples()];
            for (int i = 0; i < testSet.getNumSamples(); i++) {
                testActual[i] = testSet.getLabel(i);
            }
            
            double testAccuracy = Metrics.accuracy(testPredictions, testActual);
            double testF1 = Metrics.f1Score(testPredictions, testActual);
            double testPrecision = Metrics.precision(testPredictions, testActual);
            double testRecall = Metrics.recall(testPredictions, testActual);
            
            System.out.println("Accuracy:  " + String.format("%.4f", testAccuracy));
            System.out.println("F1 Score: " + String.format("%.4f", testF1));
            System.out.println("Precision:" + String.format("%.4f", testPrecision));
            System.out.println("Recall:   " + String.format("%.4f", testRecall));
            System.out.println();
            System.out.println(Metrics.confusionMatrixString(testPredictions, testActual));
            
            // Also evaluate on training set for comparison
            System.out.println("\n=== Training Set Evaluation (for comparison) ===");
            int[] trainPredictions = rf.predict(trainSet.getFeatures());
            int[] trainActual = new int[trainSet.getNumSamples()];
            for (int i = 0; i < trainSet.getNumSamples(); i++) {
                trainActual[i] = trainSet.getLabel(i);
            }
            
            double trainAccuracy = Metrics.accuracy(trainPredictions, trainActual);
            double trainF1 = Metrics.f1Score(trainPredictions, trainActual);
            double trainPrecision = Metrics.precision(trainPredictions, trainActual);
            double trainRecall = Metrics.recall(trainPredictions, trainActual);
            
            System.out.println("Accuracy:  " + String.format("%.4f", trainAccuracy));
            System.out.println("F1 Score: " + String.format("%.4f", trainF1));
            System.out.println("Precision:" + String.format("%.4f", trainPrecision));
            System.out.println("Recall:   " + String.format("%.4f", trainRecall));

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
                double probHeartDisease = rf.predictProbability(sample);
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
