import java.io.IOException;
import java.nio.file.Path;

/**
 * Main class to train and evaluate the Random Forest model
 */
public class Main {
    public static void main(String[] args) {
        ProcessLogger logger = new ProcessLogger();

        int treeIndexToVisualize = 0;
        if (args.length > 0) {
            try {
                treeIndexToVisualize = Integer.parseInt(args[0]);
            } catch (NumberFormatException e) {
                logger.error("Invalid argument for tree index. Using default: 0");
            }
        }

        String dataPath = "python-analysis/artifacts/features_model_ready_balanced.csv";

        try {
            RunOutputs runOutputs = new RunOutputs();
            logger.info("Saving run artifacts to: " + runOutputs.getRunDir().toAbsolutePath());

            logger.info("Loading dataset...");
            Dataset dataset = DataLoader.loadFromCSV(dataPath);

            logger.info("Dataset loaded:");
            logger.info("  Samples: " + dataset.getNumSamples());
            logger.info("  Features: " + dataset.getNumFeatures());

            logger.info("\nSplitting dataset (80% train, 20% test)...");
            DataLoader.DataSplit split = DataLoader.trainTestSplit(dataset, 0.2, 42L);
            Dataset trainSet = split.getTrainSet();
            Dataset testSet = split.getTestSet();
            logger.info("  Train samples: " + trainSet.getNumSamples());
            logger.info("  Test samples: " + testSet.getNumSamples());

            logger.info("\n=== Hyperparameter Grid Definition ===");
            int[] numTreesValues = {50, 100, 200, 300};
            Integer[] maxDepthValues = {5, 10, 15, 20, null};
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

            logger.info("  numTrees: " + java.util.Arrays.toString(numTreesValues));
            logger.info("  maxDepth: " + java.util.Arrays.toString(maxDepthValues));
            logger.info("  minSamplesSplit: " + java.util.Arrays.toString(minSamplesSplitValues));
            logger.info("  maxFeatures: " + java.util.Arrays.toString(maxFeaturesValues));
            logger.info("  Total combinations: " + grid.getTotalCombinations());

            HyperparameterTuner tuner = new HyperparameterTuner(5, 42L, HyperparameterTuner.Metric.ACCURACY, true);
            HyperparameterTuner.TuningResult bestParams = tuner.tune(trainSet, grid);
            logger.info("\n" + bestParams);

            logger.info("\n=== Training Final Model with Best Hyperparameters ===");
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

            logger.info("  Number of trees: " + bestNumTrees);
            logger.info("  Max depth: " + (bestMaxDepth == null ? "unlimited" : bestMaxDepth));
            logger.info("  Min samples split: " + bestMinSamplesSplit);
            logger.info("  Max features: " + bestMaxFeatures);

            long startTime = System.currentTimeMillis();
            rf.fit(trainSet);
            long endTime = System.currentTimeMillis();
            long trainingDurationMs = endTime - startTime;
            logger.info("  Training completed in " + trainingDurationMs + "ms");

            logger.info("\n=== Comprehensive Model Evaluation on Test Set ===");
            int[] testPredictions = rf.predict(testSet.getFeatures());
            int[] testActual = new int[testSet.getNumSamples()];
            for (int i = 0; i < testSet.getNumSamples(); i++) {
                testActual[i] = testSet.getLabel(i);
            }
            double[] testProbabilities = new double[testSet.getNumSamples()];
            for (int i = 0; i < testSet.getNumSamples(); i++) {
                testProbabilities[i] = rf.predictProbability(testSet.getSample(i));
            }

            double testAccuracy = Metrics.accuracy(testPredictions, testActual);
            double testF1 = Metrics.f1Score(testPredictions, testActual);
            double testPrecision = Metrics.precision(testPredictions, testActual);
            double testRecall = Metrics.recall(testPredictions, testActual);
            logger.info("Accuracy:  " + String.format("%.4f", testAccuracy));
            logger.info("F1 Score: " + String.format("%.4f", testF1));
            logger.info("Precision:" + String.format("%.4f", testPrecision));
            logger.info("Recall:   " + String.format("%.4f", testRecall));
            logger.info("");
            logger.info(Metrics.confusionMatrixString(testPredictions, testActual));

            logger.info("\n=== Training Set Evaluation (for comparison) ===");
            int[] trainPredictions = rf.predict(trainSet.getFeatures());
            int[] trainActual = new int[trainSet.getNumSamples()];
            for (int i = 0; i < trainSet.getNumSamples(); i++) {
                trainActual[i] = trainSet.getLabel(i);
            }

            double trainAccuracy = Metrics.accuracy(trainPredictions, trainActual);
            double trainF1 = Metrics.f1Score(trainPredictions, trainActual);
            double trainPrecision = Metrics.precision(trainPredictions, trainActual);
            double trainRecall = Metrics.recall(trainPredictions, trainActual);
            logger.info("Accuracy:  " + String.format("%.4f", trainAccuracy));
            logger.info("F1 Score: " + String.format("%.4f", trainF1));
            logger.info("Precision:" + String.format("%.4f", trainPrecision));
            logger.info("Recall:   " + String.format("%.4f", trainRecall));

            Path treeVizPath = null;
            if (rf.getNumTrees() > 0) {
                if (treeIndexToVisualize >= 0 && treeIndexToVisualize < rf.getNumTrees()) {
                    logger.info("\nSaving visualization of tree " + treeIndexToVisualize + "...");
                    String dotContent = rf.getTrees().get(treeIndexToVisualize).toDotString(dataset.getFeatureNames());
                    treeVizPath = runOutputs.writeTreeVisualization(treeIndexToVisualize, dotContent);
                    logger.info("  Saved to: " + treeVizPath.toAbsolutePath());
                } else {
                    logger.error("\nTree index " + treeIndexToVisualize + " is out of bounds (0-" + (rf.getNumTrees() - 1) + "). Skipping visualization.");
                }
            }

            logger.info("\n=== Detailed Predictions on First 5 Test Samples ===");
            String[] classNames = {"No Heart Disease", "Heart Disease"};
            int samplesToDisplay = Math.min(5, testSet.getNumSamples());
            for (int i = 0; i < samplesToDisplay; i++) {
                double probHeartDisease = testProbabilities[i];
                int prediction = testPredictions[i];
                int actualLabel = testActual[i];
                double confidence = (prediction == 1) ? probHeartDisease : (1.0 - probHeartDisease);

                logger.info("\nSample #" + (i + 1));
                logger.info("  Prediction: " + classNames[prediction] +
                        String.format(" (%.1f%% confidence)", confidence * 100));
                logger.info("  Actual:     " + classNames[actualLabel]);
                logger.info("  Result:     " + (prediction == actualLabel ? "CORRECT" : "INCORRECT"));
            }

            runOutputs.writePredictionsCsv(testPredictions, testActual, testProbabilities);
            runOutputs.writeProcessLog(logger.getEntries());

            int[] testConfusion = Metrics.confusionMatrixCounts(testPredictions, testActual);
            RunOutputs.SummaryData summary = new RunOutputs.SummaryData(
                dataPath,
                dataset.getNumFeatures(),
                trainSet.getNumSamples(),
                testSet.getNumSamples(),
                bestNumTrees,
                bestMaxDepth,
                bestMinSamplesSplit,
                bestMaxFeatures,
                trainingDurationMs,
                trainAccuracy,
                trainPrecision,
                trainRecall,
                trainF1,
                testAccuracy,
                testPrecision,
                testRecall,
                testF1,
                testConfusion
            );
            runOutputs.writeSummary(summary);

            logger.info("\nArtifacts available in: " + runOutputs.getRunDir().toAbsolutePath());
            if (treeVizPath == null) {
                logger.info("No tree visualization was generated this run.");
            }
        } catch (IOException e) {
            logger.error("Error loading dataset or writing artifacts: " + e.getMessage());
            e.printStackTrace();
        } catch (Exception e) {
            logger.error("Error during training: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
