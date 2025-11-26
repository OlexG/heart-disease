import java.util.*;

/**
 * Hyperparameter tuner using grid search with K-fold cross-validation
 */
public class HyperparameterTuner {
    
    /**
     * Enum for evaluation metrics
     */
    public enum Metric {
        ACCURACY,
        F1_SCORE,
        PRECISION,
        RECALL
    }
    
    /**
     * Result class to hold hyperparameter tuning results
     */
    public static class TuningResult {
        private final int numTrees;
        private final Integer maxDepth;
        private final int minSamplesSplit;
        private final int maxFeatures;
        private final double bestScore;
        private final double meanScore;
        private final double stdScore;
        private final Metric metric;
        
        public TuningResult(int numTrees, Integer maxDepth, int minSamplesSplit, 
                           int maxFeatures, double bestScore, double meanScore, 
                           double stdScore, Metric metric) {
            this.numTrees = numTrees;
            this.maxDepth = maxDepth;
            this.minSamplesSplit = minSamplesSplit;
            this.maxFeatures = maxFeatures;
            this.bestScore = bestScore;
            this.meanScore = meanScore;
            this.stdScore = stdScore;
            this.metric = metric;
        }
        
        public int getNumTrees() { return numTrees; }
        public Integer getMaxDepth() { return maxDepth; }
        public int getMinSamplesSplit() { return minSamplesSplit; }
        public int getMaxFeatures() { return maxFeatures; }
        public double getBestScore() { return bestScore; }
        public double getMeanScore() { return meanScore; }
        public double getStdScore() { return stdScore; }
        public Metric getMetric() { return metric; }
        
        @Override
        public String toString() {
            return String.format(
                "Best Hyperparameters (optimizing for %s):\n" +
                "  numTrees: %d\n" +
                "  maxDepth: %s\n" +
                "  minSamplesSplit: %d\n" +
                "  maxFeatures: %d\n" +
                "  Best %s: %.4f (mean: %.4f, std: %.4f)",
                metric.name(),
                numTrees,
                maxDepth == null ? "unlimited" : String.valueOf(maxDepth),
                minSamplesSplit,
                maxFeatures,
                metric.name(),
                bestScore,
                meanScore,
                stdScore
            );
        }
    }
    
    /**
     * Parameter grid for hyperparameter search
     */
    public static class ParameterGrid {
        private final int[] numTreesValues;
        private final Integer[] maxDepthValues;
        private final int[] minSamplesSplitValues;
        private final int[] maxFeaturesValues;
        
        public ParameterGrid(int[] numTreesValues, Integer[] maxDepthValues,
                           int[] minSamplesSplitValues, int[] maxFeaturesValues) {
            this.numTreesValues = numTreesValues;
            this.maxDepthValues = maxDepthValues;
            this.minSamplesSplitValues = minSamplesSplitValues;
            this.maxFeaturesValues = maxFeaturesValues;
        }
        
        public int[] getNumTreesValues() { return numTreesValues; }
        public Integer[] getMaxDepthValues() { return maxDepthValues; }
        public int[] getMinSamplesSplitValues() { return minSamplesSplitValues; }
        public int[] getMaxFeaturesValues() { return maxFeaturesValues; }
        
        /**
         * Calculate total number of combinations
         */
        public int getTotalCombinations() {
            return numTreesValues.length * maxDepthValues.length * 
                   minSamplesSplitValues.length * maxFeaturesValues.length;
        }
    }
    
    private final int kFolds;
    private final long seed;
    private final Metric metric;
    private final boolean verbose;
    
    /**
     * Create a hyperparameter tuner
     * @param kFolds Number of folds for cross-validation (default: 5)
     * @param seed Random seed for reproducibility
     * @param metric Metric to optimize for (default: ACCURACY)
     * @param verbose Whether to print progress (default: true)
     */
    public HyperparameterTuner(int kFolds, long seed, Metric metric, boolean verbose) {
        this.kFolds = kFolds;
        this.seed = seed;
        this.metric = metric;
        this.verbose = verbose;
    }
    
    public HyperparameterTuner(int kFolds, long seed, Metric metric) {
        this(kFolds, seed, metric, true);
    }
    
    public HyperparameterTuner(int kFolds, long seed) {
        this(kFolds, seed, Metric.ACCURACY, true);
    }
    
    public HyperparameterTuner() {
        this(5, 42L, Metric.ACCURACY, true);
    }
    
    /**
     * Perform grid search with K-fold cross-validation
     * @param dataset Full dataset (will be split into folds)
     * @param grid Parameter grid to search
     * @return TuningResult with best hyperparameters
     */
    public TuningResult tune(Dataset dataset, ParameterGrid grid) {
        if (verbose) {
            System.out.println("\n=== Starting Hyperparameter Tuning ===");
            System.out.println("Metric: " + metric.name());
            System.out.println("K-folds: " + kFolds);
            System.out.println("Total combinations to test: " + grid.getTotalCombinations());
            System.out.println();
        }
        
        // Create K-fold splits
        List<DataLoader.DataSplit> folds = DataLoader.kFoldSplit(dataset, kFolds, seed);
        
        TuningResult bestResult = null;
        double bestScore = Double.NEGATIVE_INFINITY;
        int combinationCount = 0;
        int totalCombinations = grid.getTotalCombinations();
        
        // Grid search: iterate over all combinations
        for (int numTrees : grid.getNumTreesValues()) {
            for (Integer maxDepth : grid.getMaxDepthValues()) {
                for (int minSamplesSplit : grid.getMinSamplesSplitValues()) {
                    for (int maxFeatures : grid.getMaxFeaturesValues()) {
                        combinationCount++;
                        
                        if (verbose && combinationCount % 10 == 0) {
                            System.out.printf("Progress: %d/%d combinations tested (%.1f%%)\n",
                                combinationCount, totalCombinations,
                                100.0 * combinationCount / totalCombinations);
                        }
                        
                        // Evaluate this combination using K-fold CV
                        List<Double> foldScores = new ArrayList<>();
                        
                        for (DataLoader.DataSplit fold : folds) {
                            Dataset trainFold = fold.getTrainSet();
                            Dataset validationFold = fold.getTestSet();
                            
                            // Train model with current hyperparameters
                            RandomForest rf = new RandomForest(
                                numTrees, 
                                maxDepth == null ? Integer.MAX_VALUE : maxDepth,
                                minSamplesSplit, 
                                maxFeatures, 
                                seed
                            );
                            rf.fit(trainFold);
                            
                            // Evaluate on validation fold
                            double score = evaluateModel(rf, validationFold, metric);
                            foldScores.add(score);
                        }
                        
                        // Calculate mean and std across folds
                        double meanScore = foldScores.stream()
                            .mapToDouble(Double::doubleValue)
                            .average()
                            .orElse(0.0);
                        double stdScore = calculateStd(foldScores, meanScore);
                        
                        // Update best result if this is better
                        if (meanScore > bestScore) {
                            bestScore = meanScore;
                            bestResult = new TuningResult(
                                numTrees, maxDepth, minSamplesSplit, maxFeatures,
                                bestScore, meanScore, stdScore, metric
                            );
                            
                            if (verbose) {
                                System.out.printf("New best! %s=%.4f (std=%.4f) - numTrees=%d, maxDepth=%s, minSamplesSplit=%d, maxFeatures=%d\n",
                                    metric.name(), meanScore, stdScore,
                                    numTrees, maxDepth == null ? "unlimited" : maxDepth,
                                    minSamplesSplit, maxFeatures);
                            }
                        }
                    }
                }
            }
        }
        
        if (verbose) {
            System.out.println("\n=== Hyperparameter Tuning Complete ===");
            System.out.println(bestResult);
        }
        
        return bestResult;
    }
    
    /**
     * Evaluate model on dataset using specified metric
     */
    private double evaluateModel(RandomForest model, Dataset dataset, Metric metric) {
        int[] predictions = model.predict(dataset.getFeatures());
        int[] actual = new int[dataset.getNumSamples()];
        for (int i = 0; i < dataset.getNumSamples(); i++) {
            actual[i] = dataset.getLabel(i);
        }
        
        switch (metric) {
            case ACCURACY:
                return Metrics.accuracy(predictions, actual);
            case F1_SCORE:
                return Metrics.f1Score(predictions, actual);
            case PRECISION:
                return Metrics.precision(predictions, actual);
            case RECALL:
                return Metrics.recall(predictions, actual);
            default:
                return Metrics.accuracy(predictions, actual);
        }
    }
    
    /**
     * Calculate standard deviation
     */
    private double calculateStd(List<Double> values, double mean) {
        if (values.size() <= 1) {
            return 0.0;
        }
        
        double sumSquaredDiff = 0.0;
        for (double value : values) {
            double diff = value - mean;
            sumSquaredDiff += diff * diff;
        }
        
        return Math.sqrt(sumSquaredDiff / values.size());
    }
}

