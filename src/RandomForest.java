import java.util.*;

/**
 * Random Forest classifier for binary classification on numeric data
 */
public class RandomForest {
    private final int numTrees;
    private final int maxDepth;
    private final int minSamplesSplit;
    private final int maxFeatures;
    private final Random random;
    private final List<DecisionTree> trees;

    public RandomForest(int numTrees, int maxDepth, int minSamplesSplit, int maxFeatures, long seed) {
        this.numTrees = numTrees;
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.maxFeatures = maxFeatures;
        this.random = new Random(seed);
        this.trees = new ArrayList<>();
    }

    public RandomForest(int numTrees, int maxDepth, int minSamplesSplit, int maxFeatures) {
        this(numTrees, maxDepth, minSamplesSplit, maxFeatures, 42L);
    }

    /**
     * Train the random forest on the given dataset
     */
    public void fit(Dataset dataset) {
        trees.clear();

        for (int i = 0; i < numTrees; i++) {
            // Create bootstrap sample
            Dataset bootstrapSample = createBootstrapSample(dataset);

            // Train decision tree
            DecisionTree tree = new DecisionTree(maxDepth, minSamplesSplit, maxFeatures, random);
            tree.fit(bootstrapSample);

            trees.add(tree);
        }
    }

    /**
     * Predict the class for a single sample
     */
    public int predict(double[] features) {
        int[] votes = new int[2]; // Binary classification

        for (DecisionTree tree : trees) {
            int prediction = tree.predict(features);
            votes[prediction]++;
        }

        return votes[0] > votes[1] ? 0 : 1;
    }

    /**
     * Predict classes for multiple samples
     */
    public int[] predict(double[][] features) {
        int[] predictions = new int[features.length];
        for (int i = 0; i < features.length; i++) {
            predictions[i] = predict(features[i]);
        }
        return predictions;
    }

    /**
     * Predict probability for a single sample
     */
    public double predictProba(double[] features) {
        int positiveVotes = 0;

        for (DecisionTree tree : trees) {
            if (tree.predict(features) == 1) {
                positiveVotes++;
            }
        }

        return (double) positiveVotes / trees.size();
    }

    /**
     * Calculate accuracy on a dataset
     */
    public double score(Dataset dataset) {
        int correct = 0;
        for (int i = 0; i < dataset.getNumSamples(); i++) {
            int prediction = predict(dataset.getSample(i));
            if (prediction == dataset.getLabel(i)) {
                correct++;
            }
        }
        return (double) correct / dataset.getNumSamples();
    }

    /**
     * Create a bootstrap sample (sampling with replacement)
     */
    private Dataset createBootstrapSample(Dataset dataset) {
        int numSamples = dataset.getNumSamples();
        int[] indices = new int[numSamples];

        for (int i = 0; i < numSamples; i++) {
            indices[i] = random.nextInt(numSamples);
        }

        return dataset.subset(indices);
    }

    public int getNumTrees() {
        return numTrees;
    }

    public int getMaxDepth() {
        return maxDepth;
    }

    public int getMinSamplesSplit() {
        return minSamplesSplit;
    }

    public int getMaxFeatures() {
        return maxFeatures;
    }

    public List<DecisionTree> getTrees() {
        return trees;
    }
}
