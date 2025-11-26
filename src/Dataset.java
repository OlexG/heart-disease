/**
 * Represents a dataset with features and labels for training/testing
 */
public class Dataset {
    private final double[][] features;
    private final int[] labels;
    private final String[] featureNames;

    public Dataset(double[][] features, int[] labels, String[] featureNames) {
        this.features = features;
        this.labels = labels;
        this.featureNames = featureNames;
    }

    public double[][] getFeatures() {
        return features;
    }

    public int[] getLabels() {
        return labels;
    }

    public String[] getFeatureNames() {
        return featureNames;
    }

    public int getNumSamples() {
        return features.length;
    }

    public int getNumFeatures() {
        return features.length > 0 ? features[0].length : 0;
    }

    /**
     * Get a subset of the dataset by indices
     */
    public Dataset subset(int[] indices) {
        double[][] subsetFeatures = new double[indices.length][];
        int[] subsetLabels = new int[indices.length];

        for (int i = 0; i < indices.length; i++) {
            subsetFeatures[i] = features[indices[i]];
            subsetLabels[i] = labels[indices[i]];
        }

        return new Dataset(subsetFeatures, subsetLabels, featureNames);
    }

    /**
     * Get a single sample
     */
    public double[] getSample(int index) {
        return features[index];
    }

    public int getLabel(int index) {
        return labels[index];
    }
}
