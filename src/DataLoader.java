import java.io.*;
import java.util.*;

/**
 * Utility class for loading CSV data
 */
public class DataLoader {

    /**
     * Load dataset from CSV file
     */
    public static Dataset loadFromCSV(String filePath) throws IOException {
        List<double[]> featuresList = new ArrayList<>();
        List<Integer> labelsList = new ArrayList<>();
        List<String> headers = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            // 1. Read Header
            String line = br.readLine();
            if (line != null) {
                String[] parts = line.split(",");
                for (String p : parts) {
                    headers.add(p.trim());
                }
            }

            int numFeatures = headers.size() - 1; // Last column is target
            String[] featureNames = new String[numFeatures];
            for(int i = 0; i < numFeatures; i++) {
                featureNames[i] = headers.get(i);
            }

            // 2. Read Data Rows
            while ((line = br.readLine()) != null) {
                if (line.trim().isEmpty()) continue;
                
                String[] parts = line.split(",");
                double[] features = new double[numFeatures];
                
                // Parse features
                for (int i = 0; i < numFeatures; i++) {
                    features[i] = Double.parseDouble(parts[i].trim());
                }
                featuresList.add(features);

                // Parse label (last column)
                int label = (int) Double.parseDouble(parts[numFeatures].trim());
                labelsList.add(label);
            }

            // Convert Lists to Arrays for Dataset
            double[][] featuresArray = featuresList.toArray(new double[0][]);
            int[] labelsArray = labelsList.stream().mapToInt(Integer::intValue).toArray();

            return new Dataset(featuresArray, labelsArray, featureNames);
        }
    }

    /**
     * Split dataset into train and test sets
     */
    public static DataSplit trainTestSplit(Dataset dataset, double testSize, long seed) {
        int numSamples = dataset.getNumSamples();
        int testSamples = (int) (numSamples * testSize);
        int trainSamples = numSamples - testSamples;

        // Create shuffled indices
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < numSamples; i++) {
            indices.add(i);
        }
        Collections.shuffle(indices, new java.util.Random(seed));

        // Split indices
        int[] trainIndices = new int[trainSamples];
        int[] testIndices = new int[testSamples];

        for (int i = 0; i < trainSamples; i++) {
            trainIndices[i] = indices.get(i);
        }
        for (int i = 0; i < testSamples; i++) {
            testIndices[i] = indices.get(trainSamples + i);
        }

        Dataset trainSet = dataset.subset(trainIndices);
        Dataset testSet = dataset.subset(testIndices);

        return new DataSplit(trainSet, testSet);
    }

    /**
     * Helper class to hold train/test split
     */
    public static class DataSplit {
        private final Dataset trainSet;
        private final Dataset testSet;

        public DataSplit(Dataset trainSet, Dataset testSet) {
            this.trainSet = trainSet;
            this.testSet = testSet;
        }

        public Dataset getTrainSet() {
            return trainSet;
        }

        public Dataset getTestSet() {
            return testSet;
        }
    }
}