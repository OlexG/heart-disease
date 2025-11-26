import java.io.*;
import java.util.*;

/**
 * Utility class for loading CSV data
 */
public class DataLoader {

    /**
     * Load dataset from CSV file with categorical feature specification
     * @param filePath Path to CSV file
     * @param categoricalFeatureNames Array of feature names that are categorical
     * @return Dataset with categorical attributes specified
     */
    public static Dataset loadFromCSV(String filePath, String[] categoricalFeatureNames) throws IOException {
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

            // Map categorical feature names to indices
            Set<Integer> categoricalIndices = new HashSet<>();
            Set<String> categoricalSet = new HashSet<>();
            for (String name : categoricalFeatureNames) {
                categoricalSet.add(name.trim());
            }
            
            for (int i = 0; i < numFeatures; i++) {
                if (categoricalSet.contains(featureNames[i])) {
                    categoricalIndices.add(i);
                }
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

            return new Dataset(featuresArray, labelsArray, featureNames, categoricalIndices);
        }
    }

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
     * Split dataset into K folds for cross-validation
     * @param dataset The dataset to split
     * @param k Number of folds
     * @param seed Random seed for shuffling
     * @return List of K DataSplit objects, where each split has one fold as test and the rest as train
     */
    public static List<DataSplit> kFoldSplit(Dataset dataset, int k, long seed) {
        if (k < 2) {
            throw new IllegalArgumentException("Number of folds (k) must be at least 2");
        }
        
        int numSamples = dataset.getNumSamples();
        if (k > numSamples) {
            throw new IllegalArgumentException("Number of folds (k) cannot be greater than number of samples");
        }
        
        // Create shuffled indices
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < numSamples; i++) {
            indices.add(i);
        }
        Collections.shuffle(indices, new java.util.Random(seed));
        
        // Calculate fold size
        int foldSize = numSamples / k;
        int remainder = numSamples % k;
        
        List<DataSplit> splits = new ArrayList<>();
        
        int currentIndex = 0;
        for (int fold = 0; fold < k; fold++) {
            // Calculate size of this fold (distribute remainder across first few folds)
            int currentFoldSize = foldSize + (fold < remainder ? 1 : 0);
            
            // Extract validation fold indices
            int[] validationIndices = new int[currentFoldSize];
            for (int i = 0; i < currentFoldSize; i++) {
                validationIndices[i] = indices.get(currentIndex + i);
            }
            
            // Extract training indices (all indices not in validation fold)
            int[] trainIndices = new int[numSamples - currentFoldSize];
            int trainIdx = 0;
            for (int i = 0; i < numSamples; i++) {
                boolean isInValidation = false;
                for (int j = 0; j < currentFoldSize; j++) {
                    if (indices.get(i) == validationIndices[j]) {
                        isInValidation = true;
                        break;
                    }
                }
                if (!isInValidation) {
                    trainIndices[trainIdx++] = indices.get(i);
                }
            }
            
            Dataset trainSet = dataset.subset(trainIndices);
            Dataset validationSet = dataset.subset(validationIndices);
            
            splits.add(new DataSplit(trainSet, validationSet));
            currentIndex += currentFoldSize;
        }
        
        return splits;
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