package com.heartdisease.model;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Utility class for loading CSV data
 */
public class DataLoader {

    /**
     * Load dataset from CSV file
     * Assumes last column is the target variable
     */
    public static Dataset loadFromCSV(String filePath) throws IOException {
        try (Reader reader = new FileReader(filePath);
             CSVParser csvParser = new CSVParser(reader, CSVFormat.DEFAULT.withFirstRecordAsHeader())) {

            List<String> headers = new ArrayList<>(csvParser.getHeaderNames());
            int numFeatures = headers.size() - 1; // Last column is target

            String[] featureNames = headers.subList(0, numFeatures).toArray(new String[0]);

            List<double[]> featuresList = new ArrayList<>();
            List<Integer> labelsList = new ArrayList<>();

            for (CSVRecord record : csvParser) {
                double[] features = new double[numFeatures];
                for (int i = 0; i < numFeatures; i++) {
                    features[i] = Double.parseDouble(record.get(i));
                }
                featuresList.add(features);

                // Last column is target
                int label = (int) Double.parseDouble(record.get(numFeatures));
                labelsList.add(label);
            }

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
