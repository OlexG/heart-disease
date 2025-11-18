package com.heartdisease.model;

import java.util.*;

/**
 * Decision Tree implementation for binary classification
 */
public class DecisionTree {
    private Node root;
    private final int maxDepth;
    private final int minSamplesSplit;
    private final int maxFeatures;
    private final Random random;

    public DecisionTree(int maxDepth, int minSamplesSplit, int maxFeatures, Random random) {
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.maxFeatures = maxFeatures;
        this.random = random;
    }

    /**
     * Train the decision tree on the given dataset
     */
    public void fit(Dataset dataset) {
        root = buildTree(dataset, 0);
    }

    /**
     * Predict the class for a single sample
     */
    public int predict(double[] features) {
        return predictNode(root, features);
    }

    private Node buildTree(Dataset dataset, int depth) {
        int numSamples = dataset.getNumSamples();
        int[] labels = dataset.getLabels();

        // Check stopping criteria
        if (depth >= maxDepth || numSamples < minSamplesSplit || isPure(labels)) {
            return new Node(majorityClass(labels));
        }

        // Find best split
        Split bestSplit = findBestSplit(dataset);

        if (bestSplit == null) {
            return new Node(majorityClass(labels));
        }

        // Create child nodes
        Node leftChild = buildTree(dataset.subset(bestSplit.leftIndices), depth + 1);
        Node rightChild = buildTree(dataset.subset(bestSplit.rightIndices), depth + 1);

        return new Node(bestSplit.featureIndex, bestSplit.threshold, leftChild, rightChild);
    }

    private Split findBestSplit(Dataset dataset) {
        int numFeatures = dataset.getNumFeatures();
        int numSamples = dataset.getNumSamples();

        // Select random subset of features
        int[] featureIndices = selectRandomFeatures(numFeatures);

        Split bestSplit = null;
        double bestGini = Double.MAX_VALUE;

        for (int featureIndex : featureIndices) {
            // Get all unique values for this feature
            Set<Double> uniqueValues = new HashSet<>();
            for (int i = 0; i < numSamples; i++) {
                uniqueValues.add(dataset.getSample(i)[featureIndex]);
            }

            // Try each unique value as a threshold
            for (double threshold : uniqueValues) {
                Split split = createSplit(dataset, featureIndex, threshold);

                if (split.leftIndices.length == 0 || split.rightIndices.length == 0) {
                    continue;
                }

                double gini = calculateGiniImpurity(dataset, split);

                if (gini < bestGini) {
                    bestGini = gini;
                    bestSplit = split;
                }
            }
        }

        return bestSplit;
    }

    private Split createSplit(Dataset dataset, int featureIndex, double threshold) {
        List<Integer> leftList = new ArrayList<>();
        List<Integer> rightList = new ArrayList<>();

        for (int i = 0; i < dataset.getNumSamples(); i++) {
            if (dataset.getSample(i)[featureIndex] <= threshold) {
                leftList.add(i);
            } else {
                rightList.add(i);
            }
        }

        int[] leftIndices = leftList.stream().mapToInt(Integer::intValue).toArray();
        int[] rightIndices = rightList.stream().mapToInt(Integer::intValue).toArray();

        return new Split(featureIndex, threshold, leftIndices, rightIndices);
    }

    private double calculateGiniImpurity(Dataset dataset, Split split) {
        int totalSamples = dataset.getNumSamples();

        double leftGini = gini(dataset, split.leftIndices);
        double rightGini = gini(dataset, split.rightIndices);

        double leftWeight = (double) split.leftIndices.length / totalSamples;
        double rightWeight = (double) split.rightIndices.length / totalSamples;

        return leftWeight * leftGini + rightWeight * rightGini;
    }

    private double gini(Dataset dataset, int[] indices) {
        if (indices.length == 0) {
            return 0.0;
        }

        int[] classCounts = new int[2]; // Binary classification
        for (int index : indices) {
            classCounts[dataset.getLabel(index)]++;
        }

        double impurity = 1.0;
        for (int count : classCounts) {
            double prob = (double) count / indices.length;
            impurity -= prob * prob;
        }

        return impurity;
    }

    private int[] selectRandomFeatures(int numFeatures) {
        int numSelectedFeatures = Math.min(maxFeatures, numFeatures);
        List<Integer> allFeatures = new ArrayList<>();
        for (int i = 0; i < numFeatures; i++) {
            allFeatures.add(i);
        }
        Collections.shuffle(allFeatures, random);
        return allFeatures.stream().limit(numSelectedFeatures).mapToInt(Integer::intValue).toArray();
    }

    private boolean isPure(int[] labels) {
        if (labels.length == 0) {
            return true;
        }
        int firstLabel = labels[0];
        for (int label : labels) {
            if (label != firstLabel) {
                return false;
            }
        }
        return true;
    }

    private int majorityClass(int[] labels) {
        int[] classCounts = new int[2]; // Binary classification
        for (int label : labels) {
            classCounts[label]++;
        }
        return classCounts[0] > classCounts[1] ? 0 : 1;
    }

    private int predictNode(Node node, double[] features) {
        if (node.isLeaf()) {
            return node.predictedClass;
        }

        if (features[node.featureIndex] <= node.threshold) {
            return predictNode(node.left, features);
        } else {
            return predictNode(node.right, features);
        }
    }

    /**
     * Internal node class
     */
    private static class Node {
        int featureIndex;
        double threshold;
        Node left;
        Node right;
        int predictedClass;

        // Leaf node constructor
        Node(int predictedClass) {
            this.predictedClass = predictedClass;
        }

        // Internal node constructor
        Node(int featureIndex, double threshold, Node left, Node right) {
            this.featureIndex = featureIndex;
            this.threshold = threshold;
            this.left = left;
            this.right = right;
            this.predictedClass = -1;
        }

        boolean isLeaf() {
            return left == null && right == null;
        }
    }

    /**
     * Split information
     */
    private static class Split {
        int featureIndex;
        double threshold;
        int[] leftIndices;
        int[] rightIndices;

        Split(int featureIndex, double threshold, int[] leftIndices, int[] rightIndices) {
            this.featureIndex = featureIndex;
            this.threshold = threshold;
            this.leftIndices = leftIndices;
            this.rightIndices = rightIndices;
        }
    }
}
