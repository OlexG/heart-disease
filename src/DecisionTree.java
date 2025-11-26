import java.util.*;

/**
 * Decision Tree implementation similar to Lab 7
 */
public class DecisionTree {
    private Node root;
    private final int maxDepth;
    private final int minSamplesSplit;
    private final int maxFeatures;
    private final Random random;
    private TreeMatrix matrix;

    public DecisionTree(int maxDepth, int minSamplesSplit, int maxFeatures, Random random) {
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.maxFeatures = maxFeatures;
        this.random = random;
    }

    public void fit(Dataset dataset) {
        this.matrix = new TreeMatrix(dataset);
        
        // Initialize rows (0 to N-1)
        ArrayList<Integer> rows = new ArrayList<>();
        for (int i = 0; i < dataset.getNumSamples(); i++) {
            rows.add(i);
        }

        // Initialize available attributes
        ArrayList<Integer> attributes = new ArrayList<>();
        for (int i = 0; i < dataset.getNumFeatures(); i++) {
            attributes.add(i);
        }

        root = buildTree(rows, attributes, 0);
    }

    public int predict(double[] features) {
        return predictNode(root, features);
    }

    public String toDotString(String[] featureNames) {
        StringBuilder sb = new StringBuilder();
        sb.append("digraph DecisionTree {\n");
        sb.append("  node [shape=box, fontname=\"Arial\"];\n");
        printDot(root, sb, new int[]{0}, featureNames);
        sb.append("}\n");
        return sb.toString();
    }

    private int printDot(Node node, StringBuilder sb, int[] idCounter, String[] featureNames) {
        if (node == null) return -1;

        int myId = idCounter[0]++;
        
        if (node.isLeaf()) {
            String className = node.predictedClass == 0 ? "No Disease" : "Disease";
            String color = node.predictedClass == 0 ? "#e5f5e0" : "#ffe6e6"; // Green-ish vs Red-ish
            
            // Add count to label
            String label = className + "\\n(n=" + node.sampleCount + ")";
            
            sb.append("  ").append(myId).append(" [label=\"").append(label)
              .append("\", style=filled, fillcolor=\"").append(color).append("\"];\n");
        } else {
            String featureName = (featureNames != null && node.featureIndex < featureNames.length) 
                    ? featureNames[node.featureIndex] 
                    : "Feat " + node.featureIndex;
            
            // Add count to label
            String splitCondition;
            if (node.categoricalValues != null) {
                // Categorical split: show category membership
                ArrayList<Integer> sortedCats = new ArrayList<>(node.categoricalValues);
                Collections.sort(sortedCats);
                splitCondition = "in {" + sortedCats.toString().replaceAll("[\\[\\] ]", "") + "}";
            } else {
                // Numerical split: show threshold
                splitCondition = "<= " + String.format("%.3f", node.threshold);
            }
            String label = featureName + "\\n" + splitCondition + "\\n(n=" + node.sampleCount + ")";
            
            sb.append("  ").append(myId).append(" [label=\"").append(label).append("\"];\n");
            
            int leftId = printDot(node.left, sb, idCounter, featureNames);
            if (leftId != -1) {
                sb.append("  ").append(myId).append(" -> ").append(leftId)
                  .append(" [label=\"True\", fontsize=10];\n");
            }
            
            int rightId = printDot(node.right, sb, idCounter, featureNames);
            if (rightId != -1) {
                sb.append("  ").append(myId).append(" -> ").append(rightId)
                  .append(" [label=\"False\", fontsize=10];\n");
            }
        }
        return myId;
    }

    private Node buildTree(ArrayList<Integer> rows, ArrayList<Integer> availableAttributes, int depth) {
        double entropy = matrix.getEntropy(rows);
        // Base case: Stop if attributes exhausted or entropy is low (pure enough)
        if (availableAttributes.isEmpty() || entropy < 0.01 || depth >= maxDepth || rows.size() < minSamplesSplit) {
            return new Node(matrix.getMostCommonValue(rows), rows.size());
        }

        // Feature Selection for Random Forest (Random subset of available attributes)
        List<Integer> candidateAttributes = new ArrayList<>(availableAttributes);
        if (maxFeatures < candidateAttributes.size()) {
            Collections.shuffle(candidateAttributes, random);
            candidateAttributes = candidateAttributes.subList(0, maxFeatures);
        }

        double bestIGR = -1.0;
        int bestAttribute = -1;

        // Find best attribute to split on
        for (int attribute : candidateAttributes) {
            double igr = matrix.computeIGR(attribute, rows, entropy);
            if (igr > bestIGR) {
                bestIGR = igr;
                bestAttribute = attribute;
            }
        }

        // If no gain or invalid split, stop
        if (bestIGR <= 0.0) {
            return new Node(matrix.getMostCommonValue(rows), rows.size());
        }

        // Perform Split
        Map<Integer, ArrayList<Integer>> partitions = matrix.split(bestAttribute, rows);
        ArrayList<Integer> leftRows = partitions.get(0);
        ArrayList<Integer> rightRows = partitions.get(1);

        // Pre-prune: if splitting results in empty children
        if (leftRows.isEmpty() || rightRows.isEmpty()) {
            return new Node(matrix.getMostCommonValue(rows), rows.size());
        }

        // Create child nodes
        // "exhausted" means we remove the used attribute for the children
        ArrayList<Integer> remainingAttributes = new ArrayList<>(availableAttributes);
        remainingAttributes.remove(Integer.valueOf(bestAttribute));

        Node leftChild = buildTree(leftRows, remainingAttributes, depth + 1);
        Node rightChild = buildTree(rightRows, remainingAttributes, depth + 1);

        // Optimization: If both children are leaves and predict the same class, 
        // collapse this node into a leaf.
        if (leftChild.isLeaf() && rightChild.isLeaf() && leftChild.predictedClass == rightChild.predictedClass) {
            return new Node(leftChild.predictedClass, rows.size());
        }

        // Check if this is a categorical split
        Set<Integer> categoricalSplit = null;
        double threshold = 0.0;
        if (matrix.isCategorical(bestAttribute)) {
            categoricalSplit = matrix.getCategoricalSplit(bestAttribute);
        } else {
            threshold = matrix.getSplitThreshold(bestAttribute);
        }

        return new Node(bestAttribute, threshold, categoricalSplit, leftChild, rightChild, rows.size());
    }

    private int predictNode(Node node, double[] features) {
        if (node.isLeaf()) {
            return node.predictedClass;
        }
        
        if (node.categoricalValues != null) {
            // Categorical split: check if feature value is in categoricalValues set
            int categoryValue = (int) features[node.featureIndex];
            if (node.categoricalValues.contains(categoryValue)) {
                return predictNode(node.left, features);
            } else {
                return predictNode(node.right, features);
            }
        } else {
            // Numerical split: use threshold
            if (features[node.featureIndex] <= node.threshold) {
                return predictNode(node.left, features);
            } else {
                return predictNode(node.right, features);
            }
        }
    }

    /**
     * Internal Node class
     */
    private static class Node {
        int featureIndex;
        double threshold;
        Set<Integer> categoricalValues; // null for numerical splits, non-null for categorical
        Node left;
        Node right;
        int predictedClass;
        int sampleCount;

        Node(int predictedClass, int sampleCount) {
            this.predictedClass = predictedClass;
            this.sampleCount = sampleCount;
        }

        Node(int featureIndex, double threshold, Set<Integer> categoricalValues, Node left, Node right, int sampleCount) {
            this.featureIndex = featureIndex;
            this.threshold = threshold;
            this.categoricalValues = categoricalValues;
            this.left = left;
            this.right = right;
            this.predictedClass = -1;
            this.sampleCount = sampleCount;
        }

        boolean isLeaf() {
            return left == null && right == null;
        }
    }
}