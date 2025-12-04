import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class TreeMatrix {
    private static final double LAPLACE_ALPHA = 1.0; // Laplace smoothing parameter 
    private final Dataset dataset;
    // Cache to store the best threshold found during computeIGR for each attribute
    private final Map<Integer, Double> bestThresholds = new HashMap<>();

    public TreeMatrix(Dataset dataset) {
        this.dataset = dataset;
    }

    private double log2(double number) {
        if (number <= 0.0) {
            return 0.0;
        }
        return Math.log(number) / Math.log(2.0);
    }

    private double findEntropy(ArrayList<Integer> rows) {
        if (rows.isEmpty()) {
            return 0.0;
        }
        Map<Integer, Integer> classCounts = new HashMap<>();
        for (int rowIndex : rows) {
            int label = dataset.getLabel(rowIndex);
            classCounts.put(label, classCounts.getOrDefault(label, 0) + 1);
        }
        
        double entropy = 0.0;
        double total = (double) rows.size();
        int numClasses = classCounts.size();
        
        // Apply Laplace smoothing: p = (count + alpha) / (total + alpha * numClasses)
        double smoothedTotal = total + LAPLACE_ALPHA * numClasses;
        for (int count : classCounts.values()) {
            double p = (count + LAPLACE_ALPHA) / smoothedTotal;
            entropy -= p * log2(p);
        }
        return entropy;
    }

    public double computeIGR(int attribute, ArrayList<Integer> rows, double parentEntropy) {
        int numSamples = rows.size();
        if (numSamples <= 1) return 0.0;

        // Sort rows based on feature value to find best split
        ArrayList<Integer> sortedRows = new ArrayList<>(rows);
        sortedRows.sort((a, b) -> Double.compare(
            dataset.getSample(a)[attribute], 
            dataset.getSample(b)[attribute]
        ));

        double bestGainRatio = 0.0;
        double bestThreshold = 0.0;
        boolean foundSplit = false;

        // Calculate total class counts
        Map<Integer, Integer> rightCounts = new HashMap<>();
        for (int idx : rows) {
            int label = dataset.getLabel(idx);
            rightCounts.put(label, rightCounts.getOrDefault(label, 0) + 1);
        }
        Map<Integer, Integer> leftCounts = new HashMap<>();

        int leftSize = 0;
        int rightSize = numSamples;

        // Iterate through sorted samples
        for (int i = 0; i < numSamples - 1; i++) {
            int idx = sortedRows.get(i);
            int label = dataset.getLabel(idx);

            // Move sample from right to left
            rightCounts.put(label, rightCounts.get(label) - 1);
            rightSize--;
            leftCounts.put(label, leftCounts.getOrDefault(label, 0) + 1);
            leftSize++;

            double currentVal = dataset.getSample(idx)[attribute];
            double nextVal = dataset.getSample(sortedRows.get(i + 1))[attribute];

            if (currentVal == nextVal) continue;

            // Calculate metrics for this split
            double total = (double) numSamples;
            double leftEntropy = computeEntropyFromCounts(leftCounts, leftSize);
            double rightEntropy = computeEntropyFromCounts(rightCounts, rightSize);

            double leftWeight = leftSize / total;
            double rightWeight = rightSize / total;

            double weightedEntropy = leftWeight * leftEntropy + rightWeight * rightEntropy;
            double infoGain = parentEntropy - weightedEntropy;

            double splitInfo = 0.0;
            if (leftWeight > 0) splitInfo -= leftWeight * log2(leftWeight);
            if (rightWeight > 0) splitInfo -= rightWeight * log2(rightWeight);

            double gainRatio = (splitInfo == 0.0) ? 0.0 : infoGain / splitInfo;
            
            if (gainRatio > bestGainRatio) {
                bestGainRatio = gainRatio;
                bestThreshold = (currentVal + nextVal) / 2.0;
                foundSplit = true;
            }
        }

        if (foundSplit) {
            bestThresholds.put(attribute, bestThreshold);
            return bestGainRatio;
        }

        return 0.0;
    }

    private double computeEntropyFromCounts(Map<Integer, Integer> counts, int total) {
        double entropy = 0.0;
        int numClasses = counts.size();
        
        // Apply Laplace smoothing: p = (count + alpha) / (total + alpha * numClasses)
        double smoothedTotal = (double) total + LAPLACE_ALPHA * numClasses;
        for (int count : counts.values()) {
            double p = (count + LAPLACE_ALPHA) / smoothedTotal;
            entropy -= p * log2(p);
        }
        return entropy;
    }

    public Map<Integer, ArrayList<Integer>> split(int attribute, ArrayList<Integer> rows) {
        Map<Integer, ArrayList<Integer>> result = new HashMap<>();
        result.put(0, new ArrayList<>());
        result.put(1, new ArrayList<>());

        double threshold = bestThresholds.getOrDefault(attribute, 0.0);
        for (int rowIndex : rows) {
            double val = dataset.getSample(rowIndex)[attribute];
            if (val <= threshold) {
                result.get(0).add(rowIndex);
            } else {
                result.get(1).add(rowIndex);
            }
        }
        return result;
    }

    public int findMostCommonValue(ArrayList<Integer> rows) {
        Map<Integer, Integer> counts = new HashMap<>();
        for (int idx : rows) {
            int label = dataset.getLabel(idx);
            counts.put(label, counts.getOrDefault(label, 0) + 1);
        }
        
        int bestClass = -1;
        int maxCount = -1;
        
        for (Map.Entry<Integer, Integer> entry : counts.entrySet()) {
            if (entry.getValue() > maxCount) {
                maxCount = entry.getValue();
                bestClass = entry.getKey();
            }
        }
        return bestClass;
    }

    public double getEntropy(ArrayList<Integer> rows) {
        return findEntropy(rows);
    }

    public int getMostCommonValue(ArrayList<Integer> rows) {
        return findMostCommonValue(rows);
    }

    public double getSplitThreshold(int attribute) {
        return bestThresholds.getOrDefault(attribute, 0.0);
    }

}
