import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public class TreeMatrix {
    private static final double LAPLACE_ALPHA = 1.0; // Laplace smoothing parameter 
    private final Dataset dataset;
    // Cache to store the best threshold found during computeIGR for each attribute
    private final Map<Integer, Double> bestThresholds = new HashMap<>();
    // Cache to store categorical split info (left-side categories for each attribute)
    private final Map<Integer, Set<Integer>> categoricalSplits = new HashMap<>();

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

        // Check if attribute is categorical
        if (isCategorical(attribute)) {
            return computeIGR_Categorical(attribute, rows, parentEntropy);
        }

        // Numerical attribute - use existing threshold-based logic

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
        result.put(0, new ArrayList<>()); // Left
        result.put(1, new ArrayList<>()); // Right

        if (isCategorical(attribute)) {
            // Categorical split: check if value is in left category set
            Set<Integer> leftCategories = categoricalSplits.getOrDefault(attribute, new HashSet<>());
            for (int rowIndex : rows) {
                int categoryValue = (int) dataset.getSample(rowIndex)[attribute];
                if (leftCategories.contains(categoryValue)) {
                    result.get(0).add(rowIndex);
                } else {
                    result.get(1).add(rowIndex);
                }
            }
        } else {
            // Numerical split: use threshold
            double threshold = bestThresholds.getOrDefault(attribute, 0.0);
            for (int rowIndex : rows) {
                double val = dataset.getSample(rowIndex)[attribute];
                if (val <= threshold) {
                    result.get(0).add(rowIndex);
                } else {
                    result.get(1).add(rowIndex);
                }
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

    /**
     * Check if an attribute is categorical
     */
    public boolean isCategorical(int attribute) {
        return dataset.getCategoricalAttributes().contains(attribute);
    }

    /**
     * Compute Information Gain Ratio for categorical attributes
     */
    public double computeIGR_Categorical(int attribute, ArrayList<Integer> rows, double parentEntropy) {
        int numSamples = rows.size();
        if (numSamples <= 1) return 0.0;


        // Find all unique category values
        Set<Integer> uniqueCategories = new HashSet<>();
        Map<Integer, Map<Integer, Integer>> categoryClassCounts = new HashMap<>();
        
        for (int idx : rows) {
            int categoryValue = (int) dataset.getSample(idx)[attribute];
            int label = dataset.getLabel(idx);
            
            uniqueCategories.add(categoryValue);
            categoryClassCounts.putIfAbsent(categoryValue, new HashMap<>());
            Map<Integer, Integer> classCounts = categoryClassCounts.get(categoryValue);
            classCounts.put(label, classCounts.getOrDefault(label, 0) + 1);
        }

        if (uniqueCategories.size() <= 1) return 0.0;

        // Sort categories by class purity (ratio of majority class)
        // Optimized: Pre-calculate purity to avoid expensive stream operations in comparator
        Map<Integer, Double> purityMap = new HashMap<>();
        for (Integer cat : uniqueCategories) {
            Map<Integer, Integer> counts = categoryClassCounts.get(cat);
            int total = 0;
            int max = 0;
            for (int count : counts.values()) {
                total += count;
                if (count > max) max = count;
            }
            double purity = total > 0 ? (double) max / total : 0.0;
            purityMap.put(cat, purity);
        }

        ArrayList<Integer> sortedCategories = new ArrayList<>(uniqueCategories);
        sortedCategories.sort((a, b) -> Double.compare(purityMap.get(a), purityMap.get(b)));

        double bestGainRatio = 0.0;
        Set<Integer> bestLeftCategories = null;
        boolean foundSplit = false;

        // Evaluate sequential splits
        Set<Integer> leftCategories = new HashSet<>();
        Map<Integer, Integer> leftCounts = new HashMap<>();
        Map<Integer, Integer> rightCounts = new HashMap<>();
        
        // Initialize right counts with all samples
        for (int idx : rows) {
            int label = dataset.getLabel(idx);
            rightCounts.put(label, rightCounts.getOrDefault(label, 0) + 1);
        }
        
        int leftSize = 0;
        int rightSize = numSamples;

        for (int i = 0; i < sortedCategories.size() - 1; i++) {
            int category = sortedCategories.get(i);
            leftCategories.add(category);
            
            // Move samples with this category from right to left
            Map<Integer, Integer> currentCatCounts = categoryClassCounts.get(category);
            if (currentCatCounts != null) {
                for (Map.Entry<Integer, Integer> entry : currentCatCounts.entrySet()) {
                    int label = entry.getKey();
                    int count = entry.getValue();
                    
                    // Update right counts
                    rightCounts.put(label, rightCounts.getOrDefault(label, 0) - count);
                    
                    // Update left counts
                    leftCounts.put(label, leftCounts.getOrDefault(label, 0) + count);
                    
                    // Update sizes
                    rightSize -= count;
                    leftSize += count;
                }
            }

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
                bestLeftCategories = new HashSet<>(leftCategories);
                foundSplit = true;
            }
        }

        if (foundSplit && bestLeftCategories != null) {
            categoricalSplits.put(attribute, bestLeftCategories);
            return bestGainRatio;
        }

        return 0.0;
    }

    /**
     * Get the categorical split (left-side categories) for an attribute
     */
    public Set<Integer> getCategoricalSplit(int attribute) {
        return categoricalSplits.getOrDefault(attribute, new HashSet<>());
    }
}
