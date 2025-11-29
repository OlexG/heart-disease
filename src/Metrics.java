/**
 * Utility class for calculating evaluation metrics for binary classification
 */
public class Metrics {
    
    /**
     * Calculate accuracy: (TP + TN) / (TP + TN + FP + FN)
     */
    public static double accuracy(int[] predictions, int[] actual) {
        if (predictions.length != actual.length) {
            throw new IllegalArgumentException("Predictions and actual arrays must have the same length");
        }
        
        int correct = 0;
        for (int i = 0; i < predictions.length; i++) {
            if (predictions[i] == actual[i]) {
                correct++;
            }
        }
        return (double) correct / predictions.length;
    }
    
    /**
     * Calculate precision: TP / (TP + FP)
     */
    public static double precision(int[] predictions, int[] actual) {
        int[] cm = confusionMatrix(predictions, actual);
        int tp = cm[0]; // True Positives
        int fp = cm[1]; // False Positives
        
        if (tp + fp == 0) {
            return 0.0; // Avoid division by zero
        }
        return (double) tp / (tp + fp);
    }
    
    /**
     * Calculate recall: TP / (TP + FN)
     */
    public static double recall(int[] predictions, int[] actual) {
        int[] cm = confusionMatrix(predictions, actual);
        int tp = cm[0]; // True Positives
        int fn = cm[3]; // False Negatives
        
        if (tp + fn == 0) {
            return 0.0; // Avoid division by zero
        }
        return (double) tp / (tp + fn);
    }
    
    /**
     * Calculate F1 score: 2 * (precision * recall) / (precision + recall)
     */
    public static double f1Score(int[] predictions, int[] actual) {
        double prec = precision(predictions, actual);
        double rec = recall(predictions, actual);
        
        if (prec + rec == 0) {
            return 0.0; // Avoid division by zero
        }
        return 2.0 * (prec * rec) / (prec + rec);
    }
    
    /**
     * Compute confusion matrix for binary classification
     * Returns array: [TP, FP, TN, FN]
     * Assumes class 1 is positive, class 0 is negative
     */
    private static int[] confusionMatrix(int[] predictions, int[] actual) {
        if (predictions.length != actual.length) {
            throw new IllegalArgumentException("Predictions and actual arrays must have the same length");
        }
        
        int tp = 0; // True Positives (predicted 1, actual 1)
        int fp = 0; // False Positives (predicted 1, actual 0)
        int tn = 0; // True Negatives (predicted 0, actual 0)
        int fn = 0; // False Negatives (predicted 0, actual 1)
        
        for (int i = 0; i < predictions.length; i++) {
            if (predictions[i] == 1 && actual[i] == 1) {
                tp++;
            } else if (predictions[i] == 1 && actual[i] == 0) {
                fp++;
            } else if (predictions[i] == 0 && actual[i] == 0) {
                tn++;
            } else if (predictions[i] == 0 && actual[i] == 1) {
                fn++;
            }
        }
        
        return new int[]{tp, fp, tn, fn};
    }
    
    /**
     * Expose confusion matrix counts for downstream reporting.
     */
    public static int[] confusionMatrixCounts(int[] predictions, int[] actual) {
        return confusionMatrix(predictions, actual);
    }
    
    /**
     * Get confusion matrix as a readable string
     */
    public static String confusionMatrixString(int[] predictions, int[] actual) {
        int[] cm = confusionMatrix(predictions, actual);
        int tp = cm[0];
        int fp = cm[1];
        int tn = cm[2];
        int fn = cm[3];
        
        return String.format(
            "Confusion Matrix:\n" +
            "                Predicted\n" +
            "                0      1\n" +
            "Actual  0    %4d  %4d\n" +
            "        1    %4d  %4d\n" +
            "TP: %d, FP: %d, TN: %d, FN: %d",
            tn, fp, fn, tp, tp, fp, tn, fn
        );
    }
}

