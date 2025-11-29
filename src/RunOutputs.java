import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.Locale;

/**
 * Handles creation of per-run output folders and writing model artifacts.
 */
public class RunOutputs {
    private static final DateTimeFormatter DIR_FORMAT = DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss");
    private static final DateTimeFormatter ISO_FORMAT = DateTimeFormatter.ISO_LOCAL_DATE_TIME;
    private static final Locale LOCALE = Locale.US;

    private final Path runDir;
    private final String runId;
    private final String timestamp;

    public RunOutputs() throws IOException {
        this("outputs");
    }

    public RunOutputs(String baseDir) throws IOException {
        LocalDateTime now = LocalDateTime.now();
        this.timestamp = ISO_FORMAT.format(now);
        this.runId = "run_" + DIR_FORMAT.format(now);

        Path basePath = Paths.get(baseDir);
        Files.createDirectories(basePath);
        this.runDir = Files.createDirectories(basePath.resolve(runId));
    }

    public Path getRunDir() {
        return runDir;
    }

    public String getRunId() {
        return runId;
    }

    public String getTimestamp() {
        return timestamp;
    }

    public void writeProcessLog(List<String> entries) throws IOException {
        Path logFile = runDir.resolve("process_log.txt");
        Files.write(logFile, entries, StandardCharsets.UTF_8);
    }

    public void writePredictionsCsv(int[] predictions, int[] actual, double[] probabilities) throws IOException {
        if (predictions.length != actual.length || predictions.length != probabilities.length) {
            throw new IllegalArgumentException("Predictions, actual labels, and probabilities must have the same length.");
        }

        Path file = runDir.resolve("test_predictions.csv");
        StringBuilder sb = new StringBuilder();
        sb.append("sample_index,prediction,actual,prob_heart_disease,confidence,correct\n");

        for (int i = 0; i < predictions.length; i++) {
            double probability = clamp(probabilities[i]);
            double confidence = predictions[i] == 1 ? probability : (1.0 - probability);
            sb.append(i)
              .append(',')
              .append(predictions[i])
              .append(',')
              .append(actual[i])
              .append(',')
              .append(String.format(LOCALE, "%.6f", probability))
              .append(',')
              .append(String.format(LOCALE, "%.6f", confidence))
              .append(',')
              .append(predictions[i] == actual[i])
              .append('\n');
        }

        Files.writeString(file, sb.toString(), StandardCharsets.UTF_8);
    }

    public void writeSummary(SummaryData data) throws IOException {
        Path file = runDir.resolve("run_summary.json");
        String indent = "  ";
        StringBuilder sb = new StringBuilder();

        sb.append("{\n");
        sb.append(indent).append("\"runId\": \"").append(runId).append("\",\n");
        sb.append(indent).append("\"timestamp\": \"").append(timestamp).append("\",\n");
        sb.append(indent).append("\"dataset\": {\n");
        sb.append(indent).append(indent).append("\"path\": \"").append(escapeJson(data.datasetPath)).append("\",\n");
        sb.append(indent).append(indent).append("\"features\": ").append(data.numFeatures).append(",\n");
        sb.append(indent).append(indent).append("\"trainSamples\": ").append(data.trainSamples).append(",\n");
        sb.append(indent).append(indent).append("\"testSamples\": ").append(data.testSamples).append("\n");
        sb.append(indent).append("},\n");

        sb.append(indent).append("\"hyperparameters\": {\n");
        sb.append(indent).append(indent).append("\"numTrees\": ").append(data.numTrees).append(",\n");
        sb.append(indent).append(indent).append("\"maxDepth\": ").append(data.maxDepth == null ? "null" : data.maxDepth).append(",\n");
        sb.append(indent).append(indent).append("\"minSamplesSplit\": ").append(data.minSamplesSplit).append(",\n");
        sb.append(indent).append(indent).append("\"maxFeatures\": ").append(data.maxFeatures).append("\n");
        sb.append(indent).append("},\n");

        sb.append(indent).append("\"training\": {\n");
        sb.append(indent).append(indent).append("\"durationMs\": ").append(data.trainingDurationMs).append("\n");
        sb.append(indent).append("},\n");

        sb.append(indent).append("\"trainMetrics\": {\n");
        sb.append(indent).append(indent).append("\"accuracy\": ").append(formatDouble(data.trainAccuracy)).append(",\n");
        sb.append(indent).append(indent).append("\"precision\": ").append(formatDouble(data.trainPrecision)).append(",\n");
        sb.append(indent).append(indent).append("\"recall\": ").append(formatDouble(data.trainRecall)).append(",\n");
        sb.append(indent).append(indent).append("\"f1\": ").append(formatDouble(data.trainF1)).append("\n");
        sb.append(indent).append("},\n");

        sb.append(indent).append("\"testMetrics\": {\n");
        sb.append(indent).append(indent).append("\"accuracy\": ").append(formatDouble(data.testAccuracy)).append(",\n");
        sb.append(indent).append(indent).append("\"precision\": ").append(formatDouble(data.testPrecision)).append(",\n");
        sb.append(indent).append(indent).append("\"recall\": ").append(formatDouble(data.testRecall)).append(",\n");
        sb.append(indent).append(indent).append("\"f1\": ").append(formatDouble(data.testF1)).append("\n");
        sb.append(indent).append("},\n");

        sb.append(indent).append("\"testConfusionMatrix\": {\n");
        sb.append(indent).append(indent).append("\"truePositives\": ").append(data.confusionMatrix[0]).append(",\n");
        sb.append(indent).append(indent).append("\"falsePositives\": ").append(data.confusionMatrix[1]).append(",\n");
        sb.append(indent).append(indent).append("\"trueNegatives\": ").append(data.confusionMatrix[2]).append(",\n");
        sb.append(indent).append(indent).append("\"falseNegatives\": ").append(data.confusionMatrix[3]).append("\n");
        sb.append(indent).append("}\n");

        sb.append("}\n");

        Files.writeString(file, sb.toString(), StandardCharsets.UTF_8);
    }

    public Path writeTreeVisualization(int treeIndex, String dotContent) throws IOException {
        Path file = runDir.resolve(String.format("tree_viz_%d.dot", treeIndex));
        Files.writeString(file, dotContent, StandardCharsets.UTF_8);
        return file;
    }

    private static String escapeJson(String value) {
        return value.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    private static double clamp(double value) {
        if (Double.isNaN(value)) {
            return 0.0;
        }
        if (value < 0.0) {
            return 0.0;
        }
        if (value > 1.0) {
            return 1.0;
        }
        return value;
    }

    private static String formatDouble(double value) {
        return String.format(LOCALE, "%.4f", value);
    }

    /**
     * Container for the values that make up the JSON summary.
     */
    public static class SummaryData {
        private final String datasetPath;
        private final int numFeatures;
        private final int trainSamples;
        private final int testSamples;
        private final int numTrees;
        private final Integer maxDepth;
        private final int minSamplesSplit;
        private final int maxFeatures;
        private final long trainingDurationMs;
        private final double trainAccuracy;
        private final double trainPrecision;
        private final double trainRecall;
        private final double trainF1;
        private final double testAccuracy;
        private final double testPrecision;
        private final double testRecall;
        private final double testF1;
        private final int[] confusionMatrix;

        public SummaryData(
            String datasetPath,
            int numFeatures,
            int trainSamples,
            int testSamples,
            int numTrees,
            Integer maxDepth,
            int minSamplesSplit,
            int maxFeatures,
            long trainingDurationMs,
            double trainAccuracy,
            double trainPrecision,
            double trainRecall,
            double trainF1,
            double testAccuracy,
            double testPrecision,
            double testRecall,
            double testF1,
            int[] confusionMatrix
        ) {
            this.datasetPath = datasetPath;
            this.numFeatures = numFeatures;
            this.trainSamples = trainSamples;
            this.testSamples = testSamples;
            this.numTrees = numTrees;
            this.maxDepth = maxDepth;
            this.minSamplesSplit = minSamplesSplit;
            this.maxFeatures = maxFeatures;
            this.trainingDurationMs = trainingDurationMs;
            this.trainAccuracy = trainAccuracy;
            this.trainPrecision = trainPrecision;
            this.trainRecall = trainRecall;
            this.trainF1 = trainF1;
            this.testAccuracy = testAccuracy;
            this.testPrecision = testPrecision;
            this.testRecall = testRecall;
            this.testF1 = testF1;
            if (confusionMatrix == null || confusionMatrix.length != 4) {
                throw new IllegalArgumentException("Confusion matrix must contain exactly four entries.");
            }
            this.confusionMatrix = confusionMatrix.clone();
        }
    }
}

