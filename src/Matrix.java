import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;

public class Matrix {
	private final int[][] table;

	public Matrix(int[][] data) {
		this.table = data;
	}

	private int findFrequency(int attribute, int value, ArrayList<Integer> rows) {
		int count = 0;
		for (int rowIndex : rows) {
			if (table[rowIndex][attribute] == value) {
				count++;
			}
		}
		return count;
	}

	private HashSet<Integer> findDifferentValues(int attribute, ArrayList<Integer> rows) {
		HashSet<Integer> values = new HashSet<Integer>();
		for (int rowIndex : rows) {
			values.add(table[rowIndex][attribute]);
		}
		return values;
	}

	private ArrayList<Integer> findRows(int attribute, int value, ArrayList<Integer> rows) {
		ArrayList<Integer> matchingRows = new ArrayList<Integer>();
		for (int rowIndex : rows) {
			if (table[rowIndex][attribute] == value) {
				matchingRows.add(rowIndex);
			}
		}
		return matchingRows;
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
		HashMap<Integer, Integer> classCounts = new HashMap<Integer, Integer>();
		for (int rowIndex : rows) {
			int classKey = table[rowIndex][4];
			classCounts.put(classKey, classCounts.getOrDefault(classKey, 0) + 1);
		}
		double total = (double) rows.size();
		double entropy = 0.0;
		for (Map.Entry<Integer, Integer> entry : classCounts.entrySet()) {
			double p = entry.getValue() / total;
			if (p > 0.0) {
				entropy -= p * log2(p);
			}
		}
		return entropy;
	}

	private double findEntropy(int attribute, ArrayList<Integer> rows) {
		if (rows.isEmpty()) {
			return 0.0;
		}
		HashSet<Integer> values = findDifferentValues(attribute, rows);
		double total = (double) rows.size();
		double conditionalEntropy = 0.0;
		for (int value : values) {
			ArrayList<Integer> subset = findRows(attribute, value, rows);
			if (subset.isEmpty()) {
				continue;
			}
			double weight = subset.size() / total;
			conditionalEntropy += weight * findEntropy(subset);
		}
		return conditionalEntropy;
	}

	private double findGain(int attribute, ArrayList<Integer> rows) {
		double baseEntropy = findEntropy(rows);
		double conditionalEntropy = findEntropy(attribute, rows);
		return baseEntropy - conditionalEntropy;
	}

	public double computeIGR(int attribute, ArrayList<Integer> rows) {
		if (rows.isEmpty()) {
			return 0.0;
		}
		HashSet<Integer> values = findDifferentValues(attribute, rows);
		double total = (double) rows.size();
		double splitInfo = 0.0;
		for (int value : values) {
			int freq = findFrequency(attribute, value, rows);
			double ratio = freq / total;
			if (ratio > 0.0) {
				splitInfo -= ratio * log2(ratio);
			}
		}
		if (splitInfo <= 0.0) {
			return 0.0;
		}
		double gain = findGain(attribute, rows);
		return gain / splitInfo;
	}

	public int findMostCommonValue(ArrayList<Integer> rows) {
		HashMap<Integer, Integer> classCounts = new HashMap<Integer, Integer>();
		for (int rowIndex : rows) {
			int klass = table[rowIndex][4];
			classCounts.put(klass, classCounts.getOrDefault(klass, 0) + 1);
		}
		int bestClass = -1;
		int bestCount = -1;
		for (Map.Entry<Integer, Integer> entry : classCounts.entrySet()) {
			int klass = entry.getKey();
			int count = entry.getValue();
			if (count > bestCount || (count == bestCount && klass < bestClass)) {
				bestCount = count;
				bestClass = klass;
			}
		}
		return bestClass;
	}

	public HashMap<Integer, ArrayList<Integer>> split(int attribute, ArrayList<Integer> rows) {
		HashMap<Integer, ArrayList<Integer>> result = new HashMap<Integer, ArrayList<Integer>>();
		for (int rowIndex : rows) {
			int value = table[rowIndex][attribute];
			ArrayList<Integer> list = result.get(value);
			if (list == null) {
				list = new ArrayList<Integer>();
				result.put(value, list);
			}
			list.add(rowIndex);
		}
		return result;
	}

	public double getEntropy(ArrayList<Integer> rows) {
		return findEntropy(rows);
	}
}


