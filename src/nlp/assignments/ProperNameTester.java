package nlp.assignments;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import nlp.classify.FeatureExtractor;
import nlp.classify.LabeledInstance;
import nlp.classify.ProbabilisticClassifier;
import nlp.classify.ProbabilisticClassifierFactory;
import nlp.util.CommandLineUtils;
import nlp.util.Counter;

/**
 * This is the main harness for assignment 2. To run this harness, use
 * <p/>
 * java nlp.assignments.ProperNameTester -path ASSIGNMENT_DATA_PATH -model
 * MODEL_DESCRIPTOR_STRING
 * <p/>
 * First verify that the data can be read on your system using the baseline
 * model. Second, find the point in the main method (near the bottom) where a
 * MostFrequentLabelClassifier is constructed. You will be writing new
 * implementations of the ProbabilisticClassifer interface and constructing them
 * there.
 */
public class ProperNameTester {

	public static class ProperNameFeatureExtractor implements
			FeatureExtractor<String, String> {

		/**
		 * This method takes the list of characters representing the proper name
		 * description, and produces a list of features which represent that
		 * description. The basic implementation is that only character-unigram
		 * features are extracted. An easy extension would be to also throw
		 * character bigrams into the feature list, but better features are also
		 * possible.
		 */
		public Counter<String> extractFeatures(String name) {
			char[] characters = name.toCharArray();
			Counter<String> features = new Counter<String>();
			
			int len = characters.length;
			// add character ngram features
			for (int i = 0; i < len; i++) {
				String ngram = "";
				for (int j = 0; j < 4; j++) {
					if (i + j >= len) break;
					ngram += characters[i + j];
					if (j >= 1) features.incrementCount(j + "GRAM-" + ngram, 1.0);
				}
			}
			
			// length & num of capital letters
			int countUppercase = 0;
			for (int i = 0; i < len; i++) {
				if (characters[i] >= 'A' && characters[i] <= 'Z') countUppercase++;
			}
			features.incrementCount("LENGTH-" + len, 1.0);
			features.incrementCount("UPPER-" + countUppercase, 1.0);
			for (int i = 3; i < 4; i++) {
				if (len < i) continue;
				String prefix = "";
				String suffix = "";
				for (int j = 0; j < i; j++) {
					prefix += characters[j];
					suffix += characters[len - j - 1];
				}
				features.incrementCount("PREFIX-" + i + "-" + prefix, 1.0);
				features.incrementCount("SUFFIX-" + i + "-" + suffix, 1.0);
			}
			
			
			
			boolean foundDigit = false;
			int countSpace = 0;
			boolean foundDash = false;
			for (int i = 0; i < len; i++) {
				if (characters[i] >= '0' && characters[i] <= '9') foundDigit = true;
				if (characters[i] == ' ') countSpace++;
				if (characters[i] == '-') foundDash = true;
			}
			/*if (foundDigit) 
				features.incrementCount("HASDIGIT", 1.0); 
			*/
			features.incrementCount("SPACE-" + countSpace, 1.0);
			
			/* if (foundDash) 
				features.incrementCount("HASDASH", 1.0);
			*/
			
			/*
			 * String []dict = new String[] {"John", "William", "St", "James", "George", "Sir", "Paul", "Charles", "Henry", "Robert", "Inc",
										  "Corporation", "Fund", "Co", "Trust", "Group","Income", "Corp", "Cap", "I", "West", "La", "Bay",
										  "Hill", "North", "Le", "Bad", "New", "South", "East", "Bridge", "Point", "Strength", 
                                          "Caplets", "with", "Plus", "Cold", "Gel", "Formula", "DM", "The", "the", "a",
                                          "and", "A", "de", "in", "La", "to", "of"};
			 */
			
			String []dict = new String[] {"James", "George", "Sir", "Paul", "Charles", "Henry", "Robert", "Inc",
										  "Corporation", "Fund", "Co", "Trust", "Group","Income", "Corp", "Cap", "I", "West", "La", "Bay",
										  "Hill", "North", "Le", "Bad", "New", "South", "East", "Bridge", "Point", "Strength", 
                                          "Caplets", "with", "Plus", "Cold", "Gel", "Formula", "DM", "The", "the", "a",
                                          "and", "A", "de", "in", "La", "to", "of"};
			
			for (int i = 0; i < 0; i++) {
				String word = " " + dict[i] + " ";
				String newname = " " + name.replaceAll("[^a-zA-Z ]", "") + " ";
				if (newname.contains(word))
				{
					features.incrementCount("DICT-" + word, 1.0);
				}
			}
			
			
			return features;
		}
	}

	private static List<LabeledInstance<String, String>> loadData(
			String fileName) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(fileName));
		List<LabeledInstance<String, String>> labeledInstances = new ArrayList<LabeledInstance<String, String>>();
		while (reader.ready()) {
			String line = reader.readLine();
			String[] parts = line.split("\t");
			String label = parts[0];
			String name = parts[1];
			LabeledInstance<String, String> labeledInstance = new LabeledInstance<String, String>(
					label, name);
			labeledInstances.add(labeledInstance);
		}
		reader.close();
		return labeledInstances;
	}

	private static void testClassifier(
			ProbabilisticClassifier<String, String> classifier,
			List<LabeledInstance<String, String>> testData, boolean verbose) {
		double numCorrect = 0.0;
		double numTotal = 0.0;
		
		Map<String, Integer> labelIndex = new HashMap<String, Integer>();
		int MAXN = 10;
		int [][]confusionMatrix = new int[MAXN][MAXN];
		int countIndex = 0;
		int []confidenceCorrect = new int[10];
		int []confidenceTotal = new int[10];
		
		for (LabeledInstance<String, String> testDatum : testData) {
			String name = testDatum.getInput();
			String label = classifier.getLabel(name);
			String trueLabel = testDatum.getLabel();
			
			if (!labelIndex.containsKey(label))
			{
				labelIndex.put(label, countIndex++);
			}
			if (!labelIndex.containsKey(trueLabel))
			{
				labelIndex.put(trueLabel, countIndex++);
			}
			confusionMatrix[labelIndex.get(trueLabel)][labelIndex.get(label)]++;
			
			double confidence = classifier.getProbabilities(name).getCount(
					label);
			int confidenceLevel = (int) (confidence * 10);
			if (confidenceLevel == 10) confidenceLevel--;
			confidenceTotal[confidenceLevel]++;
			
			if (label.equals(trueLabel)) {
				numCorrect += 1.0;
				confidenceCorrect[confidenceLevel]++;
			} else {
				if (verbose) {
					// display an error
					System.err.println("Error: " + name + "    guess=" + label
							+ "    gold=" + testDatum.getLabel() + "    confidence="
							+ confidence);
				}
			}
			numTotal += 1.0;
		}
		double accuracy = numCorrect / numTotal;
		System.out.println("Accuracy: " + accuracy);
		
		if (verbose) {	
			System.out.println("Confusion Matrix:");
			System.out.print(String.format("%1$10s", ""));
			for (String label: labelIndex.keySet()) {
				System.out.print(String.format("%1$10s", label));
			}
			System.out.println();
			for (String label: labelIndex.keySet()) {
				System.out.print(String.format("%1$10s", label));
				for (String label2: labelIndex.keySet()) {
					System.out.print(String.format("%1$10s", confusionMatrix[labelIndex.get(label)][labelIndex.get(label2)] + ""));
				}
				System.out.println();
			}
			for (int i = 0; i < 10; i++) {
				System.out.println("Confidence Level " + i + ", Accuracy: " + (double)confidenceCorrect[i] / confidenceTotal[i]);
			}
		}
		
	}

	public static void main(String[] args) throws IOException {
		// Parse command line flags and arguments
		Map<String, String> argMap = CommandLineUtils
				.simpleCommandLineParser(args);

		// Set up default parameters and settings
		String basePath = ".";
		String model = "baseline";
		boolean verbose = false;
		boolean useValidation = true;

		// Update defaults using command line specifications

		// The path to the assignment data
		if (argMap.containsKey("-path")) {
			basePath = argMap.get("-path");
		}
		System.out.println("Using base path: " + basePath);

		// A string descriptor of the model to use
		if (argMap.containsKey("-model")) {
			model = argMap.get("-model");
		}
		System.out.println("Using model: " + model);

		// A string descriptor of the model to use
		if (argMap.containsKey("-test")) {
			String testString = argMap.get("-test");
			if (testString.equalsIgnoreCase("test"))
				useValidation = false;
		}
		System.out.println("Testing on: "
				+ (useValidation ? "validation" : "test"));

		// Whether or not to print the individual speech errors.
		if (argMap.containsKey("-verbose")) {
			verbose = true;
		}

		// Load training, validation, and test data
		List<LabeledInstance<String, String>> trainingData = loadData(basePath
				+ "/pnp-train.txt");
		List<LabeledInstance<String, String>> validationData = loadData(basePath
				+ "/pnp-validate.txt");
		List<LabeledInstance<String, String>> testData = loadData(basePath
				+ "/pnp-test.txt");

		// Learn a classifier
		ProbabilisticClassifier<String, String> classifier = null;
		if (model.equalsIgnoreCase("baseline")) {
			classifier = new MostFrequentLabelClassifier.Factory<String, String>()
					.trainClassifier(trainingData);
		} else if (model.equalsIgnoreCase("n-gram")) {
			// TODO: construct your n-gram model here
		} else if (model.equalsIgnoreCase("perceptron")){
			ProbabilisticClassifierFactory<String, String> factory = new PerceptronClassifier.Factory<String, String, String>(
					1.0, 20, new ProperNameFeatureExtractor());
			classifier = factory.trainClassifier(trainingData);
		} else if (model.equalsIgnoreCase("maxent")) {
			// TODO: construct your maxent model here
			ProbabilisticClassifierFactory<String, String> factory = new MaximumEntropyClassifier.Factory<String, String, String>(
					1.0, 20, new ProperNameFeatureExtractor());
			classifier = factory.trainClassifier(trainingData);
		} else {
			throw new RuntimeException("Unknown model descriptor: " + model);
		}

		// Test classifier
		testClassifier(classifier, (useValidation ? validationData : testData),
				verbose);
	}
}
