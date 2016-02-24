package nlp.assignments;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.AbstractCollection;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Set;

import nlp.langmodel.LanguageModel;
import nlp.util.CommandLineUtils;

/**
 * This is the main harness for assignment 1. To run this harness, use
 * <p/>
 * java nlp.assignments.LanguageModelTester -path ASSIGNMENT_DATA_PATH -model
 * MODEL_DESCRIPTOR_STRING
 * <p/>
 * First verify that the data can be read on your system. Second, find the point
 * in the main method (near the bottom) where an EmpiricalUnigramLanguageModel
 * is constructed. You will be writing new implementations of the LanguageModel
 * interface and constructing them there.
 */
public class LanguageModelTester {

	private static NumberFormat nf = new DecimalFormat("0.00E00");

	public static void main(String[] args) throws IOException {
		// Parse command line flags and arguments
		final Map<String, String> argMap = CommandLineUtils
				.simpleCommandLineParser(args);

		// Set up default parameters and settings
		String basePath = ".";
		String model = "baseline";
		boolean verbose = false;

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

		// Whether or not to print the individual speech errors.
		if (argMap.containsKey("-verbose")) {
			verbose = true;
		}
		if (argMap.containsKey("-quiet")) {
			verbose = false;
		}

		// Read in all the assignment data
		final String trainingSentencesFile = "/treebank-sentences-spoken-train.txt";
		final String speechNBestListsPath = "/wsj_n_bst";
		final Collection<List<String>> trainingSentenceCollection = SentenceCollection.Reader
				.readSentenceCollection(basePath + trainingSentencesFile);
		final Set<String> trainingVocabulary = extractVocabulary(
				trainingSentenceCollection);
		final List<SpeechNBestList> speechNBestLists = SpeechNBestList.Reader
				.readSpeechNBestLists(basePath + speechNBestListsPath,
						trainingVocabulary);

		// String validationSentencesFile =
		// "/treebank-sentences-spoken-validate.txt";
		// Collection<List<String>> validationSentenceCollection =
		// SentenceCollection.Reader.readSentenceCollection(basePath +
		// validationSentencesFile);

		// String testSentencesFile = "/treebank-sentences-spoken-test.txt";
		// Collection<List<String>> testSentenceCollection =
		// SentenceCollection.Reader.readSentenceCollection(basePath +
		// testSentencesFile);

		// Build the language model
		LanguageModel languageModel = null;
		if (model.equalsIgnoreCase("baseline")) {
			languageModel = new EmpiricalUnigramLanguageModel(
					trainingSentenceCollection);
		} else if (model.equalsIgnoreCase("sri")) {
			languageModel = new SriLanguageModel(argMap.get("-sri"));
		} else if (model.equalsIgnoreCase("bigram")) {
			languageModel = new EmpiricalBigramLanguageModel(
					trainingSentenceCollection);
		} else if (model.equalsIgnoreCase("trigram")) {
			languageModel = new EmpiricalTrigramLanguageModel(
					trainingSentenceCollection);
		} else if (model.equalsIgnoreCase("katz-bigram")) {
			languageModel = new KatzBigramLanguageModel(
					trainingSentenceCollection);
		} else if (model.equalsIgnoreCase("katz-bigram-pp")) {
			languageModel = new KatzPPBigramLanguageModel(
					trainingSentenceCollection);
		} else if (model.equalsIgnoreCase("katz-trigram")) {
			throw new IllegalStateException(
					"Katz trigram model not fully implemented -- remove exception and uncomment next line if implemented");
			// languageModel = new KatzTrigramLanguageModel(
			// trainingSentenceCollection);
		} else {
			throw new RuntimeException("Unknown model descriptor: " + model);
		}

		// Evaluate the language model
		// final double wsjPerplexity = calculatePerplexity(languageModel,
		// testSentenceCollection);
		final double hubPerplexity = calculatePerplexity(languageModel,
				extractCorrectSentenceList(speechNBestLists));
		// System.out.println("WSJ Perplexity: " + wsjPerplexity);
		System.out.println("HUB Perplexity:  " + hubPerplexity);
		System.out.println("WER Baselines:");
		System.out.println("  Best Path:  "
				+ calculateWordErrorRateLowerBound(speechNBestLists));
		System.out.println("  Worst Path: "
				+ calculateWordErrorRateUpperBound(speechNBestLists));
		System.out.println("  Avg Path:   "
				+ calculateWordErrorRateRandomChoice(speechNBestLists));
		final double wordErrorRate = calculateWordErrorRate(languageModel,
				speechNBestLists, verbose);
		System.out.println("HUB Word Error Rate: " + wordErrorRate);
		// System.out.println("Generated Sentences:");
		// for (int i = 0; i < 10; i++)
		// System.out.println(" " + languageModel.generateSentence());
	}

	private static void displayHypothesis(String prefix, List<String> guess,
			SpeechNBestList speechNBestList, LanguageModel languageModel) {
		final double acoustic = speechNBestList.getAcousticScore(guess) / 16.0;
		final double language = Math
				.log(languageModel.getSentenceProbability(guess));
		System.out.println(prefix + "\tAM: " + nf.format(acoustic) + "\tLM: "
				+ nf.format(language) + "\tTotal: "
				+ nf.format(acoustic + language) + "\t" + guess);
	}

	static double calculatePerplexity(LanguageModel languageModel,
			Collection<List<String>> sentenceCollection) {
		double logProbability = 0.0;
		double numSymbols = 0.0;
		for (final List<String> sentence : sentenceCollection) {
			logProbability += Math
					.log(languageModel.getSentenceProbability(sentence))
					/ Math.log(2.0);
			numSymbols += sentence.size();
		}
		final double avgLogProbability = logProbability / numSymbols;
		final double perplexity = Math.pow(0.5, avgLogProbability);
		return perplexity;
	}

	static double calculateWordErrorRate(LanguageModel languageModel,
			List<SpeechNBestList> speechNBestLists, boolean verbose) {
		double totalDistance = 0.0;
		double totalWords = 0.0;
		final EditDistance editDistance = new EditDistance();
		for (final SpeechNBestList speechNBestList : speechNBestLists) {
			final List<String> correctSentence = speechNBestList
					.getCorrectSentence();
			List<String> bestGuess = null;
			double bestScore = Double.NEGATIVE_INFINITY;
			double numWithBestScores = 0.0;
			double distanceForBestScores = 0.0;
			for (final List<String> guess : speechNBestList
					.getNBestSentences()) {
				final double score = Math
						.log(languageModel.getSentenceProbability(guess))
						+ speechNBestList.getAcousticScore(guess) / 16.0;
				final double distance = editDistance
						.getDistance(correctSentence, guess);
				if (score == bestScore) {
					numWithBestScores += 1.0;
					distanceForBestScores += distance;
				}
				if (score > bestScore || bestGuess == null) {
					bestScore = score;
					bestGuess = guess;
					distanceForBestScores = distance;
					numWithBestScores = 1.0;
				}
			}
			// double distance = editDistance.getDistance(correctSentence,
			// bestGuess);
			totalDistance += distanceForBestScores / numWithBestScores;
			totalWords += correctSentence.size();
			if (verbose) {
				System.out.println();
				displayHypothesis("GUESS:", bestGuess, speechNBestList,
						languageModel);
				displayHypothesis("GOLD:", correctSentence, speechNBestList,
						languageModel);
			}
		}
		return totalDistance / totalWords;
	}

	static double calculateWordErrorRateLowerBound(
			List<SpeechNBestList> speechNBestLists) {
		double totalDistance = 0.0;
		double totalWords = 0.0;
		final EditDistance editDistance = new EditDistance();
		for (final SpeechNBestList speechNBestList : speechNBestLists) {
			final List<String> correctSentence = speechNBestList
					.getCorrectSentence();
			double bestDistance = Double.POSITIVE_INFINITY;
			for (final List<String> guess : speechNBestList
					.getNBestSentences()) {
				final double distance = editDistance
						.getDistance(correctSentence, guess);
				if (distance < bestDistance) {
					bestDistance = distance;
				}
			}
			totalDistance += bestDistance;
			totalWords += correctSentence.size();
		}
		return totalDistance / totalWords;
	}

	static double calculateWordErrorRateRandomChoice(
			List<SpeechNBestList> speechNBestLists) {
		double totalDistance = 0.0;
		double totalWords = 0.0;
		final EditDistance editDistance = new EditDistance();
		for (final SpeechNBestList speechNBestList : speechNBestLists) {
			final List<String> correctSentence = speechNBestList
					.getCorrectSentence();
			double sumDistance = 0.0;
			double numGuesses = 0.0;
			for (final List<String> guess : speechNBestList
					.getNBestSentences()) {
				final double distance = editDistance
						.getDistance(correctSentence, guess);
				sumDistance += distance;
				numGuesses += 1.0;
			}
			totalDistance += sumDistance / numGuesses;
			totalWords += correctSentence.size();
		}
		return totalDistance / totalWords;
	}

	static double calculateWordErrorRateUpperBound(
			List<SpeechNBestList> speechNBestLists) {
		double totalDistance = 0.0;
		double totalWords = 0.0;
		final EditDistance editDistance = new EditDistance();
		for (final SpeechNBestList speechNBestList : speechNBestLists) {
			final List<String> correctSentence = speechNBestList
					.getCorrectSentence();
			double worstDistance = Double.NEGATIVE_INFINITY;
			for (final List<String> guess : speechNBestList
					.getNBestSentences()) {
				final double distance = editDistance
						.getDistance(correctSentence, guess);
				if (distance > worstDistance) {
					worstDistance = distance;
				}
			}
			totalDistance += worstDistance;
			totalWords += correctSentence.size();
		}
		return totalDistance / totalWords;
	}

	static Collection<List<String>> extractCorrectSentenceList(
			List<SpeechNBestList> speechNBestLists) {
		final Collection<List<String>> correctSentences = new ArrayList<List<String>>();
		for (final SpeechNBestList speechNBestList : speechNBestLists) {
			correctSentences.add(speechNBestList.getCorrectSentence());
		}
		return correctSentences;
	}

	static Set<String> extractVocabulary(
			Collection<List<String>> sentenceCollection) {
		final Set<String> vocabulary = new HashSet<String>();
		for (final List<String> sentence : sentenceCollection) {
			for (final String word : sentence) {
				vocabulary.add(word);
			}
		}
		return vocabulary;
	}

	// HELPER CLASS FOR THE HARNESS, CAN IGNORE
	static class EditDistance {
		static double	DELETE_COST		= 1.0;
		static double	INSERT_COST		= 1.0;
		static double	SUBSTITUTE_COST	= 1.0;

		private static double[][] initialize(double[][] d) {
			for (int i = 0; i < d.length; i++) {
				for (int j = 0; j < d[i].length; j++) {
					d[i][j] = Double.NaN;
				}
			}
			return d;
		}

		public double getDistance(List<? extends Object> firstList,
				List<? extends Object> secondList) {
			final double[][] bestDistances = initialize(
					new double[firstList.size() + 1][secondList.size() + 1]);
			return getDistance(firstList, secondList, 0, 0, bestDistances);
		}

		private double getDistance(List<? extends Object> firstList,
				List<? extends Object> secondList, int firstPosition,
				int secondPosition, double[][] bestDistances) {
			if (firstPosition > firstList.size()
					|| secondPosition > secondList.size()) {
				return Double.POSITIVE_INFINITY;
			}
			if (firstPosition == firstList.size()
					&& secondPosition == secondList.size()) {
				return 0.0;
			}
			if (Double.isNaN(bestDistances[firstPosition][secondPosition])) {
				double distance = Double.POSITIVE_INFINITY;
				distance = Math.min(distance,
						INSERT_COST + getDistance(firstList, secondList,
								firstPosition + 1, secondPosition,
								bestDistances));
				distance = Math.min(distance,
						DELETE_COST + getDistance(firstList, secondList,
								firstPosition, secondPosition + 1,
								bestDistances));
				distance = Math.min(distance,
						SUBSTITUTE_COST + getDistance(firstList, secondList,
								firstPosition + 1, secondPosition + 1,
								bestDistances));
				if (firstPosition < firstList.size()
						&& secondPosition < secondList.size()) {
					if (firstList.get(firstPosition)
							.equals(secondList.get(secondPosition))) {
						distance = Math.min(distance,
								getDistance(firstList, secondList,
										firstPosition + 1, secondPosition + 1,
										bestDistances));
					}
				}
				bestDistances[firstPosition][secondPosition] = distance;
			}
			return bestDistances[firstPosition][secondPosition];
		}
	}

	// HELPER CLASS FOR THE HARNESS, CAN IGNORE
	static class SentenceCollection extends AbstractCollection<List<String>> {
		String fileName;

		public SentenceCollection(String fileName) {
			this.fileName = fileName;
		}

		@Override
		public Iterator<List<String>> iterator() {
			try {
				final BufferedReader reader = new BufferedReader(
						new FileReader(fileName));
				return new SentenceIterator(reader);
			} catch (final FileNotFoundException e) {
				throw new RuntimeException(
						"Problem with SentenceIterator for " + fileName);
			}
		}

		@Override
		public int size() {
			int size = 0;
			final Iterator<List<String>> i = iterator();
			while (i.hasNext()) {
				size++;
				i.next();
			}
			return size;
		}

		public static class Reader {
			static Collection<List<String>> readSentenceCollection(
					String fileName) {
				return new SentenceCollection(fileName);
			}
		}

		static class SentenceIterator implements Iterator<List<String>> {

			BufferedReader reader;

			public SentenceIterator(BufferedReader reader) {
				this.reader = reader;
			}

			@Override
			public boolean hasNext() {
				try {
					return reader.ready();
				} catch (final IOException e) {
					return false;
				}
			}

			@Override
			public List<String> next() {
				try {
					final String line = reader.readLine();
					final String[] words = line.split("\\s+");
					final List<String> sentence = new ArrayList<String>();
					for (int i = 0; i < words.length; i++) {
						final String word = words[i];
						sentence.add(word.toLowerCase());
					}
					return sentence;
				} catch (final IOException e) {
					throw new NoSuchElementException();
				}
			}

			@Override
			public void remove() {
				throw new UnsupportedOperationException();
			}
		}

	}
}
