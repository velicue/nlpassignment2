package nlp.assignments;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import nlp.langmodel.LanguageModel;
import nlp.util.Counter;
import nlp.util.CounterMap;

/**
 * Katz-backoff++ bigram language model. A Katz model with practical
 * improvements. If yuo are working with this model, try to understand what it
 * does and why!
 */
class KatzPPBigramLanguageModel implements LanguageModel {

	static final int			cutOff					= 5;
	static final String			START					= "<S>";
	static final String			STOP					= "</S>";
	static final String			UNKNOWN					= "*UNKNOWN*";

	Counter<String>				backoffs				= new Counter<String>();
	CounterMap<String, String>	bigramCounter			= new CounterMap<String, String>();
	Counter<String>				discountedBigramCounter	= new Counter<String>();
	Counter<String>				discountedWordCounter	= new Counter<String>();
	Counter<String>				probabilities			= new Counter<String>();
	Counter<String>				wordCounter				= new Counter<String>();

	public KatzPPBigramLanguageModel(
			Collection<List<String>> sentenceCollection) {
		for (final List<String> sentence : sentenceCollection) {
			final List<String> stoppedSentence = new ArrayList<String>(
					sentence);
			stoppedSentence.add(0, START);
			stoppedSentence.add(STOP);
			String previousWord = stoppedSentence.get(0);
			for (int i = 1; i < stoppedSentence.size(); i++) {
				final String word = stoppedSentence.get(i);
				wordCounter.incrementCount(word, 1.0);
				discountedWordCounter.incrementCount(word, 1.0);
				discountedBigramCounter
						.incrementCount(previousWord + " " + word, 1.0);
				bigramCounter.incrementCount(previousWord, word, 1.0);
				previousWord = word;
			}
		}
		normalizeDistributions();
	}

	private static void verifyProbability(double prob) {
		if (Double.isNaN(prob) || Double.isInfinite(prob) || prob < 0) {
			throw new IllegalStateException("Invalid probability: " + prob);
		}
	}

	@Override
	public List<String> generateSentence() {
		System.out.println("WARNING -- DUMMY PLACEHOLDER IMPLEMENTATION");
		final List<String> sentence = new ArrayList<String>();
		String word = generateWord();
		while (!word.equals(STOP)) {
			sentence.add(word);
			word = generateWord();
		}
		return sentence;
	}

	public double getBigramProbability(String previousWord, String word) {
		final double bigramProbability = probabilities
				.getCount(previousWord + " " + word);
		verifyProbability(bigramProbability);

		if (bigramProbability != 0) {
			return bigramProbability;
		}

		double unigramProbability = probabilities.getCount(word);
		if (unigramProbability == 0) {
			// System.out.println("UNKNOWN Word: " + word);
			unigramProbability = probabilities.getCount(UNKNOWN);
		}
		verifyProbability(unigramProbability);

		double backoff = backoffs.getCount(previousWord);
		if (backoff == 0.0) {
			if (probabilities.getCount(previousWord) == 0) {
				backoff = 1.0;
			}
		}
		return unigramProbability * backoff;
	}

	@Override
	public double getSentenceProbability(List<String> sentence) {
		final List<String> stoppedSentence = new ArrayList<String>(sentence);
		stoppedSentence.add(0, START);
		stoppedSentence.add(STOP);
		double probability = 1.0;
		String previousWord = stoppedSentence.get(0);
		for (int i = 1; i < stoppedSentence.size(); i++) {
			final String word = stoppedSentence.get(i);
			probability *= getBigramProbability(previousWord, word);
			previousWord = word;
		}
		return probability;
	}

	private void normalizeDistributions() {
		final double[] unigramBuckets = new double[cutOff + 2];
		for (final String word : wordCounter.keySet()) {
			final double count = wordCounter.getCount(word);
			if (count <= cutOff + 1) {
				unigramBuckets[(int) count]++;
			}
		}

		final double[] bigramBuckets = new double[cutOff + 2];
		for (final String previousWord : bigramCounter.keySet()) {
			final Counter<String> currentCounter = bigramCounter
					.getCounter(previousWord);
			for (final String word : currentCounter.keySet()) {
				final double count = currentCounter.getCount(word);
				if (count <= cutOff + 1) {
					bigramBuckets[(int) count]++;
				}
			}
		}

		double normalizer = 1.0 / wordCounter.totalCount();
		double A = (cutOff + 1) * unigramBuckets[cutOff + 1]
				/ unigramBuckets[1];
		for (final String word : wordCounter.keySet()) {
			final double count = wordCounter.getCount(word);
			if (count > cutOff) {
				probabilities.setCount(word, count * normalizer);
			} else {
				final double discountedCount = (count + 1)
						* unigramBuckets[(int) count + 1]
						/ unigramBuckets[(int) count];
				final double probability = count * normalizer
						* (discountedCount / count - A) / (1 - A);
				probabilities.setCount(word, probability);
				verifyProbability(probability);
			}
		}
		probabilities.setCount(UNKNOWN, unigramBuckets[1] * normalizer);

		A = (cutOff + 1) * bigramBuckets[cutOff + 1] / bigramBuckets[1];
		final Counter<String> forwardProbability = new Counter<String>();
		final Counter<String> backwardProbability = new Counter<String>();
		for (final String previousWord : bigramCounter.keySet()) {
			final Counter<String> currentCounter = bigramCounter
					.getCounter(previousWord);
			normalizer = 1.0 / currentCounter.totalCount();
			double probability = 0;
			double probabilitySoFar = 0;
			for (final String word : currentCounter.keySet()) {
				final double count = currentCounter.getCount(word);
				if (count > cutOff) {
					probability = count * normalizer;
				} else {
					final double discountedCount = (count + 1)
							* bigramBuckets[(int) count + 1]
							/ bigramBuckets[(int) count];
					probability = count * normalizer
							* (discountedCount / count - A) / (1 - A);
				}
				verifyProbability(probability);
				probabilities.setCount(previousWord + " " + word, probability);
				backwardProbability.incrementCount(previousWord,
						probabilities.getCount(word));
				probabilitySoFar += probability;
			}
			forwardProbability.setCount(previousWord, probabilitySoFar);
		}

		for (final String word : wordCounter.keySet()) {
			final double backoff = (1.0 - forwardProbability.getCount(word))
					/ (1.0 - backwardProbability.getCount(word));
			// Verify back-off.
			if (Double.isNaN(backoff) || Double.isInfinite(backoff)) {
				System.err.println("stop");
			}
			backoffs.setCount(word, backoff);
		}
	}

	String generateWord() {
		final double sample = Math.random();
		double sum = 0.0;
		for (final String word : wordCounter.keySet()) {
			sum += wordCounter.getCount(word);
			if (sum > sample) {
				return word;
			}
		}
		return UNKNOWN;
	}
}
