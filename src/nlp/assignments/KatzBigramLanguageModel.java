package nlp.assignments;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import nlp.langmodel.LanguageModel;
import nlp.util.Counter;
import nlp.util.CounterMap;

/**
 * A vanilla Katz Bigram Language model
 */
public class KatzBigramLanguageModel implements LanguageModel {

	static final double			beta					= 0.1;
	static final String			START					= "<S>";
	static final String			STOP					= "</S>";
	static final String			UNKNOWN					= "*UNKNOWN*";

	Counter<String>				alpha					= new Counter<String>();
	CounterMap<String, String>	bigramCounter			= new CounterMap<String, String>();
	Counter<String>				discountedWordCounter	= new Counter<String>();
	Counter<String>				wordCounter				= new Counter<String>();
	Counter<String>				z						= new Counter<String>();

	public KatzBigramLanguageModel(
			Collection<List<String>> sentenceCollection) {
		for (final List<String> sentence : sentenceCollection) {
			final List<String> stoppedSentence = new ArrayList<String>(
					sentence);
			stoppedSentence.add(0, START);
			stoppedSentence.add(STOP);
			wordCounter.incrementCount(START, 1.0);
			String previousWord = stoppedSentence.get(0);
			for (int i = 1; i < stoppedSentence.size(); i++) {
				final String word = stoppedSentence.get(i);
				wordCounter.incrementCount(word, 1.0);
				bigramCounter.incrementCount(previousWord, word, 1.0);
				discountedWordCounter.incrementCount(word, 1.0);
				previousWord = word;
			}
		}
		normalizeDistributions();
	}

	@Override
	public List<String> generateSentence() {
		final List<String> sentence = new ArrayList<String>();
		String word = generateWord(START);
		while (!word.equals(STOP)) {
			sentence.add(word);
			word = generateWord(word);
		}
		return sentence;
	}

	/**
	 * If c(u,v) > 0 then results are given by:
	 * prob(v|u) = c*(u,v)/c(u)
	 * else prob(v|u) = alpha(u) * [c(v)]/\sum_{v'; c(u,v') = 0} c(v')]
	 * where c*(u,v) = c(u,v) - beta
	 * alpha(u) = 1 - \sum_v c*(u,v)/c(u)
	 *
	 * the counts c(u) are computed using 1-smoothing.
	 */
	public double getBigramProbability(String previousWord, String word) {

		if (this.wordCounter.getCount(word) == 0) {
			word = UNKNOWN;
		}

		if (this.wordCounter.getCount(previousWord) == 0) {
			previousWord = UNKNOWN;
		}

		final Counter<String> keys = this.bigramCounter
				.getCounter(previousWord);
		final double probability;
		if (keys.getCount(word) > 0) {
			probability = (keys.getCount(word) - beta) / keys.totalCount();
		} else {
			probability = this.alpha.getCount(previousWord)
					* this.wordCounter.getCount(word)
					/ this.z.getCount(previousWord);
		}

		if (probability > 1 || probability <= 0 || Double.isNaN(probability)) {
			System.err.println("Wrong probabilities. Found " + probability);
		}

		return probability;
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

	public void normalizeDistributions() {
		// Do 1-smoothing for unknown words
		this.wordCounter.incrementCount(UNKNOWN, 1);
		for (final String previousWord : this.wordCounter.keySet()) {
			this.wordCounter.incrementCount(previousWord, 1);
		}

		// alpha, z
		for (final String previousWord : this.wordCounter.keySet()) {
			double sum = 0;
			final Counter<String> ctr = this.bigramCounter
					.getCounter(previousWord);
			for (final String word : ctr.keySet()) {
				if (ctr.getCount(word) > 0) {
					sum = sum + ctr.getCount(word) - beta;
				}
			}
			sum = sum / this.wordCounter.getCount(previousWord);
			this.alpha.incrementCount(previousWord, 1 - sum);

			double zCount = 0;
			for (final String word : this.wordCounter.keySet()) {
				if (ctr.getCount(word) == 0) {
					zCount = zCount + this.wordCounter.getCount(word);
				}
			}
			this.z.incrementCount(previousWord, zCount);
		}
	}

	String generateWord(String previousWord) {
		final double sample = Math.random();
		double sum = 0.0;
		for (final String word : wordCounter.keySet()) {
			sum += this.getBigramProbability(previousWord, word);
			if (sum > sample) {
				return word;
			}
		}
		return UNKNOWN;
	}
}
