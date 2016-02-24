package nlp.assignments;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import nlp.langmodel.LanguageModel;
import nlp.util.Counter;

/**
 * Vanilla uni-gram language model.
 */
class EmpiricalUnigramLanguageModel implements LanguageModel {

	static final String	STOP		= "</S>";
	static final String	UNKNOWN		= "*UNKNOWN*";

	Counter<String>		wordCounter	= new Counter<String>();

	public EmpiricalUnigramLanguageModel(
			Collection<List<String>> sentenceCollection) {
		for (final List<String> sentence : sentenceCollection) {
			final List<String> stoppedSentence = new ArrayList<String>(
					sentence);
			stoppedSentence.add(STOP);
			for (final String word : stoppedSentence) {
				wordCounter.incrementCount(word, 1.0);
			}
		}
		wordCounter.incrementCount(UNKNOWN, 1.0);
		wordCounter.normalize();
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

	@Override
	public double getSentenceProbability(List<String> sentence) {
		final List<String> stoppedSentence = new ArrayList<String>(sentence);
		stoppedSentence.add(STOP);
		double probability = 1.0;
		for (int index = 0; index < stoppedSentence.size(); index++) {
			probability *= getWordProbability(stoppedSentence, index);
		}
		return probability;
	}

	public double getWordProbability(List<String> sentence, int index) {
		final String word = sentence.get(index);
		final double count = wordCounter.getCount(word);
		if (count == 0) {
			return wordCounter.getCount(UNKNOWN);
		}
		return count;
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
