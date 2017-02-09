package com.support.automate.jarvis;

import java.text.SimpleDateFormat;
import java.util.List;
import java.util.Properties;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreAnnotations.NamedEntityTagAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;

public class TestNlp {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		String inputText = "Could you please let us know when the file abcd.txt will be delivered and GP-40-1-FB will be posted";

		// Next we generate an annotation object that we will use to annotate
		// the text with
		SimpleDateFormat formatter = new SimpleDateFormat("yyyy-MM-dd");
		String currentTime = formatter.format(System.currentTimeMillis());

		Properties props = new Properties();

		props.put("annotators", "tokenize, ssplit, pos, lemma, ner, parse, dcoref");

		StanfordCoreNLP pipeLine = new StanfordCoreNLP(props);

		Annotation document = new Annotation(inputText);

		//document.set(CoreAnnotations.DocDateAnnotation.class, currentTime);

		// Finally we use the pipeline to annotate the document we created
		pipeLine.annotate(document);

		/*
		 * now that we have the document (wrapping our inputText) annotated we
		 * can extract the annotated sentences from it, Annotated sentences are
		 * represent by a CoreMap Object
		 */
		List<CoreMap> sentences = document.get(SentencesAnnotation.class);

		for (CoreMap sentence : sentences) {
			for (CoreLabel token : sentence.get(TokensAnnotation.class)) {
				// Using the CoreLabel object we can start retrieving NLP
				// annotation data
				// Extracting the Text Entity
				String text = token.getString(TextAnnotation.class);

				// Extracting Name Entity Recognition
				String ner = token.getString(NamedEntityTagAnnotation.class);

				// Extracting Part Of Speech
				String pos = token.get(CoreAnnotations.PartOfSpeechAnnotation.class);

				// Extracting the Lemma
				String lemma = token.get(CoreAnnotations.LemmaAnnotation.class);
				System.out.println("text=" + text + ";NER=" + ner + ";POS=" + pos + ";LEMMA=" + lemma);

				/*
				 * There are more annotation that are available for extracting
				 * (depending on which "annotators" you initiated with the
				 * pipeline properties", examine the token, sentence and
				 * document objects to find any relevant annotation you might
				 * need
				 */
			}
		}
	}

}
