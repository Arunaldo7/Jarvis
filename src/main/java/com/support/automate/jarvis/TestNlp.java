package com.support.automate.jarvis;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
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

	String[] verbPos = { "RB", "VB" };
	String[] nouns = { "JJ", "NN" };

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		// String inputText = "Could you please let us know when the file
		// abcd.txt will be delivered and GP-40-1-FB will be posted";

		/*
		 * String inputText = "";
		 * 
		 * FileReader inputFile = new
		 * FileReader("src/test/resources/sample-content.txt"); BufferedReader
		 * reader = new BufferedReader(inputFile); String line =
		 * reader.readLine();
		 * 
		 * while (line != null) { inputText += line; line = reader.readLine(); }
		 */
		Properties props = new Properties();

		props.put("annotators", "tokenize, ssplit, pos, lemma, ner, parse, dcoref");

		String entity = "BIDNOFP15";
		
		SimpleDateFormat dateFormat = new SimpleDateFormat("dd-MM-yyyy");
        @SuppressWarnings("deprecation")
		String date = dateFormat.format( new Date("15-JAN-2017"));

		
		String issue = "File fbsi-oracle-line-item was very large to send through Axway. So BIDNOFP15 failed";
		List<String> issueList = new ArrayList<String>();
		processSentence(props, issue, issueList);
		
		String issueTags = String.join(";", issueList);
		
		String solution = "The file was resent successfully using ftp process";
		List<String> solnList = new ArrayList<String>();
		processSentence(props, solution, solnList);
		
		String solnTags = String.join(";", solnList);
		
		String user = "A558985";
		
		NLPService nlpService = new NLPService();
		nlpService.insertValues(entity, issueTags, solnTags, "weekly", user);
	}

	private static void processSentence(Properties props, String input, List<String> listText) {
		Annotation document = new Annotation(input);

		// Finally we use the pipeline to annotate the document we created
		StanfordCoreNLP pipeLine = new StanfordCoreNLP(props);
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
				// System.out.println("text=" + text + ";NER=" + ner + ";POS=" +
				// pos + ";LEMMA=" + lemma);
				if (!ner.equalsIgnoreCase("DATE")) {
					if (pos.startsWith("RB") || pos.startsWith("VB") || pos.startsWith("NN") || pos.startsWith("JJ")) {
						if (!listText.contains(text)) {
							listText.add(text);
						}
					}
				}

			}
		}
	}

}
