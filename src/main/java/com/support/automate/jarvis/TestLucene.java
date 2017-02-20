package com.support.automate.jarvis;

import java.io.IOException;
import java.io.StringReader;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.core.StopAnalyzer;
import org.apache.lucene.analysis.tokenattributes.TermAttribute;
import org.apache.lucene.util.Version;

public class TestLucene {
	public static final String CONTENTS = "contents";
	public static final String FILE_NAME = "filename";
	public static final String FILE_PATH = "filepath";
	public static final int MAX_SEARCH = 10;

	public static void main(String[] args) {
		TestLucene tester;

		tester = new TestLucene();

		try {
			tester.displayTokenUsingStopAnalyzer();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private void displayTokenUsingStopAnalyzer() throws IOException {
		String text = "Lucene is simple yet powerful java based search library.";
		Analyzer analyzer = new StopAnalyzer(Version.LUCENE_36);
		TokenStream tokenStream = analyzer.tokenStream(CONTENTS, new StringReader(text));
		TermAttribute term = tokenStream.addAttribute(TermAttribute.class);
		while (tokenStream.incrementToken()) {
			System.out.print("[" + term.term() + "] ");
		}
	}
}
