package com.support.automate.jarvis;

import simplenlg.framework.*;
import simplenlg.lexicon.*;
import simplenlg.realiser.english.*;
import simplenlg.phrasespec.*;
import simplenlg.features.*;

public class SimpleNLG {

	public static void main(String[] args) {
		Lexicon lexicon = Lexicon.getDefaultLexicon();
		NLGFactory nlgFactory = new NLGFactory(lexicon);
		Realiser realiser = new Realiser(lexicon);
		
		NLGElement s1 = nlgFactory.createSentence("are you serious");
		System.out.println(realiser.realiseSentence(s1));
		
		SPhraseSpec p = nlgFactory.createClause();
	    p.setSubject("Mary");
	    p.setVerb("chase");
	    p.setObject("monkey");
	   // p.setFeature(Feature.TENSE, Tense.PAST);
	    p.setFeature(Feature.INTERROGATIVE_TYPE, InterrogativeType.YES_NO);
	    p.addComplement("despite her exhaustion"); // Prepositional phrase, string
	    p.addComplement("very quickly"); // Adverb phrase, passed as a string
	    
	    System.out.println(realiser.realiseSentence(p));
	    
	    NPPhraseSpec subject = nlgFactory.createNounPhrase("Mary");
	    NPPhraseSpec object = nlgFactory.createNounPhrase("the monkey");  
	    VPPhraseSpec verb = nlgFactory.createVerbPhrase("chase");
	    
	    SPhraseSpec p1 = nlgFactory.createClause();
	    
	    subject.addModifier("fast");
	    verb.addModifier("quickly");
	    p1.setSubject(subject);
	    p1.setObject(object);
	    p1.setVerb(verb);
	    
	    String output3 = realiser.realiseSentence(p1); // Realiser created earlier.
	    System.out.println(output3);
	}

}
