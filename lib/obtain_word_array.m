function [words] = obtain_word_array(web_text)
%obtain_word_array Get the words in the description (plus some additional
%processing).

words = tokenizedDocument(web_text);
words = erasePunctuation(words);
words = words.Vocabulary;
end

