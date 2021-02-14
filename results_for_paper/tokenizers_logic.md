# Tokenizer pipelines

* word :  lower -> sent_tokenize -> word_tokenize -> token.isalpha() and token not in stopwords

* word_without_stop_words : lower -> sent_tokenize -> word_tokenize -> token.isalpha()

* word_without_stop_words_and_not_alpha :  lower -> sent_tokenize -> word_tokenize

* word_with_simple_split: split on whitespaces, fullstops, new lines etc

[sent_tokenize](https://www.kite.com/python/docs/nltk.sent_tokenize)

[word_tokenize](https://www.kite.com/python/docs/nltk.tokenize.word_tokenize)
