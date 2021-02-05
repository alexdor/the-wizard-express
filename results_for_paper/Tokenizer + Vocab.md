# Tokenizer + Vocab

## 1. Tokenizer

### 1.1 Word level tokenizer

Steps:

* Build vocab (see next section)
* Passes things on to Bert tokenizer (next section)

### 1.2 Word level Bert tokenizer

Steps:

* Gets a vocab
* Adds special Bert tokens to the vocab (CLS, UNK, PAD, SEP, MASK)
* Normalize unicode characters
* Lowcase everything
* Split on whitespace
* Adds special tokens to each incoming question

## 2. Build vocab

Steps:

### 2.1. Sentence tokenizer

Split paragraph into sentences. A sentence tokenizer uses an unsupervised algorithm to build a model for abbreviation words, collocations, and words that start sentences; and then uses that model to find sentence boundaries. This approach has been shown to work well for many European languages.

PUNCTUATION = (';', ':', ',', '.', '!', '?')

### 2.2. Word tokenizer

The Treebank tokenizer uses regular expressions to tokenize text as in Penn Treebank.  It assumes that the text has already been segmented into sentences, e.g. using `sent_tokenize()`.

This tokenizer performs the following steps:

* split standard contractions, e.g. `don't` -> `do n't` and `they'll` -> `they 'll`
* treat most punctuation characters as separate tokens
* split off commas and single quotes, when followed by whitespace
* separate periods that appear at the end of line

### 2.3. Drop strings that aren't words or they are stop words

### 2.4. Keep the 8000 most common words (or how many we've selected in the config)

## 3. Example of tokenizers

### Word tokenizer

Question: "What causes precipitation to fall?"
Token ids: [8002, 8000, 1725, 2343, 8000, 8000, 8001]
Tokens: ['[CLS]', '[UNK]', 'causes', 'precipitation', '[UNK]', '[UNK]', '[SEP]']

Question: What headdress does the Canadian Army wear?
Token ids: [8002, 8000, 8000, 8000, 8000, 828, 98, 8000, 8001]
Tokens: ['[CLS]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', 'canadian', 'army', '[UNK]', '[SEP]']

### Wordpiece tokenizer

Question: "What causes precipitation to fall?"
Token ids: [101, 1327, 4680, 14886, 1106, 2303, 136, 102]
Tokens: ['[CLS]', 'What', 'causes', 'precipitation', 'to', 'fall', '?', '[SEP]']

Question: What headdress does the Canadian Army wear?
Token ids: [101, 1327, 1246, 18380, 1674, 1103, 2122, 1740, 4330, 136, 102]
Tokens: ['[CLS]', 'What', 'head', '##dress', 'does', 'the', 'Canadian', 'Army', 'wear', '?', '[SEP]']

## 4. Example of vocab

### Word vocab

```json
'per':165
'councils':4712
'flame':5166
'viceroy':5878
'sunni':4889
'allow':827
'holy':1090
'corporation':1436
'spielberg':1983
'pharmacists':5564
'drew':3196
'agency':1355
'creek':5270
'edinburgh':4505
'away':701
'peaceful':4079
'gold':1016
'landing':1916
'macintosh':2613
'entity':3878
'bachelor':4221
'bid':5228
'conscription':7248
'holders':7437
'someone':3742
'habitat':3139
'broadway':3787
'caliphate':5035
'texts':1299
'attained':6262
```

Wordpiece vocab:

```json
'##dorf':10414
'horizontal':10012
'##ches':7486
'robots':16013
'compares':26153
'##natch':25703
'Quaker':20536
'contacts':10492
'##gles':15657
'Wolves':16484
'Maritime':11081
'Cyrus':17093
'Dorset':16180
'Slow':19826
'twist':11079
'gone':2065
'Ultimate':11618
'Destiny':16784
'##rya':20845
'Process':18821
'makers':12525
'recorder':18898
'dams':20245
'Seven':5334
'medium':5143
'##box':8757
'Experience':15843
'Beyond':8270
'Laboratory':8891
'Jesus':3766
'boots':6954
'awhile':21985
'shrub':16965
'##は':28809
'excluded':12728
'rhythms':21890
'ლ':701
'Di':12120
'[unused97]':97
'fragile':14434
'grass':5282
'Salvatore':18746
'##ungen':25821
'appreciate':8856
'barrels':14619
'1796':14342
'speaker':7349
'Somalia':15350
'Maia':26685
'Protestant':7999
'Justice':3302
'remained':1915
'Record':7992
'pickup':17257
'pads':21910
'grounded':18395
```
