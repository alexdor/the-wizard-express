# Example of tf-idf sparse matrix

## Example of words in document

**Vocab size**: 8000

**Dataset**: Squad

**Document index**: 4

**Actual document**: '"Bairn" and "hyem", meaning "child" and "home", respectively, are examples of Geordie words with origins in Scandinavia; barn and hjem are the corresponding modern Norwegian and Danish words. Some words used in the Geordie dialect are used elsewhere in the Northern United Kingdom. The words "bonny" (meaning "pretty"), "howay" ("come on"), "stot" ("bounce") and "hadaway" ("go away" or "you\'re kidding"), all appear to be used in Scots; "aye" ("yes") and "nowt" (IPA://naʊt/, rhymes with out,"nothing") are used elsewhere in Northern England. Many words, however, appear to be used exclusively in Newcastle and the surrounding area, such as "Canny" (a versatile word meaning "good", "nice" or "very"), "hacky" ("dirty"), "netty" ("toilet"), "hoy" ("throw", from the Dutch gooien, via West Frisian), "hockle" ("spit").'

**Word indexes for non-zero tf-idf**: [5, 7, 13, 56, 80, 184, 445, 513, 535, 556, 689, 733, 1186, 1368, 1418, 1800, 2046, 2446, 2863, 3086, 6050]

**Words**: ['many', 'used', 'united', 'west', 'modern', 'northern', 'word', 'meaning', 'dutch', 'words', 'via', 'examples', 'appear', 'surrounding', 'dialect','newcastle','origins','elsewhere','exclusively','corresponding', 'danish']

**tf-df for the above words** : [24.94308943089431, 91.26016260162602, 16.056910569105693, 6.455284552845529, 8.040650406504065, 10.032520325203253, 2.59349593495935, 4.634146341463415, 1.4796747967479675, 5.414634146341464, 2.3170731707317076, 2.1463414634146343, 3.284552845528456, 1.5528455284552847, 0.7317073170731708, 0.48780487804878053, 0.7886178861788619, 0.861788617886179, 0.6829268292682927, 0.6016260162601627, 0.22764227642276424]

## Example of documents including a word

**Vocab size**: 8000

**Dataset**: Squad

**Word index**: 2580

**Word**: file

**Document indexes**: [96, 324, 576, 918, 1206, 1335, 1508, 2393, 2432, 2686, 3204, 3577, 3834, 3836, 3861, 5118, 5633, 6082, 6628, 6714, 7010, 7546, 7653, 7789, 8323, 8651, 9019, 9478, 9820, 9885, 10346, 10569, 11244, 11571, 11676, 12595, 12833, 12861, 12908, 12946, 15324, 16902, 17436, 17647, 17732, 18214, 18215, 18513, 18891, 19403, 19472, 19884, 19896, 19997, 20409, 20638, 20643, 20685, 20783]

**Tf-idf per document**: [1.9666666666666666, 0.9672131147540984, 1.156862745098039, 0.329608938547486, 1.10625, 1.7878787878787878, 0.355421686746988, 1.0172413793103448, 0.595959595959596, 1.372093023255814, 0.6704545454545455, 0.33146067415730335, 1.063063063063063, 0.27314814814814814, 0.7195121951219512, 0.6020408163265306, 0.37579617834394907, 0.46825396825396826, 0.3259668508287293, 0.34104046242774566, 0.4609375, 0.5784313725490196, 1.1132075471698113, 1.0172413793103448, 0.5728155339805825, 1.372093023255814, 0.595959595959596, 0.29207920792079206, 0.4836065573770492]
