# Summary

UD Coptic contains manually annotated Sahidic Coptic texts, including Biblical texts, sermons, letters, and hagiography.

# Introduction

The Coptic Universal Dependency Treebank is a manually annotated corpus of Sahidic Coptic texts, currently containing excerpts from the Sahidic New Testament Gospel of Mark, Works by Archmandrite Shenoute of Atripe, the Letters of Besa, lives of Sts. Cyrus and Onnophrius, Epistle of Pseudo-Ephrem, the Dormition of John the Apostle and short stories from the Apophthegmata Patrum (Sayings of the Desert Fathers). Detailed information about the treebank is available here:

http://copticscriptorium.org/treebank.html

The data was digitized and annotated manually for part of speech in the project Coptic Scriptorium. For individual credit and further information see:

http://copticscriptorium.org/

Coptic POS tags come from the Coptic Scriptorium tag set, which is available from the project and treebank websites.

# Further details

## Basic statistics
|      subcorpus           |        documents            | tokens  |
| ------------------------ | --------------------------- | ------- |
| Not Because a Fox Barks  | MONB XH204-216              |   2,547 |
| Abraham our Father       | MONB XL93-94, YA518-520     |   1,197 |
| Acephalous Work 22       | MONB YA421-428              |   1,698 |
| I See Your Eagerness     | MONB GF31-32                |     439 |
| Epistle of Pseudo-Ephrem | psephrem.letter             |   1,925 |
| Gospel of Mark           | Chapters 1 - 9              |  10,812 |
| 1 Corinthians            | Chapters 1 - 6              |   3,570 |
| Book of Ruth             | Chapters 1 - 4 (complete)   |   3,470 |
| Letters of Besa          | #1,2,13,15,25               |   3,939 |
| Life of Cyrus            | life.cyrus.01               |   1,962 |
| Life of Onnophrius       | life.onnophrius.01          |   2,745 |
| Apophthegmata Patrum     | #1-6,18-19,23-32,114-139    |   4,152 |
| Martyrdom of St. Victor  | Chapters 1 - 6              |   1,985 |
| Dormition of John        | dormition.john.mercad       |   3,064 |
| Pseudo-Athanasius        | mercy_judgment              |   2,782 |
| Proclus Homilies         | #13 On Easter               |   2,344 |
|                          | Total:                      |  48,631 |

## Tokenization

Coptic was originally written in scriptio continua, without spaces, and modern conventions fuse multiple tokens into so-called bound groups, collapsing clitic pronouns, prepositions and other morphemes into single orthographic units.

The Coptic Treebank now annotates these bound groups as 'multi-unit' tokens. However, morphological units below the POS level, including affixes and fused compounds, are now annotated in the treebank in the MISC column, using an attribute Morphs=A-B-C, where A, B and C are constituent morphemes of a complex word. There is also a further attribute in the MISC column, called Orig, which appears whenever normalization has taken place and renders the word form as it appeared in the original manuscript. This can include removal of optional diacritics and contracted forms of nomina sacra, which appear expanded in the word form column.

Note that for some fused forms carrying multiple parts of speech, the native Coptic POS tag set assigns portmanteau tags, such as APST_PPERS (auxiliary, past, fused with a subject personal pronoun). In the UD guidelines for Coptic, these forms are tolerated by always selecting the argument as the function-determining unit. Thus APST_PPERS, the past auxiliary with fused pronoun, is attached as *nsubj*, ignoring the *aux* dependency. In a future version, we are considering integrating a more subtle analysis using enhanced dependencies.

For more information on Coptic tokenization, see the Coptic Scriptorium website.

# Acknowledgments

The underlying POS tagged material was produced as part of the projects Coptic Scriptorium, KOMeT and KELLIA, funded by the NEH in the USA and BMBF and DFG in Germany (see http://copticscriptorium.org/ for more details). Treebank annotation was done mainly by Mitchell Abrams, Liz Davidson and Amir Zeldes. Thanks are also due to Israel Avrahamy, Asael Benyami, Yinon Kahan and Oran Szachter for their contributions.

# Data Splits

Dataset splits attempt to balance genres across all sets, as well as preserve contiguous documents whenever possible. Sentence and document IDs in the data indicate the respective source texts. Sentences are not shuffled.

# References

To cite the treebank please refer to the following paper:

  * Zeldes, Amir & Abrams, Mitchell (2018). "The Coptic Universal Dependency Treebank". In: *Proceedings of the Universal Dependencies Workshop 2018*. Brussels, Belgium, 192-201.

```
@InProceedings{ZeldesAbrams2018,
  author    = {Amir Zeldes and Mitchell Abrams},
  title     = {The {C}optic {U}niversal {D}ependency {T}reebank},
  booktitle = {Proceedings of the Universal Dependencies Workshop 2018},
  pages     = {192--201},
  year      = {2018},
  address   = {Brussels}
}
```

Further information on relevant annotation standards and NLP tools used prior to manual correction can be found in these papers:

  * Zeldes, Amir & Schroeder, Caroline T. (2016a). SCRIPTORIUM Part-of-Speech Tagsets for Sahidic Coptic. Georgetown University and University of the Pacific, Technical Report.
  * Zeldes, Amir & Schroeder, Caroline T. (2016b). "An NLP Pipeline for Coptic". In: Proceedings of LaTeCH 2016 - The 10th SIGHUM Workshop at the Annual Meeting of the ACL. Berlin, 146-155.

# Changelog

  * CHANGELOG 2.7 -> 2.8

Added `Foreign` feature and `OrigLang` annotation in MISC for loanwords

  * CHANGELOG 2.6 -> 2.7

Added Proclus of Cyzicus Homily 13, On Easter. Nested named and non-named entity annotations have been added to the entire corpus.

  * CHANGELOG 2.5 -> 2.6

Added Discourse of Pseudo-Athanasius on Mercy and Judgment, numerous corrections and consistency checks to (especially auxiliary) lemmas

  * CHANGELOG 2.4 -> 2.5

Added Besa's Letters 1-2 (On Vigilance, Exhortations), Excerpt from Shenoute's I See Your Eagerness, Life of Cyrus, Life of Onnophrius, Epistle of Pseudo-Ephrem, Dormition of John and Apophthegmata Patrum 114-139, numerous corrections and stabilized splits. 

The corpus is now larger than 30K word forms, and train/dev/test splits should now be stable, with only the train partition growing.

  * CHANGELOG 2.3 -> 2.4

Added Mark 7-9. Rearranged documents in train/dev/test so that documents and parts of larger works are contiguous and genere balance is maintained. Negative polarity added for negative auxiliaries, deprel obl:npmod added for non-prepositional adverbial NPs (formerly part of advmod).

  * CHANGELOG 2.2 -> 2.3

Added new documents: Shenoute's Acephalous Work 22 (YA421-428) and Abraham our Father (XL93-94, YA518-520), the Martyrdom of St. Victor Chapters 1-6, Apophthegmata Patrum 27-32, 1 Corinthians Chapters 3-6 (corpus now includes 1-6) and Mark 6 (coprus now includes 1-6). Rearranged documents in train/dev/test so that all documents are contiguous and genere balance is improved (all sets now include different Shenoute texts). 

Also added MISC attributes: Morphs (for sub-word morphological segmentation) and Orig (records unnormalized word form where normalization has been carried out)

  * CHANGELOG 2.1 -> 2.2

Repository renamed from UD_Coptic to UD_Coptic-Scriptorium. Added 1 Corinthians 1-3, rearranged material from Mark to make contiguous documents in train/dev/test.

  * CHANGELOG 2.0 -> 2.1

Added Apophthegmata Patrum (AP) 1-4 to training data and moved AP24 to dev and AP26 to test to create contiguous document partitions for the AP portions.

  * CHANGELOG 1.4 -> 2.0

Switched to UD v2 and added bound group information (multiword super-tokens). Added Gospel of Mark chapters 3-5 to previously available dev data, and numerous error corrections.

  * CHANGELOG 1.3 -> 1.4

First inclusion in full release as of v1.4. Added Gospel of Mark chapter 2 to previously available dev data, and numerous error corrections.

=== Machine readable metadata ==============

Documentation status: complete
Data source: manual
Data available since: UD v1.4
License: CC BY 4.0
Includes text: yes
Lemmas: manual native
UPOS: converted from manual
XPOS: manual native
Features: automatic
Relations: manual native
Genre: bible fiction nonfiction
Contributors: Abrams, Mitchell; Davidson, Elizabeth; Zeldes, Amir
Contributing: elsewhere
Contact: amir.zeldes@georgetown.edu

============================================