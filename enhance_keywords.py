import etl
import wordfreq
import advertools as adv
import re
import numpy
import config
import itertools
from transformers             import pipeline
from collections              import defaultdict, Counter
from stanza.pipeline.core     import Pipeline as StanzaPipeline
from stanza                   import download as stanza_download
from copy                     import deepcopy
from sklearn.metrics.pairwise import cosine_similarity
from math                     import log
from os                       import path
from sentence_transformers    import SentenceTransformer
from nltk.corpus              import stopwords as nltk_stopwords


# ------------------------MONKEY-PACHING--------------------------------
def patched_iter_wordlist(lang, wordlist = 'best', reverse = False):
  is_reversed = wordfreq.reversed_list[lang]
  freq_list   = wordfreq.get_frequency_list(lang, wordlist)

  if is_reversed != reverse:
    freq_list.reverse()
    wordfreq.reversed_list[lang] = reverse

  return itertools.chain(*freq_list)


def patched_top_n_list(lang, n, wordlist = 'best', ascii_only = False, reverse = False):
  results = []
  for word in wordfreq.iter_wordlist(lang, wordlist, reverse):
    if (not ascii_only) or max(word) <= '~':
      results.append(word)
      if len(results) >= n:
        break

  return results


wordfreq.reversed_list = defaultdict(lambda: False)
wordfreq.iter_wordlist = patched_iter_wordlist
wordfreq.top_n_list    = patched_top_n_list
# ----------------------------------------------------------------------


class KeyBasedDefaultDict(defaultdict):
  def __missing__(self, key):
    if self.default_factory is None:
      raise KeyError(key)

    self[key] = self.default_factory(key)
    return self[key]


config.script_dir                      = path.dirname(path.realpath(__file__))
config.translators                     = defaultdict(lambda: defaultdict(lambda: None))
config.taggers                         = dict()
config.embedding_similarity_model      = None
config.embedding_similarity_model_path = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
config.accepted_upos                   = set(['NOUN', 'PROPN'])
config.most_frequent_words             = dict()
config.least_frequent_words            = dict()


def init_stopwords(lang_code):
  most_frequent_words = wordfreq.top_n_list(lang_code, 500)
  language = config.code_to_language[lang_code]

  _stopwords = deepcopy(adv.stopwords[language])
  _stopwords.update(most_frequent_words)
  _stopwords.update(nltk_stopwords.words(language))

  if lang_code != 'en':
    _stopwords.update(adv.stopwords['english'])
    _stopwords.update(wordfreq.top_n_list('en', 500))
    _stopwords.update(nltk_stopwords.words('english'))

  return _stopwords


config.stopwords = KeyBasedDefaultDict(init_stopwords)


def get_most_frequent_word(lang_code):
  if lang_code not in config.most_frequent_words:
    mf_word      = wordfreq.top_n_list(lang_code, 1)[0]
    mf_word_freq = wordfreq.word_frequency(mf_word, lang_code)
    config.most_frequent_words[lang_code] = mf_word_freq

  return config.most_frequent_words[lang_code]


def get_least_frequent_word(lang_code):
  if lang_code not in config.least_frequent_words:
    lf_word      = wordfreq.top_n_list(lang_code, 1, reverse = True)[0]
    lf_word_freq = wordfreq.word_frequency(lf_word, lang_code)
    config.least_frequent_words[lang_code] = lf_word_freq

  return config.least_frequent_words[lang_code]


def get_inverse_www_normalized_frequency(lang_codes, word):
  mf_freq = get_most_frequent_word('en')
  w_freq  = wordfreq.word_frequency(word, 'en')
  if w_freq == 0.0:
    w_freq = get_least_frequent_word('en')

  en_nf = w_freq / mf_freq

  if len(lang_codes) == 1 and 'en' in lang_codes:
    return 1.0 / en_nf

  max_nf = en_nf
  for lang_code in lang_codes:
    if lang_code == 'en':
      continue

    mf_freq = get_most_frequent_word(lang_code)
    w_freq  = wordfreq.word_frequency(word, lang_code)
    if w_freq == 0.0:
      w_freq = get_least_frequent_word(lang_code)

    lang_nf = w_freq / mf_freq
    if lang_nf > max_nf:
      max_nf = lang_nf

  return 1.0 / max_nf


def get_normalized_tf_iwwwf_weight(lang_codes, words_frequencies, mf_term, terms):
  tf_iwwwf_weights = []
  for term in terms:
    tf_iwwwf_weights.append((words_frequencies[term] / mf_term) * log(get_inverse_www_normalized_frequency(lang_codes, term)))

  return max(tf_iwwwf_weights)


def init_keywords_dict():
  return {
    'words':      set(),
    'lang_codes': set()
  }


def init_ngram_lists(lang_code, n_gram):
  ngram_keyword = [''    for _ in range(n_gram)]
  ngram_words   = [set() for _ in range(n_gram)]
  ngram_langs   = [set() for _ in range(n_gram)]
  ngram_langs[0].add(lang_code)
  return ngram_keyword, ngram_words, ngram_langs


def clean_keyword(word):
  word.text  = re.sub(r'[!?\.:;,<>\\\/\+\*]+', '', word.text)
  word.text  = re.sub(r'\-+$', '',                 word.text)
  word.text  = re.sub(r'\s+', ' ',                 word.text)
  word.text  = word.text.strip()
  word.lemma = re.sub(r'[!?\.:;,<>\\\/\+\*]+', '', word.lemma)
  word.lemma = re.sub(r'\-+$', '',                 word.lemma)
  word.lemma = re.sub(r'\s+', ' ',                 word.lemma)
  word.lemma = word.lemma.strip()


def validate_keyword(word, stopwords):
  if word.lemma in stopwords or word.text in stopwords:
    return False
  if len(word.text) < 3:
    return False

  return True


def rotate_list(new_elem, _list):
  _list.pop()
  _list.insert(0, new_elem)


def update_keywords(keywords, ngram_keyword, ngram_words, ngram_langs, lang_code, n_gram):
  for n in range(n_gram):
    if len(ngram_keyword[n]) == 0:
      break

    keyword = ''
    words   = set()
    langs   = set()
    count   = 0

    for k in range(n, n_gram):
      if len(ngram_keyword[k]) == 0:
        break

      if len(keyword) > 0:
        keyword += ' '

      keyword += ngram_keyword[k]
      words.update(ngram_words[k])
      langs.update(ngram_langs[k])
      count += 1

    k_dict = keywords[count - 1][keyword]
    k_dict['words'].update(words)
    k_dict['lang_codes'].update(langs)

  rotate_list('', ngram_keyword)
  rotate_list(set(), ngram_words)
  rotate_list(set([lang_code]), ngram_langs)


def extract_tagged_text_keywords(tagged_text, lang_code, n_gram):
  keywords      = [defaultdict(init_keywords_dict) for _ in range(n_gram)]
  words_counter = Counter()
  stopwords     = config.stopwords[lang_code]

  for tagged_sentence in tagged_text.sentences:
    ngram_keyword, ngram_words, ngram_langs = init_ngram_lists(lang_code, n_gram)
    for token in tagged_sentence.tokens:
      for word in token.words:
        clean_keyword(word)
        words_counter[word.lemma] += 1

      ner_tag = token.ner[0]
      if ner_tag == 'O':
        for word in token.words:
          pos_tag = word.upos
          if pos_tag in config.accepted_upos:
            if len(ngram_keyword[0]) > 0:
              ngram_keyword[0] += ' '
            if validate_keyword(word, stopwords):
              ngram_keyword[0] += word.lemma
              ngram_words[0].add(word.lemma)
          elif pos_tag == 'PUNCT' and len(ngram_keyword[0]) > 0:
            update_keywords(keywords, ngram_keyword, ngram_words, ngram_langs, lang_code, n_gram)

        if len(ngram_keyword[0]) > 0:
          update_keywords(keywords, ngram_keyword, ngram_words, ngram_langs, lang_code, n_gram)

      elif ner_tag == 'B' or ner_tag == 'I':
        for word in token.words:
          if len(ngram_keyword[0]) > 0:
            ngram_keyword[0] += ' '
          if validate_keyword(word, stopwords):
            ngram_keyword[0] += word.lemma
            ngram_words[0].add(word.lemma)

      elif ner_tag == 'E' or ner_tag == 'S':
        for word in token.words:
          if len(ngram_keyword[0]) > 0:
            ngram_keyword[0] += ' '
          if validate_keyword(word, stopwords):
            ngram_keyword[0] += word.lemma
            ngram_words[0].add(word.lemma)

        update_keywords(keywords, ngram_keyword, ngram_words, ngram_langs, lang_code, n_gram)

  return keywords, words_counter


def init_embedding_similarity_model():
  if config.embedding_similarity_model is None:
    config.embedding_similarity_model = SentenceTransformer(config.embedding_similarity_model_path)


def merge_keywords_results(global_keywords, global_words, local_keywords, local_words):
  global_words.update(local_words)

  for ngram in range(len(local_keywords)):
    ngram_data = local_keywords[ngram]
    if ngram in global_keywords:
      gngram_data = global_keywords[ngram]
      for keyword, keyword_data in ngram_data.items():
        if keyword in gngram_data:
          gkeyword_data = global_keywords[keyword]
          gkeyword_data['words'].update(keyword_data['words'])
          gkeyword_data['lang_codes'].update(keyword_data['lang_codes'])
        else:
          gngram_data[keyword] = keyword_data
    else:
      global_keywords[ngram] = ngram_data


def filter_top_keywords(
  ranked_keywords,
  top_rank_fraction = None,
  top_rank_min      = None,
  top_rank_max      = None
):
  filtered_keywords = []
  if top_rank_fraction is None or top_rank_min is None or top_rank_max is None:
    for ngram in range(len(ranked_keywords)):
      keywords = ranked_keywords[ngram]
      for keyword, _ in keywords:
        filtered_keywords.append(keyword)
  else:
    count = 0
    for ngram in range(len(ranked_keywords)):
      count += len(ranked_keywords[ngram])

    k = int(count * top_rank_fraction)
    if k < top_rank_min:
      k = top_rank_min
    if k > top_rank_max:
      k = top_rank_max

    for ngram in range(len(ranked_keywords)):
      keywords = ranked_keywords[ngram]
      i       = 0
      local_k = int(k / float(len(ranked_keywords)))
      for keyword, _ in keywords:
        filtered_keywords.append(keyword)
        i += 1
        if i == local_k:
          break

  return filtered_keywords


def rank_keywords_from_texts(
  texts, tagged_texts, lang_codes,
  n_gram              = 2,
  top_rank_fraction   = 0.2,
  top_rank_min        = 20,
  top_rank_max        = 100,
  custom_text_weights = None
):
  if texts is None or tagged_texts is None or lang_codes is None:
    return None
  if len(texts) == 0 or len(tagged_texts) == 0 or len(lang_codes) == 0:
    return None
  if len(texts) != len(tagged_texts) or len(texts) != len(lang_codes):
    return None
  if custom_text_weights is not None and len(custom_text_weights) != len(texts):
    return None

  init_embedding_similarity_model()

  text_factors  = []
  keywords      = [defaultdict(init_keywords_dict) for _ in range(n_gram)]
  words_counter = Counter()
  text_max_size = 0

  for text, tagged_text, lang_code in zip(texts, tagged_texts, lang_codes):
    text_factors.append(len(text))
    if len(text) > text_max_size:
      text_max_size = len(text)

    text_keywords, text_words_counter = extract_tagged_text_keywords(tagged_text, lang_code, n_gram)
    merge_keywords_results(keywords, words_counter, text_keywords, text_words_counter)

  mf_term = 0
  for _, val in words_counter.items():
    if val > mf_term:
      mf_term = val

  mf_term = float(mf_term)
  for key in words_counter.keys():
    words_counter[key] /= mf_term

  if custom_text_weights is None:
    for i in range(len(text_factors)):
      text_factors[i] = (text_factors[i] / float(text_max_size))
  else:
    for i in range(len(text_factors)):
      text_factors[i] = (text_factors[i] / float(text_max_size)) * custom_text_weights[i]

  text_factors       = numpy.array(text_factors)
  keywords_lists     = [list(n_keywords.keys()) for n_keywords in keywords]
  text_embeddings    = config.embedding_similarity_model.encode(texts)
  keyword_embeddings = [config.embedding_similarity_model.encode(n_keywords_list) for n_keywords_list in keywords_lists]
  sim_matrices       = [cosine_similarity(n_keyword_embeddings, text_embeddings) for n_keyword_embeddings in keyword_embeddings]

  text_count      = len(texts)
  ranked_keywords = list()
  for n in range(n_gram):
    sim_matrix = sim_matrices[n]
    ranked_keywords.append(list())
    index = 0
    for texts_sim in sim_matrix:
      mean_sim   = numpy.multiply(texts_sim, text_factors).sum() / text_count
      keyword    = keywords_lists[n][index]
      terms      = keywords[n][keyword]['words']
      lang_codes = keywords[n][keyword]['lang_codes']
      ranked_keywords[n].append([keyword, mean_sim * get_normalized_tf_iwwwf_weight(lang_codes, words_counter, mf_term, terms)])
      index += 1

    ranked_keywords[n].sort(key = lambda x: -x[1])

  return filter_top_keywords(ranked_keywords, top_rank_fraction, top_rank_min, top_rank_max), filter_top_keywords(ranked_keywords)


def load_tagger(lang_code):
  if lang_code not in config.taggers:
    if not path.exists(f'/home/infected/stanza_resources/{lang_code}/default.zip'):
      stanza_download(lang_code)

    config.taggers[lang_code] = StanzaPipeline(
      lang       = lang_code,
      processors = 'tokenize,mwt,pos,lemma,ner',
      use_gpu    = False,
      verbose    = False
    )

  return config.taggers[lang_code]


def lemmatize_str(in_str, lang_code):
  tagger = load_tagger(lang_code)
  doc    = tagger(in_str)
  return doc


class enhance_keywords(object):
  def process(self, parameters = dict(), data = dict()):
    analyse_fields = [
      'title_txt', 'content_txt', 'description_txt', 'ocr_t', 'ocr_descew_t'
    ]

    text = ''
    for field in analyse_fields:
      if field in data:
        text = "{}{}\n".format(text, data[field])

    lang_code   = data['language_s']
    tagged_text = lemmatize_str(text, lang_code)
    keywords, tag_set = rank_keywords_from_texts(
      [text], [tagged_text], [lang_code], n_gram = 1
    )

    data['tag_set']    = tag_set
    data['hashtag_ss'] = keywords

    return parameters, data
