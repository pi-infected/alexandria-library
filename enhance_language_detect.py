import config
import pycountry
from lingua import Language, LanguageDetectorBuilder

config.language_to_code = dict()
config.code_to_language = dict()
config.languages        = [Language.ENGLISH, Language.FRENCH]
config.lang_detector    = LanguageDetectorBuilder.from_languages(*config.languages).build()
config.language_list    = [
  'arabic', 'danish', 'dutch', 'english', 'finnish', 'french', 'german',
  'hungarian', 'italian', 'norwegian', 'portuguese', 'romanian', 'russian',
  'spanish', 'swedish', 'FRENCH', 'ENGLISH'
]

for language in config.language_list:
  pyc_lang = pycountry.languages.get(name = language.lower())
  if pyc_lang is None:
    print(f'unmatched language {language}')
  else:
    lang = pyc_lang.alpha_2
    if lang is not None:
      config.code_to_language[lang]     = language.lower()
      config.language_to_code[language] = lang


class enhance_language_detect(object):
  def process(self, parameters = dict(), data = dict()):
    analyse_fields = [
      'title_txt', 'content_txt', 'description_txt', 'ocr_t', 'ocr_descew_t'
    ]

    text = ''
    for field in analyse_fields:
      if field in data:
        text = "{}{}\n".format(text, data[field])

    confidence_values  = config.lang_detector.compute_language_confidence_values(text)
    data['language_s'] = config.language_to_code[confidence_values[0][0].name]

    return parameters, data
