#!/usr/bin/env ruby

filepath   = ARGV[0]
processors = [
  'enhance_mapping_id',
  'filter_blacklist',
  'filter_file_not_modified',
  'enhance_extract_text_tika_server',
  'enhance_file_mtime',
  'enhance_path',
  'clean_title',
  'clean_tika_output_no_ocr',
  'enhance_language_detect',
  'enhance_multilingual',
  'enhance_ner_spacy',
  'enhance_extract_and_validate_emails',
  'enhance_extract_law',
  'enhance_extract_and_validate_phones',
  'enhance_file_size',
  'enhance_mimetype',
  'enhance_keywords',
  'filter_nlp_tags',
  'enhance_entity_linking',
  'enhance_annotations',
  'delete_tags',
  # 'enhance_debug'
]

system("(sudo ./install_plugin.sh); (yes | etl-delete -e); etl-enrich -p \"#{processors.join(',')}\" #{filepath};")
