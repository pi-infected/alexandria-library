import config
from functools import partial

config.filter_indexes = [
  'organization_ss', 'person_ss', 'product_ss', 'event_ss', 'work_of_art_ss',
  'location_ss'
]

def is_in_tag_set(tag, tag_set):
  if tag in tag_set:
    return True
  else:
    return False


class filter_nlp_tags(object):
  def process(self, parameters = dict(), data = dict()):
    tag_set = set(data['tag_set'])

    for index in config.filter_indexes:
      if index in data:
        data[index] = list(filter(partial(is_in_tag_set, tag_set = tag_set), data[index]))

    return parameters, data
