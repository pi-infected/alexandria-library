import config

config.delete_indexes = [
  'currency_ss_taxonomy0_ss', 'currency_ss', 'money_ss',
  'currency_ss_matchtext_ss', 'currency_ss_uri_ss',
  'currency_ss_preflabel_and_uri_ss'
]

class delete_tags(object):
  def process(self, parameters = dict(), data = dict()):
    for index in config.delete_indexes:
      if index in data:
        del data[index]

    return parameters, data
