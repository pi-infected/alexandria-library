class enhance_debug(object):
  def process(self, parameters = dict(), data = dict()):
    import pdb; pdb.set_trace()
    return parameters, data
