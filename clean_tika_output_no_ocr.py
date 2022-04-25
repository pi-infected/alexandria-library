import config
import re

config.delete_regexes = [
  r'Chroma.*', r'Component.*', r'Compression.*', r'Chroma.*',
  r'Content-Type-Parser-Override.*', r'Data.*', r'Dimension.*', r'Exif.*',
  r'File.*', r'ICC.*', r'IHDR.*', r'Image.*', r'Number.*',
  r'Resolution.Units', r'Thumbnail.*', r'Transparency.Alpha', r'Version.*',
  r'.*Resolution', r'access.*', r'embeddedResourceType.*', r'height.*', r'pHYs.*',
  r'pdf.has.*', r'pdf.unmappedUnicodeCharsPerPage.*', r'resourceName.*', r'tiff.*',
  r'width.*', r'xmpTPg.NPages.*'
]

class clean_tika_output_no_ocr(object):
  def process(self, parameters = dict(), data = dict()):
    text = data['content_txt']
    text = re.sub(r'\s*\[Image \(no OCR yet\)\s*\]', '', text)
    data['content_txt'] = text

    index_list = list()
    for index in data.keys():
      index_list.append(index)

    for index in index_list:
      for regex in config.delete_regexes:
        if re.match(regex, index):
          del data[index]
          break

    return parameters, data
