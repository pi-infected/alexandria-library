import config
import re
from copy import deepcopy

config.delete_indexes = [
  r'Chroma_.*', r'Component_.*', r'Compression_.*', r'Chroma_.*',
  r'Content-Type-Parser-Override', r'Data_.*', r'Dimension_.*', r'Exif_.*',
  r'Data_.*', r'File_.*', r'ICC_.*', r'IHDR', r'Image_.*', r'Number_.*',
  r'Resolution_Units', r'Thumbnail_.*', r'Transparency_Alpha', r'Version',
  r'.*_Resolution', r'access_.*', r'embeddedResourceType', r'height', r'pHYs',
  r'pdf_has.*', r'pdf_unmappedUnicodeCharsPerPage', r'resourceName', r'tiff_.*',
  r'width', r'xmpTPg_NPages'
]

class clean_tika_output_no_ocr(object):
  def process(self, parameters = dict(), data = dict()):
    text = data['content_txt']
    text = re.sub(r'\s*\[Image \(no OCR yet\)\s*\]', '', text)

    for index in deepcopy(data.keys()):
      for regex in config.delete_indexes:
        if re.match(regex, index):
          del data[index]
          break

    return parameters, data
