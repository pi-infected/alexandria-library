import re
import etl_plugin_core
import phonenumbers
from phonenumbers.phonenumberutil import NumberParseException

def normalize_phonenumber(phone):
  chars = ['+','0','1','2','3','4','5','6','7','8','9']
  phone_normalized = ''
  for char in phone:
    if char in chars:
      # only first +
      if char == '+':
        if not phone_normalized:
          phone_normalized = '+'
      else:
        phone_normalized += char

  return phone_normalized


def validate_phone_number(num):
  try:
    parsed_num = phonenumbers.parse(num, None)
    if phonenumbers.is_valid_number(parsed_num):
      return phonenumbers.format_number(parsed_num, phonenumbers.PhoneNumberFormat.E164)
    else:
      return None
  except NumberParseException:
    return None

class enhance_extract_and_validate_phones(object):
    def process(self, parameters = {}, data = {}):
      text = etl_plugin_core.get_text(data=data)

      for match in re.finditer('[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', text, re.IGNORECASE):
        phone = match.group(0)
        phone_normalized = validate_phone_number(phone)

        if phone_normalized is None:
          phone_normalized = normalize_phonenumber(phone)
          phone_normalized = validate_phone_number(phone_normalized)

        if phone_normalized is not None:
          etl_plugin_core.append(data, 'phone_ss', phone)
          etl_plugin_core.append(data, 'phone_normalized_ss', phone_normalized)

      return parameters, data
