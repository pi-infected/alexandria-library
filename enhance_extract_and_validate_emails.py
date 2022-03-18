import re
import etl_plugin_core
from email_validator import validate_email, EmailNotValidError


class enhance_extract_and_validate_email(object):
  def process(self, parameters = {}, data = {}):
    text = etl_plugin_core.get_text(data = data)

    for match in re.finditer('[\w\.-]+@[\w\.-]+', text, re.IGNORECASE):
      email = match.group(0)

      try:
        valid = validate_email(email, check_deliverability = False)
        email = valid.email
        etl_plugin_core.append(data, 'email_ss', email)
      except EmailNotValidError as e:
        pass

    if 'email_ss' in data:
      for match in re.finditer('From: (.* )?([\w\.-]+@[\w\.-]+)', text, re.IGNORECASE):
        value = match.group(2)
        etl_plugin_core.append(data, 'Message-From_ss', value)

      # extract email adresses (to)
      for match in re.finditer('To: (.* )?([\w\.-]+@[\w\.-]+)', text, re.IGNORECASE):
        value = match.group(2)
        etl_plugin_core.append(data, 'Message-To_ss', value)

      # extract the domain part from all emailadresses to facet email domains
      data['email_domain_ss'] = []
      emails = data['email_ss']
      if not isinstance(emails, list):
        emails = [emails]

      for email in emails:
        domain = email.split('@')[1]
        etl_plugin_core.append(data, 'email_domain_ss', domain)

    return parameters, data
