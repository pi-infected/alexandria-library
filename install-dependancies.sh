sudo apt install python3 python3-pip python-is-python3 ruby ruby-dev
sudo pip uninstall django
sudo pip install 'django==3.2.12' wordfreq advertools numpy transformers stanza sklearn sentence_transformers nltk phonenumbers email-validator python-magic lingua-language-detector
sudo pip install rdflib -U
wget https://opensemanticsearch.org/download/open-semantic-search_22.03.04.deb
sudo apt install ./open-semantic-search_22.03.04.deb

# solr-data path ex: /media/user/data/solr
# chmod +x /media
# chmod +x /media/user
# chmod +x /media/user/data
# chmod +x /media/user/data/solr
# ln -s /media/user/data/solr /var/solr/data
# chown solr:solr -R /var/solr/data
