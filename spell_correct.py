import requests
import os

API_KEY = os.environ['BING_SEARCH_API_KEY']
ENDPOINT = "https://api.bing.microsoft.com/v7.0/search"

class SpellChecker:
    def __init__(self, api_key, endpoint):
        self.api_key = api_key
        self.endpoint = endpoint
        self.text = None
        self.corrected_text = None

    def set_text(self, text):
        self.text = text

    def set_corrected_text(self, corrected_text):
        self.corrected_text = corrected_text

    def get_response(self, text):
        params = {'q':text, 'mode':'proof', 'responseFilter':'SpellSuggestions'}
        headers = {'Ocp-Apim-Subscription-Key': self.api_key}
        response = requests.get(self.endpoint, headers=headers, params=params).json()
        return response
    
    def spell_check(self, text):
            self.set_text(text)
            response = self.get_response(text)
            corrected_text = response['queryContext']['alteredQuery']
            self.set_corrected_text(corrected_text)
            return corrected_text

    def __call__(self, text):
        return self.spell_check(text)
    
    def __repr__(self):
        if not self.text:
            return 'No text entered for correction'
        else:
            return f'Text: {self.text}\nCorrected Text: {self.corrected_text}'


example_text = "Hollo, wrld! I am back!"
spell_checker = SpellChecker(API_KEY, ENDPOINT)
print(spell_checker(example_text))

    