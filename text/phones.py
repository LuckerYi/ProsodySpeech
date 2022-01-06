import os
import json

class Phones():
    def __init__(self, phone_set_file):
        with open(phone_set_file, 'r', encoding="utf-8") as fi:
            phone_to_id = json.load(fi)
        _pad = '_'
        symbols = [_pad] + phone_to_id
        self._symbol_to_id = {s: i for i, s in enumerate(symbols)}
        self._id_to_symbol = {i: s for i, s in enumerate(symbols)}
    
    def text_to_sequence(self, text, cleaner_names=None):
        phone_list = text.strip().split()
        return self._symbols_to_sequence(phone_list)

    def _symbols_to_sequence(self, phone_list):
        
        return [self._symbol_to_id[s] for s in phone_list if self._should_keep_symbol(s)]
    
    def _should_keep_symbol(self, s):
        if s not in self._symbol_to_id and s is not '_' and s is not '.':
            print("Warning: {} not in phone set".format(s))
        return s in self._symbol_to_id and s is not '_' and s is not '.'