import sys, os, unittest

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')

from itergen.main import IterGen
from itergen import Grammar
from itergen import parsers

class TestSymbolMap(unittest.TestCase):
    @staticmethod
    def essay_grammar():
        # A Lark grammar for paragraphs in text
        return """
        start: paragraph+
        ?paragraph: sentence+
        ?sentence: word+ punctuation
        word: /[a-zA-Z0-9]+/ | COMMA | SINGLE_QUOTE | ESCAPED_DOUBLE_QUOTE
        punctuation: /[.!?]/
        COMMA: ","
        SINGLE_QUOTE: "'"
        ESCAPED_DOUBLE_QUOTE: "\\\""

        %import common.WS
        %ignore WS
    """
    
    def test_symbol_map1(self):
        inp = "Hello, world. This is a test.\nThe quick brown fox jumps over the lazy dog."
        grammar = Grammar(self.essay_grammar())
        inc_parser = parsers.create_parser(grammar)
        r = inc_parser.get_acceptable_next_terminals(inp)

        print(inc_parser.symbol_pos_map)


    def test_view_sentences(self):
        inp = "My name is John. I am a software engineer. I work at Microsoft."
        iter_gen = IterGen(grammar=self.essay_grammar(), model_id="microsoft/Phi-3-mini-128k-instruct")

        print('Prompt:', inp)
        iter_gen.start(inp)
        
        # Generate 4 sentences
        _ = iter_gen.forward(unit='sentence', num=4)

        sentences = iter_gen.view('sentence')
        assert len(sentences[0]) == 4
        print(sentences)
