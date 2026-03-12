import sys, os, time, unittest

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')

from itergen.main import IterGen
from itergen import Grammar
from itergen import parsers

class TestBasic(unittest.TestCase):
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
    
    def test_grammar(self):
        inp = "Hello, world. This is a test.\nThe quick brown fox jumps over the lazy dog."
        grammar = Grammar(self.essay_grammar())
        parser = parsers.create_base_parser(grammar)
        tree = parser.parse(inp)
        print(tree.pretty())
    
    def test_grammar_simple(self):
        grammar = """
            start: (WORD | EMAIL)*
            WORD: /[^ ]+/
            EMAIL: /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/
            %import common.WS
            %ignore WS
        """
        inc_parser = parsers.create_parser(Grammar(grammar))
        partial_code = f"""Predicates:\nPer something random s;;; ::: x is a person dependent on caffeine. sds@gmail.com ssdsd 1234"""
        r = inc_parser.get_acceptable_next_terminals(partial_code)
        print(r)
    
    def test_next(self):
        inp = "The quick brown fox jumps over the lazy dog."
        iter_gen = IterGen(grammar=self.essay_grammar(), model_id="microsoft/Phi-3-mini-128k-instruct")

        print('Prompt:', inp)

        iter_gen.start(inp)
        output = iter_gen.forward(unit='token', num=20)
        print(output)

    def test_next_sentence(self):
        inp = "My name is John. I am a software engineer. I work at Microsoft."
        iter_gen = IterGen(grammar=self.essay_grammar(), model_id="microsoft/Phi-3-mini-128k-instruct")

        print('Prompt:', inp)

        iter_gen.start(inp)
        
        # Generate 2 sentences
        output = iter_gen.forward(unit='sentence', num=2)

        print(output[0])
        assert sum([1 for c in output[0] if c=='.']) == 2
    
    def test_next_sentence_batch(self):
        inp = "My name is John. I am a software engineer. I work at Microsoft."
        iter_gen = IterGen(grammar=self.essay_grammar(), model_id="microsoft/Phi-3-mini-128k-instruct", num_return_sequences=5, do_sample=True, temperature=0.9)
        print('Prompt:', inp)

        iter_gen.start(inp)
        
        # Generate 2 sentences
        output = iter_gen.forward(unit='sentence', num=2)

        print(output)
        # assert sum([1 for c in output[0] if c=='.']) == 2

    # def test_temp(self):
    #     import torch
    #     inp = "My name is John. I am a software engineer. I work at Microsoft."
    #     iter_gen = IterGen(grammar=self.essay_grammar(), model_id="microsoft/Phi-3-mini-128k-instruct", num_return_sequences=5, do_sample=True, temperature=0.9)
    #     inp = iter_gen.tokenizer(inp, return_tensors='pt').to(iter_gen.device)
    #     input_ids = inp['input_ids']
    #     # Concatenate input_ids with 10 pad tokens
    #     # Can we use position_ids to make the semantics same?
    #     input_ids = torch.cat([torch.full((1, 100), iter_gen.tokenizer.pad_token_id).to(iter_gen.device), input_ids], dim=1)
    #     attention_mask = (input_ids != iter_gen.tokenizer.pad_token_id)
    #     # attention_mask = None
    #     out = iter_gen.model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=20)
    #     out_str = iter_gen.tokenizer.decode(out[0], skip_special_tokens=True)
    #     print(out_str)

    def test_next_sentence2(self):
        inp = "The quick brown fox jumps over the lazy dog."
        iter_gen = IterGen(grammar=self.essay_grammar(), model_id="microsoft/Phi-3-mini-128k-instruct")

        print('Prompt:', inp)

        iter_gen.start(inp)
        
        # Generate 2 sentences
        output = iter_gen.forward(unit='sentence', num=2)

        # Generate 2 sentences again
        output = iter_gen.forward(unit='sentence', num=2)
    
        # Output should contain 4 sentences
        print(output[0])
        assert sum([1 for c in output[0] if c=='.']) == 4
    
    def test_next_prev_sentence(self):
        inp = "The quick brown fox jumps over the lazy dog."
        iter_gen = IterGen(grammar=self.essay_grammar(), model_id="microsoft/Phi-3-mini-128k-instruct")

        print('Prompt:', inp)

        iter_gen.start(inp)
        
        # Generate 2 sentences
        _ = iter_gen.forward(unit='sentence', num=2)

        # Go back by 1 sentence
        output = iter_gen.backward(unit='sentence', num=1)
    
        # Output should contain  sentences
        print(output[0])
        assert sum([1 for c in output[0] if c=='.']) == 1
    
    def test_prev_token(self):
        inp = "The quick brown fox jumps over the lazy dog."
        iter_gen = IterGen(grammar=self.essay_grammar(), model_id="microsoft/Phi-3-mini-128k-instruct")

        print('Prompt:', inp)

        iter_gen.start(inp)
        
        # Generate 2 sentences
        _ = iter_gen.forward(unit='sentence', num=2)

        # Go back by 1 token
        output = iter_gen.backward(num=1)
    
        # Output should contain  sentences
        print(output[0])
        assert sum([1 for c in output[0] if c=='.']) == 1
        assert output[0].endswith('dog')


    def test_next_prev_next_sentence(self):
        inp = "The quick brown fox jumps over the lazy dog."
        iter_gen = IterGen(grammar=self.essay_grammar(), model_id="microsoft/Phi-3-mini-128k-instruct")

        print('Prompt:', inp)

        iter_gen.start(inp)
        
        # Generate 2 sentences
        _ = iter_gen.forward(unit='sentence', num=2)

        # Go back by 1 sentence
        _ = iter_gen.backward(unit='sentence', num=1)

        # Generate 2 sentences
        output = iter_gen.forward(unit='sentence', num=2)

        # Output should contain  sentences
        print(output[0])
        assert sum([1 for c in output[0] if c=='.']) == 3
    
    def test_mixed1(self):
        inp = "My name is John. I am a software engineer. I work at Microsoft."
        iter_gen = IterGen(grammar=self.essay_grammar(), model_id="microsoft/Phi-3-mini-128k-instruct")

        print('Prompt:', inp)

        iter_gen.start(inp)
        
        # Generate 2 sentences
        output1 = iter_gen.forward(unit='sentence', num=2)
        print(output1)
        output2 = iter_gen.backward(unit='word', num=3)
        print(output2)
        output3 = iter_gen.forward(unit='sentence', num=1)
        print(output3)
        output4 = iter_gen.forward(unit='word', num=5)
        print(output4)
        
        assert sum([1 for c in output4[0] if c=='.']) == 2
        # Check 5 words in last sentence of output4
        assert len(output4[0].split('.')[-1].split()) == 5
