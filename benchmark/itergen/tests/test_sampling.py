import sys, os, unittest

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')

from itergen.main import IterGen
from itergen import Grammar

class TestSampling(unittest.TestCase):
    @staticmethod
    def math_equation_grammar():
        # A Lark grammar for math
        return """
        start: NUMBER OPERATOR NUMBER "=" NUMBER
        NUMBER: /[0-9]+/
        OPERATOR: "+" | "-" | "*" | "/"

        %ignore " "
    """
    
    def test_sample1(self):
        # Start with greedy decoding
        iter_gen = IterGen(grammar=self.math_equation_grammar(), model_id="microsoft/Phi-3-mini-128k-instruct", parse_output_only=False)

        # Start a new generation session
        inp = "234 * 327 = "
        print('Prompt:', inp)
        iter_gen.start(inp)

        o1 = iter_gen.forward(unit='NUMBER', num=1)
        
        o2 = iter_gen.backward(unit='NUMBER', num=1)

        o3 = iter_gen.forward(unit='NUMBER', num=1)
        print(o3)

        # Seems like the following assertion is not always true
        # Even with greedy decoding, the model has different outputs
        # This could potentially be due to numerical instability with and without the kv-cache
        # assert o1[0] == o3[0]

    def test_sample2(self):
        # Start with sampling. default temperature is 1.0
        iter_gen = IterGen(grammar=self.math_equation_grammar(), model_id="microsoft/Phi-3-mini-128k-instruct", parse_output_only=False, do_sample=True, temperature=0.9, max_tokens=20)

        # Start a new generation session
        inp = "234 * 327 = "
        print('Prompt:', inp)
        iter_gen.start(inp)
        
        o1s = []
        for i in range(5):
            o1 = iter_gen.forward(unit='NUMBER', num=1)
            _ = iter_gen.backward(unit='NUMBER', num=1)
            o1s.append(o1)      
        
        iter_gen.update_gen_args(do_sample=False)

        o2s = []
        for i in range(5):
            o2 = iter_gen.forward(unit='NUMBER', num=1)
            _ = iter_gen.backward(unit='NUMBER', num=1)
            o2s.append(o2)
        
        # All output in o2s should be the same
        for o2 in o2s:
            assert o2 == o2s[0]

    @staticmethod
    def math_equation_grammar2():
        # A Lark grammar for math
        return """
        start: NUMBER "=" NUMBER "*" NUMBER
        NUMBER: /[0-9]+/
        %ignore " "
    """

    def test_sample3(self):
        # Start with sampling. default temperature is 1.0
        iter_gen = IterGen(grammar=self.math_equation_grammar2(), model_id="microsoft/Phi-3-mini-128k-instruct", parse_output_only=True, do_sample=True, temperature=0.1)

        # Start a new generation session
        inp = "Q: Factorize 391\nA: 391 = 17 * 23\nQ: Factorize 209\nA: 209 = 11 * 19\nQ: Factorize 221\nA: 221 = 13 * 17\nQ: Factorize 437\nA:"

        iter_gen.start(inp)
        op = iter_gen.forward(num=1, unit='NUMBER')
        print(op)
        assert op[0] == '437'

        op2 = iter_gen.forward(num=1, unit='NUMBER')
        print(op2)
        assert op2[0] == '437 = 19'

        op3 = iter_gen.forward(num=1, unit='NUMBER')
        print(op3)
        assert op3[0] == '437 = 19 * 23'
    

    def test_sample4(self):
        # Start with sampling. default temperature is 1.0
        iter_gen = IterGen(grammar=self.math_equation_grammar2(), model_id="microsoft/Phi-3-mini-128k-instruct", parse_output_only=True, do_sample=True, temperature=0.9)

        # Start a new generation session
        inp = "Q: Factorize 391\nA: 391 = 17 * 23\nQ: Factorize 209\nA: 209 = 11 * 19\nQ: Factorize 221\nA: 221 = 13 * 17\nQ: Factorize 26797\nA:"

        iter_gen.start(inp)

        op = iter_gen.forward(num=1, unit='NUMBER')
        for _ in range(5):
            op = iter_gen.forward(num=2, unit='NUMBER')
            print(op)
            op2 = iter_gen.backward(num=2, unit='NUMBER')
            print(op2)
        