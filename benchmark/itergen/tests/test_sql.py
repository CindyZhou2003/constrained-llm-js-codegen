import sys, os, unittest

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')

from itergen.main import IterGen
from itergen import Grammar
from itergen import parsers

class TestSQL(unittest.TestCase):    
    def test_sql1(self):
        prompt = 'db_id: concert_singer\ndb_info: # stadium ( stadium_id , location , name , capacity , highest , lowest , average )\n# singer ( singer_id , name , country , song_name , song_release_year , age , is_male )\n# concert ( concert_id , concert_name , theme , stadium_id , year )\n# singer_in_concert ( concert_id , singer_id )\n# concert.stadium_id = stadium.stadium_id\n# singer_in_concert.singer_id = singer.singer_id\n# singer_in_concert.concert_id = concert.concert_id\n\nquestion: What are the names of all stadiums that did not have a concert in 2014?\nSQL:'

        # Start with greedy decoding
        iter_gen = IterGen(grammar='sql', model_id="meta-llama/Llama-2-7b-chat-hf", parse_output_only=True, device='cuda:1')

        # Start a new generation session
        print('Prompt:', prompt)
        iter_gen.start(prompt)

        o1 = iter_gen.forward()
        print(o1)

    def test_sql2_parser(self):
        """
        Test for the parser symbol_pos_map
        """
        gen = '\nSELECT singer.name, singer.country, singer.age\nFROM singer\nJOIN concert_singer'
        grammar = Grammar('sql')
        ip = parsers.create_parser(grammar)
        r = ip.get_acceptable_next_terminals(gen)
        pos = ip.symbol_pos_map.get_symbol_pos_all('table_name')[0] 
        assert gen[pos[0]:pos[1]+1] == 'singer'
    