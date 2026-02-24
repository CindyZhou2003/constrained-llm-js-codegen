from lark import Lark, Transformer, v_args

# The refined grammar for a flexible Vega-Lite specification
vega_lite_grammar = r"""
    start: specification

    specification: "{" pair ("," pair)* "}"

    pair: schema_property
        | data_property
        | mark_property
        | encoding_property
        | other_property

    schema_property: "\"$schema\"" ":" string
    data_property: "\"data\"" ":" "{" data_url_property "}"
    mark_property: "\"mark\"" ":" mark_value
    encoding_property: "\"encoding\"" ":" "{" encoding_pairs "}"
    
    other_property: key ":" value
    key: string

    data_url_property: "\"url\"" ":" string

    mark_value: string
              | "{" mark_type_property ("," mark_option_pair)* "}"

    mark_type_property: "\"type\"" ":" MARK_TYPE

    mark_option_pair: string ":" value

    encoding_pairs: encoding_pair ("," encoding_pair)*
    encoding_pair: string ":" encoding_value

    encoding_value: object | string

    MARK_TYPE.2: "\"bar\"" 
             | "\"circle\"" 
             | "\"square\"" 
             | "\"tick\"" 
             | "\"line\"" 
             | "\"area\"" 
             | "\"point\"" 
             | "\"rule\"" 
             | "\"geoshape\"" 
             | "\"text\""

    ?value: object
          | array
          | string
          | SIGNED_NUMBER      -> number
          | "true"             -> true
          | "false"            -> false
          | "null"             -> null

    array  : "[" [value ("," value)*] "]"
    object : "{" [pair ("," pair)*] "}"

    string: /\"[^"]*\"/ | "\"type\""
    SIGNED_NUMBER: ["+"|"-"] NUMBER

    DIGIT: "0".."9"
    HEXDIGIT: "a".."f"|"A".."F"|DIGIT
    INT: DIGIT+
    SIGNED_INT: ["+"|"-"] INT
    DECIMAL: INT "." INT? | "." INT

    _EXP: ("e"|"E") SIGNED_INT
    FLOAT: INT _EXP | DECIMAL _EXP?
    NUMBER: FLOAT | INT

    WS: /[ \t\f\r\n]/+
    %ignore WS
"""

# Sample Vega-Lite JSON strings to test
test_cases = [
    # Valid example with flexible encoding values
    '''{
        "$schema": "https://vega.github.io/schema/vega-lite/v3.json",
        "data": {"url": "datasets/sample.csv"},
        "mark": {"type": "point"},
        "encoding": {
            "x": {"field": "city", "type": "nominal"},
            "y": {"aggregate": "average", "field": "price", "type": "quantitative"},
            "color": {"value": "red"}
        }
    }''',
    
    # Invalid example (invalid mark type)
    '''{
        "$schema": "https://vega.github.io/schema/vega-lite/v3.json",
        "data": {"url": "datasets/sample.csv"},
        "mark": {"type": "random"},
        "encoding": {
            "x": {"field": "city", "type": "nominal"},
            "y": {"aggregate": "average", "field": "price", "type": "quantitative"}
        }
    }'''
]

def main():
    # Create the parser
    parser = Lark(vega_lite_grammar, start="start", parser="lalr")

    # Test each case
    for idx, case in enumerate(test_cases, 1):
        try:
            # Parse the test case
            parser.parse(case)
            print(f"Test case {idx}: Passed")
        except Exception as e:
            print(f"Test case {idx}: Failed - {str(e)}")
    
if __name__ == "__main__":
    main()
    