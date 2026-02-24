import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/syncode')

# Import commonly used parts from syncode
from .syncode.syncode import parsers as parsers 
from .syncode.syncode import Grammar
from .syncode.syncode import SyncodeLogitsProcessor
from .syncode.syncode import common
from .syncode.syncode import dataset
