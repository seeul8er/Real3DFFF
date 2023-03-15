# This file is licenced under GPL3
# source: https://github.com/fragmuffin/pygcode
# modified by Wolfgang Christl

import re

REGEX_FLOAT = re.compile(r'^-?(\d+\.?\d*|\.\d+)')  # testcase: ..tests.test_words.WordValueMatchTests.test_float
REGEX_INT = re.compile(r'^\s*-?\d+')
REGEX_POSITIVEINT = re.compile(r'^\s*\d+')
REGEX_CODE = re.compile(r'^\s*\d+(\.\d)?')  # float, but can't be negative


# Value cleaning functions
def _clean_codestr(value):
    if value < 10:
        return "0%g" % value
    return "%g" % value


CLEAN_NONE = lambda v: v
CLEAN_FLOAT = lambda v: "{0:g}".format(round(v, 3))
CLEAN_CODE = _clean_codestr
CLEAN_INT = lambda v: "%g" % v

WORD_MAP = {
    # Rotational Axes
    'A': {
        'class': float,
        'value_regex': REGEX_FLOAT,
        'description': "Absolute or incremental position of A axis (rotational axis around X axis)",
        'clean_value': CLEAN_FLOAT,
    },
    'B': {
        'class': float,
        'value_regex': REGEX_FLOAT,
        'description': "Absolute or incremental position of B axis (rotational axis around Y axis)",
        'clean_value': CLEAN_FLOAT,
    },
    'C': {
        'class': float,
        'value_regex': REGEX_FLOAT,
        'description': "Absolute or incremental position of C axis (rotational axis around Z axis)",
        'clean_value': CLEAN_FLOAT,
    },
    'D': {
        'class': float,
        'value_regex': REGEX_FLOAT,
        'description': "Defines diameter or radial offset used for cutter compensation. D is used for depth of cut on lathes. It is used for aperture selection and commands on photoplotters.",
        'clean_value': CLEAN_FLOAT,
    },
    # Feed Rates
    'E': {
        'class': float,
        'value_regex': REGEX_FLOAT,
        'description': "Precision feedrate for threading on lathes",
        'clean_value': CLEAN_FLOAT,
    },
    'F': {
        'class': float,
        'value_regex': REGEX_FLOAT,
        'description': "Feedrate",
        'clean_value': CLEAN_FLOAT,
    },
    # G-Codes
    'G': {
        'class': float,
        'value_regex': REGEX_CODE,
        'description': "Address for preparatory commands",
        'clean_value': CLEAN_CODE,
    },
    # Tool Offsets
    'H': {
        'class': float,
        'value_regex': REGEX_FLOAT,
        'description': "Defines tool length offset; Incremental axis corresponding to C axis (e.g., on a turn-mill)",
        'clean_value': CLEAN_FLOAT,
    },
    # Arc radius center coords
    'I': {
        'class': float,
        'value_regex': REGEX_FLOAT,
        'description': "Defines arc center in X axis for G02 or G03 arc commands. Also used as a parameter within some fixed cycles.",
        'clean_value': CLEAN_FLOAT,
    },
    'J': {
        'class': float,
        'value_regex': REGEX_FLOAT,
        'description': "Defines arc center in Y axis for G02 or G03 arc commands. Also used as a parameter within some fixed cycles.",
        'clean_value': CLEAN_FLOAT,
    },
    'K': {
        'class': float,
        'value_regex': REGEX_FLOAT,
        'description': "Defines arc center in Z axis for G02 or G03 arc commands. Also used as a parameter within some fixed cycles, equal to L address.",
        'clean_value': CLEAN_FLOAT,
    },
    # Loop Count
    'L': {
        'class': int,
        'value_regex': REGEX_POSITIVEINT,
        'description': "Fixed cycle loop count; Specification of what register to edit using G10",
        'clean_value': CLEAN_INT,
    },
    # Miscellaneous Function
    'M': {
        'class': float,
        'value_regex': REGEX_CODE,
        'description': "Miscellaneous function",
        'clean_value': CLEAN_CODE,
    },
    # Real3DFFF normal vector X-Value
    'N': {
        'class': float,
        'value_regex': REGEX_FLOAT,
        'description': "Real3DFFF normal vector X-Value",
        'clean_value': CLEAN_FLOAT,
    },
    # Real3DFFF normal vector Y-Value
    'O': {
        'class': float,
        'value_regex': REGEX_FLOAT,
        'description': "Real3DFFF normal vector Y-Value",
        'clean_value': CLEAN_FLOAT,
    },
    # Parameter (arbitrary parameter)
    'P': {
        'class': float,  # parameter is often an integer, but can be a float
        'value_regex': REGEX_FLOAT,
        'description': "Serves as parameter address for various G and M codes",
        'clean_value': CLEAN_FLOAT,
    },
    # Peck increment
    'Q': {
        'class': float,
        'value_regex': REGEX_FLOAT,
        'description': "Depth to increase on each peck; Peck increment in canned cycles",
        'clean_value': CLEAN_FLOAT,
    },
    # Real3DFFF normal vector Z-Value
    'R': {
        'class': float,
        'value_regex': REGEX_FLOAT,
        'description': "Real3DFFF normal vector Z-Value",
        'clean_value': CLEAN_FLOAT,
    },
    # Spindle speed
    'S': {
        'class': float,
        'value_regex': REGEX_FLOAT,
        'description': "Defines speed, either spindle speed or surface speed depending on mode",
        'clean_value': CLEAN_FLOAT,
    },
    # Tool Selecton
    'T': {
        'class': str,
        'value_regex': REGEX_POSITIVEINT,  # tool string may have leading '0's, but is effectively an index (integer)
        'description': "Tool selection",
        'clean_value': CLEAN_NONE,
    },
    # Incremental axes
    'U': {
        'class': float,
        'value_regex': REGEX_FLOAT,
        'description': "Incremental axis corresponding to X axis (typically only lathe group A controls) Also defines dwell time on some machines (instead of 'P' or 'X').",
        'clean_value': CLEAN_FLOAT,
    },
    'V': {
        'class': float,
        'value_regex': REGEX_FLOAT,
        'description': "Incremental axis corresponding to Y axis",
        'clean_value': CLEAN_FLOAT,
    },
    # 'W': {
    #     'class': float,
    #     'value_regex': REGEX_FLOAT,
    #     'description': "Incremental axis corresponding to Z axis (typically only lathe group A controls)",
    #     'clean_value': CLEAN_FLOAT,
    # },
    # Linear Axes
    'X': {
        'class': float,
        'value_regex': REGEX_FLOAT,
        'description': "Absolute or incremental position of X axis.",
        'clean_value': CLEAN_FLOAT,
    },
    'Y': {
        'class': float,
        'value_regex': REGEX_FLOAT,
        'description': "Absolute or incremental position of Y axis.",
        'clean_value': CLEAN_FLOAT,
    },
    'Z': {
        'class': float,
        'value_regex': REGEX_FLOAT,
        'description': "Absolute or incremental position of Z axis.",
        'clean_value': CLEAN_FLOAT,
    },
}


class GCodeBlock:
    def __init__(self, text=None, verify=True):
        """
        Block Constructor

        :param text: gcode line content (including comments) as string
        :type text: :class:`str`
        :param verify: verify given codes (modal & non-modal are not repeated)
        :type verify: :class:`bool`

        .. note::

            State & machine specific codes cannot be verified at this point;
            they must be processed by a virtual machine to be fully verified.

        """

        self._raw_text = None
        self._text = None
        self.words = []
        self.gcodes = []
        self.modal_params = []

        # clean up block string
        if text:
            self._raw_text = text  # unaltered block content (before alteration)
            text = re.sub(r'(^\s+|\s+$)', '', text)  # remove whitespace padding
            text = re.sub(r'\s+', ' ', text)  # remove duplicate whitespace with ' '
            self._text = text  # cleaned up block content

            # Get words from text, and group into gcodes
            self.words = list(self.text2words(self._text))
            # (self.gcodes, self.modal_params) = words2gcodes(self.words)

            # Verification
            if verify:
                self._assert_gcodes()

    @property
    def text(self):
        if self._text:
            return self._text
        return str(self)

    def text2words(self, block_text):
        """
        Iterate through block text yielding Word instances

        :param block_text: text for given block with comments removed
        """
        word_map = WORD_MAP

        next_word = re.compile(r'.*?(?P<letter>[%s])' % ''.join(word_map.keys()), re.IGNORECASE)

        index = 0
        while True:
            letter_match = next_word.match(block_text[index:])
            if letter_match:
                # Letter
                letter = letter_match.group('letter').upper()
                index += letter_match.end()  # propogate index to start of value

                # Value
                value_regex = word_map[letter]['value_regex']
                value_match = value_regex.search(block_text[index:])
                if value_match is None:
                    print(f"word {letter} value invalid with line: {block_text}")
                    value = '0'
                else:
                    value = value_match.group()  # matched text

                yield Word(letter, value)

                if value_match is None:
                    index += 1
                else:
                    index += value_match.end()  # propogate index to end of value
            else:
                break

    def _assert_gcodes(self):
        modal_groups = set()
        code_words = set()

        for gc in self.gcodes:

            # Assert all gcodes are not repeated in the same block
            if gc.word in code_words:
                raise AssertionError("%s cannot be in the same block" % ([
                    x for x in self.gcodes
                    if x.modal_group == gc.modal_group
                ]))
            code_words.add(gc.word)

            # Assert all gcodes are from different modal groups
            if gc.modal_group is not None:
                if gc.modal_group in modal_groups:
                    raise AssertionError("%s cannot be in the same block" % ([
                        x for x in self.gcodes
                        if x.modal_group == gc.modal_group
                    ]))
                modal_groups.add(gc.modal_group)

    def __len__(self):
        """
        Block length = number of identified gcodes (+ 1 if any modal parameters are defined)

        :return: block length
        """
        length = len(self.gcodes)
        if self.modal_params:
            length += 1
        return length

    def __bool__(self):
        return bool(self.words)


class Word(object):
    def __init__(self, *args, **kwargs):
        # Parameters (listed)
        args_count = len(args)
        if args_count == 1:
            # Word('G90')
            letter = args[0][0]  # first letter
            value = args[0][1:]  # rest of string
        elif args_count == 2:
            # Word('G', 90)
            (letter, value) = args
        else:
            raise AssertionError("input arguments either: (letter, value) or (word_str)")

        # Parameters (keyword)

        letter = letter.upper()

        self._word_map = WORD_MAP
        self._value_class = self._word_map[letter]['class']
        # self._value_clean = self._word_map[letter]['clean_value']

        self.letter = letter
        self.value = value

    def __str__(self):
        return "{letter}{value}".format(
            letter=self.letter,
            value=self.value_str,
        )

    def __repr__(self):
        return "<{class_name}: {string}>".format(
            class_name=self.__class__.__name__,
            string=str(self),
        )

    # Sorting
    def __lt__(self, other):
        return (self.letter, self.value) < (other.letter, other.value)

    def __gt__(self, other):
        return (self.letter, self.value) > (other.letter, other.value)

    def __le__(self, other):
        return (self.letter, self.value) <= (other.letter, other.value)

    def __ge__(self, other):
        return (self.letter, self.value) >= (other.letter, other.value)

    def __ne__(self, other):
        return not self.__eq__(other)

    # Hashing
    def __hash__(self):
        return hash((self.letter, self.value))

    # @property
    # def value_str(self):
    #     """Clean string representation, for consistent file output"""
    #     return self._value_clean(self.value)

    # Value Properties
    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        self._value = self._value_class(new_value)

    @property
    def description(self):
        return "%s: %s" % (self.letter, self._word_map[self.letter].description)
