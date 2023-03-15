# This file is licenced under GPL3
# source: https://github.com/fragmuffin/pygcode
# modified by Wolfgang Christl

import re

from gcode.gcode_visualizer.GCodeBlock import GCodeBlock
from gcode.gcode_visualizer.VRepRapMove import VRepRapMove


class GCodeLine:
    line_regex = re.compile(r'^(?P<block_and_comment>.*?)?(?P<macro>%.*%?)?\s*$')
    regex = re.compile(r'\s*;\s*(?P<text>.*)$')

    def __init__(self, text=None):
        self._text = text
        self.comment = None
        self.block = None
        self.move = None  # A movement is existent
        self.layer_num = None  # Layer index
        self.layer_height = None  # Height of layer
        self.layer_z = None  # Z-Coordinate of layer
        self.local_layer_cnt = None  # For storing/debugging CLFFF (Christl) with supplied Preform and local layer count
        self.local_layer_indx = None  # For storing/debugging CLFFF (Christl) with local layer index, Preform with variable layer count

        if text is not None:
            match = self.line_regex.search(text)

            block_and_comment = match.group('block_and_comment')
            self.macro = match.group('macro')

            (block_str, comment) = self.split_line(block_and_comment)
            self.block = GCodeBlock(block_str)
            if comment:
                self.comment = comment

    def split_line(self, line_text):
        comments = []
        block_str = line_text.rstrip("\n")  # to remove potential return carriage from comment body

        matches = list(self.regex.finditer(block_str))
        if matches:
            for match in reversed(matches):
                # Build list of comment text
                comments.insert(0, match.group('text'))  # prepend
                # Remove comments from given block_str
                block_str = block_str[:match.start()] + block_str[match.end():]

        return block_str, ". ".join(comments)

    def set_raw_text(self, new_line_text):
        """
        Assign a new text to the line. Alternative to creating a new GCodeLine object. This way it is faster

        :param new_line_text: New G-Code line
        """
        if new_line_text is not None:
            self._text = new_line_text
            match = self.line_regex.search(new_line_text)

            block_and_comment = match.group('block_and_comment')
            self.macro = match.group('macro')

            (block_str, comment) = self.split_line(block_and_comment)
            self.block = GCodeBlock(block_str)
            if comment:
                self.comment = comment

    def set_move(self, new_move: VRepRapMove):
        self.move = new_move

    def set_layer(self, new_layer_num: int):
        self.layer_num = new_layer_num

    def set_layer_height(self, new_layer_height: float):
        self.layer_height = new_layer_height

    def set_layer_z(self, new_layer_z: float):
        self.layer_z = new_layer_z

    def return_regenerated_gcode(self, digits=4, x_changed=True, y_changed=True, z_changed=True) -> str:
        if self.move is None:
            return self._text
        else:
            # regenerate g code command from move
            return self.move.to_gcode(self.comment, digits, x_changed=x_changed, y_changed=y_changed, z_changed=z_changed)

    @property
    def text(self):
        if self._text is None:
            return str(self)
        return self._text

    @property
    def gcodes(self):
        return self.block.gcodes
