from enum import Enum

class Runes(Enum):
    """A class of runes"""
    WATER = 1
    FIRE = 2
    WOOD = 3
    LIGHT = 4
    DARK = 5
    HEART = 6
    NONE = 0
    
    @staticmethod
    def int2str(rune: int) -> str:
        for r in Runes:
            if r.value == rune:
                return r.name
        return 'invalid rune'
    
    @staticmethod
    def int2color_code(rune: int) -> str:
        if rune == Runes.WATER.value:
            return "\033[94m" # blue
        if rune == Runes.FIRE.value:
            return "\033[91m" # red
        if rune == Runes.WOOD.value:
            return "\033[92m" # green
        if rune == Runes.LIGHT.value:
            return "\033[93m" # yellow
        if rune == Runes.DARK.value:
            return "\033[95m" # magenta
        if rune == Runes.HEART.value:
            return "\033[0m" # white
        raise ValueError(f'Invalid rune: {rune}')
    
    @staticmethod
    def str2int(s: str) -> int:
        if s == 'WATER' or s == 'water' or s == 'Water' or s == 'w' or s == 'W':
            return Runes.WATER.value
        elif s == 'FIRE' or s == 'fire' or s == 'Fire' or s == 'f' or s == 'F':
            return Runes.FIRE.value
        elif s == 'WOOD' or s == 'wood' or s == 'Wood' or s == 'w' or s == 'W':
            return Runes.WOOD.value
        elif s == 'LIGHT' or s == 'light' or s == 'Light' or s == 'l' or s == 'L':
            return Runes.LIGHT.value
        elif s == 'DARK' or s == 'dark' or s == 'Dark' or s == 'd' or s == 'D':
            return Runes.DARK.value
        elif s == 'HEART' or s == 'heart' or s == 'Heart' or s == 'h' or s == 'H':
            return Runes.HEART.value
        else:
            return Runes.NONE.value