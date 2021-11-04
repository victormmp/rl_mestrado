import logging


COLOR_CODES = ['\x1b[1;%dm' % v for v in range(30, 38)]
BG_COLOR_CODES = ['\x1b[1;%dm' % v for v in range(40, 48)]
COLOR_CODES_DARK = ['\x1b[0;%dm' % v for v in range(30, 38)]
BG_COLOR_CODES_DARK = ['\x1b[0;%dm' % v for v in range(40, 48)]
COLOR_CODES_256 = ['\x1b[38;5;%dm' % v for v in range(0, 255)]
BG_COLOR_CODES_256 = ['\x1b[48;5;%dm' % v for v in range(0, 255)]

RESET = '\x1b[0m'

# Foreground
DGRAY, RED, GREEN, YELLOW, BLUE, PURPLE, CYAN, WHITE = COLOR_CODES
BLACK, DRED, DGREEN, DYELLOW, DBLUE, DPURPLE, DCYAN, GRAY = COLOR_CODES_DARK

# Background
BG_GRAY, BG_RED, BG_GREEN, BG_YELLOW, BG_BLUE, BG_PURPLE, BG_CYAN, BG_WHITE = BG_COLOR_CODES
(BG_BLACK, BG_DRED, BG_DGREEN, BG_DYELLOW, BG_DBLUE, BG_DPURPLE, BG_DCYAN,
    BG_DGRAY) = BG_COLOR_CODES_DARK


def get_logger(name: str, log_file_output: str = None):
    log_format = '[%(asctime)s]' 
    log_format += CYAN + '%(filename)15s:%(lineno)-4d' + RESET
    log_format += YELLOW + '%(levelname)-7s' + RESET
    log_format += '%(message)s'

    date_fmt = '%2d %2b %2y %H:%M:%S'
    logging.basicConfig(level=logging.INFO, format=log_format, datefmt=date_fmt)
    LOGGER = logging.getLogger(name)
    LOGGER.setLevel(logging.INFO)

    if log_file_output:
        ch = logging.FileHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('[%(asctime)s]%(filename)15s:%(lineno)-4d%(levelname)-7s%(message)s')
        ch.setFormatter(formatter)
        LOGGER.addHandler(ch)

    return LOGGER
