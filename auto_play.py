

from route_planning import maximize_score_parallel_forPlay, move2int, dfs, board_moves
from utils import timeit, print_two_board
import cv2
from Board import Board, MoveDir, Runes
import numpy as np
import time
from ppadb.client import Client
from ppadb.device import Device as AdbDevice


# SCREEN_WIDTH, SCREEN_HEIGHT = 1080, 2280
# LEFT_TOP = (0, 1125)

# to change
SCREEN_WIDTH, SCREEN_HEIGHT = 900, 1600
LEFT_TOP = (0, 770)

# don't change
RUNE_SIZE = SCREEN_WIDTH // 6
RUNE_SIZE_SAMPLE = 150
SAMPLE_SIZE = 60

OFFSET = (RUNE_SIZE - SAMPLE_SIZE) // 2
SCALING = 1

device: AdbDevice
rune_templates: dict[str, list[np.ndarray]]

def set_up_adb():

    global device

    adb = Client(host='127.0.0.1', port=5037)
    devices: list[AdbDevice] = adb.devices()

    if len(devices) == 0:
        print('Devices not found')
        quit()
    print(f'{devices=}')
    device = devices[0]

def set_templates():
    global rune_templates

    rune_templates_name = {
        'water' : ['water1', 'water2', 'water3', 'water4'],
        'fire' : ['fire1', 'fire2', 'fire3', 'fire4'],
        'wood' : ['wood1', 'wood2', 'wood3', 'wood4'],
        'light' : ['light1', 'light2', 'light3', 'light4'],
        'dark' : ['dark1', 'dark2', 'dark3', 'dark4'],
        'heart' : ['heart1', 'heart2', 'heart3', 'heart4'],
    }

    template_dir = 'template_60/'
    rune_templates = {k: [cv2.imread(template_dir + f'{rune}.png', 0) for rune in v] for k, v in rune_templates_name.items()}
    size = int(round(1.0 * SAMPLE_SIZE * RUNE_SIZE / RUNE_SIZE_SAMPLE / SCALING, 0))
    rune_templates = {k: [cv2.resize(v, (size, size)) for v in vs] for k, vs in rune_templates.items()}
# print(f'{size=}')
# rune_templates = {k: [cv2.cvtColor(v, cv2.COLOR_BGR2GRAY) for v in vs] for k, vs in rune_templates.items()}

def get_grid_loc(x, y):
    return LEFT_TOP[0] + x * RUNE_SIZE, LEFT_TOP[1] + y * RUNE_SIZE


def match_rune(image, grid):
    s_x, s_y = LEFT_TOP[0] // SCALING, LEFT_TOP[1] // SCALING
    e_x, e_y = (LEFT_TOP[0] + RUNE_SIZE * 6) // SCALING, (LEFT_TOP[1] + RUNE_SIZE * 5) // SCALING

    # image = image[s_y:e_y, s_x:e_x]
    loc_x, loc_y = get_grid_loc(*grid)
    loc_x, loc_y = loc_x + OFFSET, loc_y + OFFSET

    margin = 3
    s_x, e_x = loc_x - margin, loc_x + SAMPLE_SIZE + margin
    s_y, e_y = loc_y - margin, loc_y + SAMPLE_SIZE + margin

    image = image[s_y:e_y, s_x:e_x]

    # loc_x, loc_y = loc_x - LEFT_TOP[0], loc_y - LEFT_TOP[1]
    # loc_x, loc_y = int(round(loc_x / SCALING)), int(round(loc_y / SCALING))
    # print(f'{s_x=}, {s_y=}, {e_x=}, {e_y=}, {loc_x=}, {loc_y=}')
    max_res_of_each = []
    result = 'None'
    for rune, templates in rune_templates.items():
        max_res = 0
        for template in templates:

            res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
            max_res = max(max_res, np.max(res))
            # print(f'{max_res=}')

        max_res_of_each.append(max_res)
    # max_res_of_each = [round(x, 2) for x in max_res_of_each]
    # print(grid, max_res_of_each)
    max_idx = np.argmax(max_res_of_each)
    if max_res_of_each[max_idx] > 0.8:
        result = list(rune_templates.keys())[max_idx]
    return result
    
# @timeit
def read_board() -> Board:

    pic = device.screencap()
    screenshot = cv2.imdecode(np.frombuffer(pic, np.uint8), cv2.IMREAD_GRAYSCALE)

    # screenshot = cv2.imread('E:/shot_test3.png', cv2.IMREAD_GRAYSCALE)

    width, height = screenshot.shape[1], screenshot.shape[0]
    screenshot = cv2.resize(screenshot, (width // SCALING, height // SCALING))

    board_arr = []
    for y in range(5):
        arr = []
        for x in range(6):
            rune = match_rune(screenshot, (x, y))
            # print(rune, end=' ')
            arr.append(Runes.str2int(rune))
        # print()
        board_arr.append(arr)

    board = Board()
    board.set_board(np.array(board_arr))

    return board

"""-----------------send event functions-----------------"""

EV_SYN = 0
EV_KEY = 1
EV_ABS = 3

SYN_REPORT = 0
ABS_MT_POSITION_X = 53
ABS_MT_POSITION_Y = 54
ABS_MT_TRACKING_ID = 57
BTN_TOUCH = 330

def sendevent(type: int, code: int, value: int, dev: str='/dev/input/event2'):
    device.shell(f'sendevent {dev} {type} {code} {value}')

def send_SYN_REPORT() -> None:
    sendevent(EV_SYN, SYN_REPORT, 0)

def send_BTN_TOUCH_DOWN() -> None:
    sendevent(EV_KEY, BTN_TOUCH, 1)

def send_POSITION(x, y) -> None:
    sendevent(EV_ABS, ABS_MT_POSITION_X, x)
    sendevent(EV_ABS, ABS_MT_POSITION_Y, y)
    send_SYN_REPORT()


def send_ABS_MT_TRACKING_ID(x: int) -> None:
    sendevent(EV_ABS, ABS_MT_TRACKING_ID, x)


"""-----------------send event functions-----------------"""

def route_move(route) -> None:
    route_loc = [get_grid_loc(x, y) for x, y in route]
    route_loc = [(x + RUNE_SIZE // 2, y + RUNE_SIZE // 2) for x, y in route_loc]
    route_loc = [(y, SCREEN_WIDTH - x) for x, y in route_loc] # weird coordinate system
    # print(route_loc)

    send_ABS_MT_TRACKING_ID(1)
    sendevent(EV_KEY, BTN_TOUCH, 1)

    for x, y in route_loc:
        send_POSITION(x, y)

    send_ABS_MT_TRACKING_ID(-1)
    sendevent(EV_KEY, BTN_TOUCH, 0)
    send_SYN_REPORT()

@timeit
def route_planning(board: np.ndarray, iter: int, max_first_depth: int=8, max_depth: int=10) -> tuple[int, list[tuple[int, int]]]:
    """
    Given a board, return the best score and best route.
    """
    final_route = []
    max_score = -1

    score, route = maximize_score_parallel_forPlay(board, max_depth=max_first_depth)
    
    while iter > 0 and score > max_score:
        max_score = score
        if len(final_route) == 0:
            final_route = route
        else:
            final_route += route[1:]
        board = board_moves(board, route)
        # start_time = time.time()
        score, route = dfs(board, route[-1], [route[-1]], max_depth)
        # print(f'dfs of depth {max_depth} took {time.time() - start_time} seconds')
        iter -= 1

    return max_score, final_route



if __name__ == "__main__":

    set_up_adb()
    set_templates()
    # main loop
    while True:
        # read board until success
        while True:
            board = read_board()
            # board.print_board()
            if np.sum(board.board == Runes.NONE.value) == 0:
                break
            time.sleep(5)

        score, final_route = route_planning(board.board.copy(), iter=5, max_first_depth=8, max_depth=12)

        # move the board to display the result
        board.cur_x, board.cur_y = final_route[0][0], final_route[0][1]
        moves = []
        for i in range(1, len(final_route)):
            dx, dy = final_route[i][0] - final_route[i - 1][0], final_route[i][1] - final_route[i - 1][1]
            move = move2int((dx, dy))
            moves.append(move)
        
        board_copy = board.board.copy()
        for move in moves:
            board.move(move)
        
        print_two_board(board_copy, board.board)
        print(f'{score=}', final_route)

        route_move(final_route)

        time.sleep(5)


    