from Board import Board, MoveDir, evaluate_board, Runes
from utils import timeit
import numpy as np
from tqdm import tqdm
from functools import cache

from multiprocessing import Pool

NUM_COL = 6
NUM_ROW = 5
MODE = 'fixed'
MAX_DEPTH = 12

def move2int(move: tuple[int, int]) -> int:
    """
    Convert a move tuple to an integer of MoveDir enum.
    """
    if move == (-1, 0):
        return MoveDir.LEFT.value
    elif move == (1, 0):
        return MoveDir.RIGHT.value
    elif move == (0, -1):
        return MoveDir.UP.value
    elif move == (0, 1):
        return MoveDir.DOWN.value
    elif move == (0, 0):
        return MoveDir.NONE.value
    else:
        raise ValueError(f'Invalid move: {move}')
    
def board_moves(board: np.ndarray, route: list) -> np.ndarray:
    """
    Apply a route to a board and return the new board.
    """
    start_rune = board[route[0][1], route[0][0]]
    for i in range(1, len(route)):
        board[route[i - 1][1], route[i - 1][0]] = board[route[i][1], route[i][0]]
    board[route[-1][1], route[-1][0]] = start_rune
    return board

def array_to_tuple(arr):
    """
    Convert a numpy array to a nested tuple of tuples, which is hashable and can be used as a cache key.
    """
    return tuple(map(tuple, arr))



# @lru_cache(maxsize=None)
@cache
def cached_evaluate_board(board_tuple, num_col, num_row):
    """
    A wrapper function that takes a nested tuple (converted from a numpy array),
    evaluates the board, and returns the score. This function is meant to be cached.
    """
    board_array = np.array(board_tuple)  # Convert back to a numpy array for processing
    f_c, c, t_e = evaluate_board(board_array, num_col, num_row)
    return f_c * 2 + c * 5 + t_e * 1

def get_board_score(board: np.ndarray, route: list):
        
    new_board = board.copy()
    new_board = board_moves(new_board, route)
    # board_tuple = array_to_tuple(new_board)
    # return cached_evaluate_board(board_tuple, NUM_COL, NUM_ROW)
    
    f_c, c, t_e = evaluate_board(new_board, NUM_COL, NUM_ROW)
    return f_c * 5 + c * 2 + t_e * 1

def is_valid_move(x: int, y: int) -> bool:
    return 0 <= x < NUM_COL and 0 <= y < NUM_ROW

def dfs(board: np.ndarray, current_position: tuple, current_route: list[tuple], max_depth: int=MAX_DEPTH):
    
    if len(current_route) > max_depth:
        return get_board_score(board, current_route), current_route

    best_score = float('-inf')
    best_route = []

    for move in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]:  # Represents left, right, up, down, stop
        if move == (0, 0):
            score, route = get_board_score(board, current_route), current_route
            if score > best_score:
                best_score = score
                best_route = route
            elif score == best_score and len(route) < len(best_route):
                best_route = route
        else:
            new_x, new_y = current_position[0] + move[0], current_position[1] + move[1]

            if is_valid_move(new_x, new_y):
                new_position = (new_x, new_y)
                if new_position not in current_route:
                    new_route = current_route + [new_position]
                    score, route = dfs(board, new_position, new_route, max_depth)
                    if score > best_score:
                        best_score = score
                        best_route = route
                    elif score == best_score and len(route) < len(best_route):
                        best_route = route
    return best_score, best_route

# @timeit
def maximize_score(board: np.ndarray):
    best_start = (-1, -1)
    best_score = float('-inf')
    best_route = []

    tqdm.write('Start searching for best route...')

    for start_position in tqdm([(x, y) for x in range(NUM_COL) for y in range(NUM_ROW)]):
        score, route = dfs(board, start_position, [start_position])
        if score > best_score:
            best_score = score
            best_start = start_position
            best_route = route
    formatted_route = [best_start[0], best_start[1]]
    for i in range(1, len(best_route)):
        dx, dy = best_route[i][0] - best_route[i - 1][0], best_route[i][1] - best_route[i - 1][1]
        move = move2int((dx, dy))
        formatted_route.append(move)
    return best_score, formatted_route

# def dfs_wrapper(start_position):
#     return dfs(start_position, [start_position])

def dfs_wrapper(args):
    return dfs(*args)

def maximize_score_parallel(board: np.ndarray):
    best_start = (-1, -1)
    best_score = float('-inf')
    best_route = []

    # tqdm.write('Start searching for best route...')
    start_positions = [(x, y) for x in range(NUM_COL) for y in range(NUM_ROW)]
    args = [(board.copy(), start_position, [start_position]) for start_position in start_positions]

    with Pool() as pool:
        results = list(pool.map(dfs_wrapper, args))
        # results = list(tqdm(pool.imap(dfs_wrapper, args), total=len(args)))
    
    for score, route in results:
        if score > best_score:
            best_score = score
            best_start = route[0]
            best_route = route
    formatted_route = [best_start[0], best_start[1]]
    for i in range(1, len(best_route)):
        dx, dy = best_route[i][0] - best_route[i - 1][0], best_route[i][1] - best_route[i - 1][1]
        move = move2int((dx, dy))
        formatted_route.append(move)
    return best_score, formatted_route

# @timeit
def maximize_score_parallel_forPlay(board: np.ndarray, max_depth=MAX_DEPTH):
    # best_start = (-1, -1)
    best_score = float('-inf')
    best_route = []

    # tqdm.write('Start searching for best route...')
    start_positions = [(x, y) for x in range(NUM_COL) for y in range(NUM_ROW)]
    args = [(board.copy(), start_position, [start_position], max_depth) for start_position in start_positions]

    with Pool() as pool:
        results = list(pool.map(dfs_wrapper, args))
        # results = list(tqdm(pool.imap(dfs_wrapper, args), total=len(args)))
    
    for score, route in results:
        if score > best_score:
            best_score = score
            # best_start = route[0]
            best_route = route
    # formatted_route = [best_start[0], best_start[1]]
    # for i in range(1, len(best_route)):
    #     dx, dy = best_route[i][0] - best_route[i - 1][0], best_route[i][1] - best_route[i - 1][1]
    #     move = move2int((dx, dy))
    #     formatted_route.append(move)
    return best_score, best_route

    
@timeit
def main():
    # Example usage:
    board = Board(num_col=NUM_COL, num_row=NUM_ROW, mode=MODE)

    # score, route = maximize_score(board.board)
    score, route = maximize_score_parallel(board.board)

    board.print_board()
    print(route)

    board.cur_x, board.cur_y = route[0], route[1]

    print(f'starting position: {board.cur_x}, {board.cur_y}')

    action_str = "".join([MoveDir.int2str(action) + ', ' for action in route[2:]])[0:-2]
    print(f'actions=[{action_str}]')

    for i in range(2, len(route)):
        board.move(route[i])

    print(f'score={score}')
    board.print_board()

if __name__ == "__main__":
    main()


