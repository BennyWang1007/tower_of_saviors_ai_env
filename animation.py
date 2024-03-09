from Board import *
from PIL import Image, ImageTk
import numpy as np
import tkinter as tk
import time
from typing import Optional, Callable, Literal

rune_size = 80
rune_size2 = int(rune_size * 1.2)
rune_offset = (rune_size2 - rune_size) // 2
margin = 8
button_size = (70, 30)
button_margin = 7

def default_next_callback():
    print('next board')

class BoardScreen():
    def __init__(self, board: Optional[Board]=None):

        self.button_count = 6
        
        # initialize board and rune positions
        self.actions: list[int] = []

        self.window = tk.Tk()
        self.window.title('TOS')

        if board is None:
            board = Board()
        self.set_board(board)
        
        self.frame = tk.Frame(self.window)
        self.frame.pack()
        
        self.canvas = tk.Canvas(
            self.frame, bg="black", 
            width=rune_size * self.num_col + margin * (self.num_col-1), 
            height=rune_size * self.num_row + margin * (self.num_row-1)
        )
        self.canvas.pack()

        
        anchor: Literal["nw", "n", "ne", "w", "center", "e", "sw", "s", "se"] = 'center'
        justify: Literal["left", "center", "right"] = 'center'
        self.next_board_btn = tk.Button(self.window, text='Next Board', command=self.next_board, anchor=anchor, justify=justify)
        self.prev_move_btn = tk.Button(self.window, text='Prev Move', command=self.prev_move, anchor=anchor, justify=justify)
        self.next_move_btn = tk.Button(self.window, text='Next Move', command=self.next_move, anchor=anchor, justify=justify)
        self.reset_btn = tk.Button(self.window, text='Reset', command=self.reset, anchor=anchor, justify=justify)
        self.start_move_btn = tk.Button(self.window, text='Start', command=self.start_move, anchor=anchor, justify=justify)
        self.end_btn = tk.Button(self.window, text='End', command=self.end, anchor=anchor, justify=justify)

        self.buttons: list[tk.Button] = [
            self.next_board_btn, self.prev_move_btn, self.next_move_btn, self.reset_btn, self.start_move_btn, self.end_btn
        ]
        
        for i, button in enumerate(self.buttons):
            button.place(x=10 + i * (button_size[0] + button_margin), y=rune_size * self.num_row + margin * (self.num_row-1) + 10, width=button_size[0], height=button_size[1])

        self.action_idx = 0

        self.next_callback = default_next_callback

        # initialize rune images
        NAMES = ['water', 'fire', 'wood', 'light', 'dark', 'heart']
        _images = [Image.open(f'./images/rune_{name}.png').resize((rune_size, rune_size)) for name in NAMES]
        self.images = [ImageTk.PhotoImage(image) for image in _images]
        _images = [Image.open(f'./images/rune_{name}.png').resize((rune_size2, rune_size2)) for name in NAMES]
        self.images2 = [ImageTk.PhotoImage(image) for image in _images]

        # set key bindings
        self.window.bind('<Up>', lambda e: self.move(MoveDir.UP.value))
        self.window.bind('<Down>', lambda e: self.move(MoveDir.DOWN.value))
        self.window.bind('<Left>', lambda e: self.move(MoveDir.LEFT.value))
        self.window.bind('<Right>', lambda e: self.move(MoveDir.RIGHT.value))
        self.window.bind('q', lambda e: self.move(MoveDir.L_UP.value))
        self.window.bind('w', lambda e: self.move(MoveDir.UP.value))
        self.window.bind('e', lambda e: self.move(MoveDir.R_UP.value))
        self.window.bind('a', lambda e: self.move(MoveDir.LEFT.value))
        self.window.bind('d', lambda e: self.move(MoveDir.RIGHT.value))
        self.window.bind('z', lambda e: self.move(MoveDir.L_DOWN.value))
        self.window.bind('s', lambda e: self.move(MoveDir.DOWN.value))
        self.window.bind('c', lambda e: self.move(MoveDir.R_DOWN.value))

    def set_window_attributes(self):
        window_width = max(rune_size * self.num_col + margin * (self.num_col-1), button_size[0] * self.button_count + button_margin * (self.button_count-1)) + 20
        window_height = rune_size * self.num_row + margin * (self.num_row-1) + button_size[1] + 20
        self.window.geometry(f'{window_width}x{window_height}')

    def set_board(self, board: Board):
        """
        Set the board and initialize the window size
        """
        self.board = board.copy()
        self.start_board = board.copy()
        self.num_col = self.board.num_col
        self.num_row = self.board.num_row
        self.set_window_attributes()
        self.set_runes_pos()
        
        self.end_board = self.start_board.copy()
        for action in self.actions:
            self.end_board.move(action)


    def reset(self):
        self.board = self.start_board.copy()
        self.action_idx = 0
        self.draw()

    def set_next_callback(self, next_callback: Callable):
        self.next_callback = next_callback

    def set_runes_pos(self) -> None:
        self.runes_pos: dict[int, list[int]] = {}
        for i in range(self.num_row):
            for j in range(self.num_col):
                if i == self.board.cur_y and j == self.board.cur_x:
                    self.runes_pos[i * self.num_col + j] = [j * (rune_size + margin) - rune_offset, i * (rune_size + margin) - rune_offset]
                else:
                    self.runes_pos[i * self.num_col + j] = [j * (rune_size + margin), i * (rune_size + margin)]

    def set_actions(self, actions: list[int]):
        self.actions = actions

    def start_move(self):

        self.action_idx = 0
        self.draw()
        self.window.update()
        time.sleep(0.6)
        
        while len(self.actions) > self.action_idx:
            self.next_move()
            time.sleep(0.15)

        self.draw()
        self.window.mainloop()

    def prev_move(self):
        if self.action_idx > 0:
            self.action_idx -= 1
            act = self.actions[self.action_idx]
            self.move(MoveDir.opposite(act))
            self.draw()
            self.window.update()

    def next_move(self):
        if self.action_idx < len(self.actions):
            act = self.actions[self.action_idx]
            # print(f'next move: {int2MoveDir_str(act)}')
            self.move(act)
            self.action_idx += 1
            self.draw()
            self.window.update()

    def end(self):
        self.action_idx = len(self.actions)
        self.board = self.end_board.copy()
        self.draw()
        self.window.update()

    # draw runes on the canvas
    def draw(self):
        # to avoid error when closing the window
        try:
            self.canvas.delete('all')
            for i in range(self.num_row):
                for j in range(self.num_col):
                    if i == self.board.cur_y and j == self.board.cur_x:
                        self.canvas.create_image(self.runes_pos[i * self.num_col + j], image=self.images2[self.board.board[i][j]-1], anchor='nw')
                    else:
                        self.canvas.create_image(self.runes_pos[i * self.num_col + j], image=self.images[self.board.board[i][j]-1], anchor='nw')
        except Exception:
            pass

    def next_board(self):
        self.next_callback()

    def update(self):
        pass

    def move(self, action: int):
        if action == MoveDir.NONE.value:
            return
        # create animation of moving runes
        x, y = self.board.cur_x, self.board.cur_y
        start = self.board.pos2idx(x, y)
        if action == MoveDir.UP.value and y > 0:
            end = start - self.num_col
        elif action == MoveDir.DOWN.value and y < self.num_row - 1:
            end = start + self.num_col
        elif action == MoveDir.LEFT.value and x > 0:
            end = start - 1
        elif action == MoveDir.RIGHT.value and x < self.num_col - 1:
            end = start + 1
        elif action == MoveDir.L_UP.value and y > 0 and x > 0:
            end = start - self.num_col - 1
        elif action == MoveDir.R_UP.value and y > 0 and x < self.num_col - 1:
            end = start - self.num_col + 1
        elif action == MoveDir.L_DOWN.value and y < self.num_row - 1 and x > 0:
            end = start + self.num_col - 1
        elif action == MoveDir.R_DOWN.value and y < self.num_row - 1 and x < self.num_col - 1:
            end = start + self.num_col + 1
        else:
            print(f'Invalid action in move(): {x=}, {y=}, {int2MoveDir_str(action)}')
            return
        
        self.moving_animation_canvas(start, end)
        self.runes_pos[start], self.runes_pos[end] = self.runes_pos[end], self.runes_pos[start]
        self.board.move(action)

    def approach(self, cur: int, target: int, speed: int):
        if cur < target:
            return min(cur + speed, target)
        elif cur > target:
            return max(cur - speed, target)

    def moving_animation_canvas(self, start: int, end: int):
        speed = 3 * rune_size // 50
        start_target_x, start_target_y = self.grid2location(end)
        start_target_x -= rune_offset
        start_target_y -= rune_offset
        end_target_x, end_target_y = self.grid2location(start)
        while self.runes_pos[start][0] != start_target_x or self.runes_pos[start][1] != start_target_y or self.runes_pos[end][0] != end_target_x or self.runes_pos[end][1] != end_target_y:
                
            if self.runes_pos[start][0] < start_target_x:
                self.runes_pos[start][0] = min(self.runes_pos[start][0] + speed, start_target_x)
            else:
                self.runes_pos[start][0] = max(self.runes_pos[start][0] - speed, start_target_x)
            if self.runes_pos[start][1] < start_target_y:
                self.runes_pos[start][1] = min(self.runes_pos[start][1] + speed, start_target_y)
            else:
                self.runes_pos[start][1] = max(self.runes_pos[start][1] - speed, start_target_y)
            if self.runes_pos[end][0] < end_target_x:
                self.runes_pos[end][0] = min(self.runes_pos[end][0] + speed, end_target_x)
            else:
                self.runes_pos[end][0] = max(self.runes_pos[end][0] - speed, end_target_x)
            if self.runes_pos[end][1] < end_target_y:
                self.runes_pos[end][1] = min(self.runes_pos[end][1] + speed, end_target_y)
            else:
                self.runes_pos[end][1] = max(self.runes_pos[end][1] - speed, end_target_y)

            self.draw()
            self.window.update()
            time.sleep(0.001)
        self.window.update()

    def grid2location(self, grid: int) -> tuple[int, int]:
        # grid_x, grid_y = grid % COL_NUM, grid // COL_NUM
        return (grid % self.num_col) * (rune_size + margin), (grid // self.num_col) * (rune_size + margin)


if __name__ == "__main__":

    board = Board()
    board.set_board(np.array([

        # 姆姆技一固版
        [3, 3, 3, 5, 5, 3], 
        [3, 5, 5, 4, 4, 3], 
        [3, 4, 4, 2, 2, 3], 
        [3, 2, 2, 1, 1, 3], 
        [3, 1, 1, 3, 3, 3]

        # 姆姆技二固版
        # [3, 6, 6, 5, 5, 6], 
        # [3, 5, 5, 4, 4, 3], 
        # [3, 4, 4, 2, 2, 3], 
        # [3, 2, 2, 1, 1, 3], 
        # [6, 1, 1, 6, 6, 3]

        # 尼祿固版
        # [3, 3, 2, 4, 4, 1], 
        # [3, 2, 3, 4, 1, 4], 
        # [2, 3, 5, 1, 4, 6], 
        # [3, 5, 3, 4, 6, 4], 
        # [5, 3, 4, 6, 4, 3]
    ]))
    board.set_cur_pos(5, 4)
    # def next_():
    #     # print('next board')
    #     # board.random_board()
    #     # board.random_pos()
    #     action_str = "LU U LD RU"
    #     actions = [MoveDir.str2int(s) for s in action_str.split()]
    #     bs.set_actions(actions)
    #     bs.set_board(board)
    #     bs.start_move()
        # board.print_board()
    bs = BoardScreen(board)
    bs.set_board(board)
    bs.set_actions([])
    bs.start_move()
    # bs.set_next_callback(next_)
    # next_()
