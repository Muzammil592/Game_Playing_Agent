import numpy as np

class TicTacToeEnv:
    """
    Tic-Tac-Toe environment.
    Board: 3x3 flattened to length-9 tuple (0=empty, 1=X, -1=O)
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = [0] * 9
        self.current_player = 1  # X starts
        self.done = False
        return self._state()

    def _state(self):
        return tuple(self.board)

    def available_actions(self):
        return [i for i, v in enumerate(self.board) if v == 0]

    def step(self, action):
        assert not self.done, "Game is over. Call reset()."
        assert self.board[action] == 0, "Invalid move."

        self.board[action] = self.current_player
        winner = self._check_winner()
        reward = 0.0

        if winner == self.current_player:
            reward = 1.0
            self.done = True
        elif not self.available_actions():
            reward = 0.5   # draw
            self.done = True
        else:
            self.current_player *= -1  # switch player

        return self._state(), reward, self.done

    def _check_winner(self):
        b = self.board
        lines = [
            [0,1,2],[3,4,5],[6,7,8],   # rows
            [0,3,6],[1,4,7],[2,5,8],   # cols
            [0,4,8],[2,4,6]            # diags
        ]
        for line in lines:
            s = sum(b[i] for i in line)
            if s == 3:  return  1
            if s == -3: return -1
        return 0

    def render(self):
        symbols = {0: '.', 1: 'X', -1: 'O'}
        for row in range(3):
            print(' '.join(symbols[self.board[row*3 + col]] for col in range(3)))
        print()
