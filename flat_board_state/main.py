import time

from search import search, get_gen_exp_count
from heuristics import *
from flat_board_state import *


def run(name, board, heuristic):
    start_state = flat_board_state(board)
    start_time = time.time()
    search_result = search(start_state, heuristic)
    end_time = time.time() - start_time
    # solution cost (None = unsolvable)
    cost = search_result[-1].g if search_result else None
    print(f"{name:<28} {heuristic.__name__:<20} cost={cost}  "
          f"(generated, expanded)={get_gen_exp_count()}  time={end_time:.4f}s")


if __name__ == '__main__':
    examples = {
        "3x3 example": [[3, 3, 0], [2, 2, 1], [0, 1, 1]],
        "assignment figure": [[2, 2, 2, 2, 2], [2, 1, 3, 0, 2], [2, 3, 1, 1, 2],
                              [2, 1, 1, 0, 2], [2, 2, 2, 2, 2]],
        "unsolvable (walled cube)": [[3, 2, 0], [1, 2, 1]],
    }
    for name, board in examples.items():
        for h in (base_heuristic, advanced_heuristic):
            run(name, board, h)
