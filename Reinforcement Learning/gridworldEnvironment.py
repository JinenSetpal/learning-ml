from gym.envs.classic_control import rendering
from gym.envs.toy_text import discrete
from collections import defaultdict
import numpy as np
import pickle
import time
import os


# noinspection PyShadowingNames
class GridWorldEnv(discrete.DiscreteEnv):
    def __init__(self, num_rows=4, num_cols=6, delay=0.05):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.delay = delay

        move_up = lambda row, col: (max(row - 1, 0), col)
        move_down = lambda row, col: (min(row + 1, num_rows - 1), col)
        move_left = lambda row, col: (row, max(col - 1, 0))
        move_right = lambda row, col: (row, min(col + 1, num_cols - 1))

        # self.action_defs = [move_up, move_right, move_down, move_left]
        self.actions = {0: move_up, 1: move_right, 2: move_down, 3: move_left}

        nS = num_cols * num_rows
        nA = len(self.actions)
        self.grid2state_dict = {(s // num_cols, s % num_cols): s for s in range(nS)}
        self.state2grid_dict = {s: (s // num_cols, s % num_cols) for s in range(nS)}

        gold_cell = (num_rows // 2, num_cols - 2)
        trap_cells = [(gold_cell[0] + 1, gold_cell[1]), (gold_cell[0] - 1, gold_cell[1]),
                      (gold_cell[0], gold_cell[1] - 1)]

        gold_state = self.grid2state_dict[gold_cell]
        trap_states = [self.grid2state_dict[(r, c)] for r, c in trap_cells]
        self.terminal_states = [gold_state] + trap_states
        print(self.terminal_states)

        P = defaultdict(dict)
        for s in range(nS):
            row, col = self.state2grid_dict[s]
            P[s] = defaultdict(list)
            for a in range(nA):
                action = self.actions[a]
                next_s = self.grid2state_dict[action(row, col)]

                if self.is_terminal(next_s):
                    r = (1.0 if next_s == self.terminal_states[0] else -1.0)
                else:
                    r = 0.0
                if self.is_terminal(s):
                    done = True
                    next_s = s
                else:
                    done = False
                P[s][a] = [(1.0, next_s, r, done)]

        isd = np.zeros(nS)
        isd[0] = 1.0

        super(GridWorldEnv, self).__init__(nS, nA, P, isd)

        self.viewer = None
        self._build_display(gold_cell, trap_cells)

    def is_terminal(self, state):
        return state in self.terminal_states

    def _build_display(self, gold_cell, trap_cells):
        screen_width = (self.num_cols + 2) * cell_size
        screen_height = (self.num_rows + 2) * cell_size
        self.viewer = rendering.Viewer(screen_width, screen_height)
        all_objects = []

        bp_list = [(cell_size - margin, cell_size - margin),
                   (screen_width - cell_size + margin, cell_size - margin),
                   (screen_width - cell_size + margin, screen_height - cell_size + margin),
                   (cell_size - margin, screen_height - cell_size + margin)]
        border = rendering.PolyLine(bp_list, True)
        border.set_linewidth(5)
        all_objects.append(border)

        for col in range(self.num_cols + 1):
            x1, y1 = (col + 1) * cell_size, cell_size
            x2, y2 = (col + 1) * cell_size, (self.num_rows + 1) * cell_size
            line = rendering.PolyLine([(x1, y1), (x2, y2)], False)
            all_objects.append(line)

        for row in range(self.num_rows + 1):
            x1, y1 = cell_size, (row + 1) * cell_size
            x2, y2 = (self.num_cols + 1) * cell_size, (row + 1) * cell_size
            line = rendering.PolyLine([(x1, y1), (x2, y2)], False)
            all_objects.append(line)

        for cell in trap_cells:
            trap_coords = get_coords(*cell, loc='center')
            all_objects.append(draw_object([trap_coords]))

        gold_coords = get_coords(*gold_cell, loc='interior_triangle')
        all_objects.append(draw_object(gold_coords))

        if os.path.exists('robot-coordinates.pkl') and cell_size == 100:
            agent_coords = pickle.load(open('robot-coordinates.pkl', 'rb'))
            starting_coords = get_coords(0, 0, loc='center')
            agent_coords += np.array(starting_coords)
        else:
            agent_coords = get_coords(0, 0, loc='interior_corners')
        agent = draw_object(agent_coords)
        self.agent_trans = rendering.Transform()
        agent.add_attr(self.agent_trans)
        all_objects.append(agent)

        for obj in all_objects:
            self.viewer.add_geom(obj)

    def render(self, mode='human', done='False'):
        sleep_time = 1 if done else self.delay
        x_coord = self.s % self.num_cols
        y_coord = self.s // self.num_cols
        x_coord = (x_coord + 0) * cell_size
        y_coord = (y_coord + 0) * cell_size

        self.agent_trans.set_translation(x_coord, y_coord)
        rend = self.viewer.render(return_rgb_array=(mode == 'rgb_array'))
        # time.sleep(sleep_time)
        return rend

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def get_coords(row, col, loc='center'):
    xc = (col + 1.5) * cell_size
    yc = (row + 1.5) * cell_size
    if loc == 'center':
        return xc, yc
    elif loc == 'interior_corners':
        half_size = cell_size // 2 - margin
        xl, xr = xc - half_size, xc + half_size
        yt, yb = xc - half_size, xc + half_size
        return [(xl, yt), (xr, yt), (xr, yb), (xl, yb)]
    elif loc == 'interior_triangle':
        third_size = cell_size // 3
        x1, y1 = xc, yc + third_size
        x2, y2 = xc + third_size, yc - third_size
        x3, y3 = xc - third_size, yc - third_size
        return [(x1, y1), (x2, y2), (x3, y3)]


def draw_object(coord_list):
    obj = None
    if len(coord_list) == 1:
        obj = rendering.make_circle(int(0.45 * cell_size))
        obj_transform = rendering.Transform()
        obj.add_attr(obj_transform)
        obj_transform.set_translation(*coord_list[0])
        obj.set_color(0.2, 0.2, 0.2)
    elif len(coord_list) == 3:
        obj = rendering.FilledPolygon(coord_list)
        obj.set_color(0.9, 0.6, 0.2)
    elif len(coord_list) > 3:
        obj = rendering.FilledPolygon(coord_list)
        obj.set_color(0.4, 0.4, 0.8)
    return obj


cell_size = 100
margin = 10

if __name__ == '__main__':
    env = GridWorldEnv(5, 6)
    for i in range(1):
        s = env.reset()
        env.render(mode='human', done=False)

        while True:
            action = np.random.choice(env.nA)
            res = env.step(action)
            print('Action ', env.s, action, ' -> ', res)
            env.render(mode='human', done=res[2])
            if res[2]:
                break
    env.close()
