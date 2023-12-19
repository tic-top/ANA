import numpy as np
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker, draw_line
from pybullet_tools.utils import connect, wait_for_user, disconnect, get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
import time
### YOUR IMPORTS HERE ###
from queue import PriorityQueue as PQ
import itertools
G = 1e10
E = 1e10
F = None
S = None
dis2 = {}
dis2[(0,0)] = 0
log = []


def hu(m, n):
    return np.sqrt((m[0]-n[0])**2+(m[1]-n[1])**2+min(abs(m[2]-n[2]), 2*np.pi - abs(m[2]-n[2]))**2)
    # return np.sqrt((m[0]-n[0])**2+ 0.6 * (m[1]-n[1])**2+ 4 * np.sin(m[2]-n[2])**2)

def distance(m, n):
    return np.sqrt((m[0]-n[0])**2+(m[1]-n[1])**2+min(abs(m[2]-n[2]), 2*np.pi - abs(m[2]-n[2]))**2)


class RPQ(PQ):

	def put(self,xy):
		n_xy = xy[0] * (-1), xy[1], xy[2]
		PQ.put(self, n_xy)
	
	def get(self):
		xy = PQ.get(self)
		n_xy = xy[0] * (-1), xy[1], xy[2]
		return n_xy
    
# pose expandion
dx, dy, dtheta = 0.15, 0.15, np.pi/4
d8 = [i for i in itertools.product([-dx, 0, dx], [-dy, 0, dy], [-dtheta, 0, dtheta]) if i != (0,0,0)]
neighbour = d8
def expand(pose, fn):
    global dis2, S
    next_pose = []
    for d in neighbour:
        new_theta = pose[2] + d[2]
        if new_theta > np.pi:
            new_theta -= 2*np.pi
        elif new_theta < -np.pi:
            new_theta += 2*np.pi
        new_pose = (pose[0]+d[0], pose[1]+d[1], new_theta)
        if fn(new_pose):
            pos = (round((new_pose[0]-S[0])/dx), round((new_pose[1]-S[1])/dy))
            if pos in dis2:
                dis2[pos] *= 1
            else:
                dis2[pos] = 1
            continue
        next_pose.append(new_pose)
    return next_pose

# prune
def prune(OPEN, G, goal):
    NEW_OPENE = RPQ(0)
    while not OPEN.empty():
        pose = OPEN.get()
        g_s = pose[1]
        state = pose[2]
        h_s = hu(state, goal)

        if g_s + h_s < G:
            n_s = (G - g_s)/(h_s+0.00001)
            NEW_OPENE.put((n_s, g_s, state))
    return NEW_OPENE

# improve_solution
def improve_solution(pred, costs, OPEN, G, E, fn, goal):
    global dis2, S
    while not OPEN.empty():
        pose = OPEN.get()
        e_s = pose[0]
        g_s = pose[1]
        state = pose[2]
        if e_s < E:
            E = e_s
        if distance(state, goal) < 0.1:
            G = g_s
            global F
            F = state
            break

        for next_pose in expand(state, fn):
            new_cost = costs[state] + distance(state, next_pose)
            if next_pose not in costs or new_cost < costs[next_pose]:
                costs[next_pose] = new_cost
                h_next = hu(next_pose, goal)
                if new_cost + h_next < G:
                    e_next_pose = (G - new_cost)/(h_next+0.00001)
                    OPEN.put((e_next_pose, new_cost, next_pose))
                    dis2[(round((next_pose[0]-S[0])/dx), round((next_pose[1]-S[1])/dy))] = 0
                pred[next_pose] = state
    return pred, costs, OPEN, G, E

# ANA * 
def ANA(fn, start, goal):
    global G, E, F, log
    pred = {}
    costs = {}
    pred[start] = None
    costs[start] = 0
    cnt = 0
    h_start = hu(start, goal)
    e_start = (G - 0)/(h_start+0.00001)
    OPEN = RPQ(0)
    OPEN.put((e_start, 0, start))
    
    f_time = time.time()
    while not OPEN.empty():
        st_time = time.time()
        pred, costs, OPEN, G, E = improve_solution(pred, costs, OPEN, G, E, fn, goal)
        OPEN = prune(OPEN, G, goal)
        cnt += 1
        log.append([time.time()-st_time, G])
        print(log)
        if (time.time()-f_time>1000):
            break

    path = []
    current_pose = F
    while current_pose!= start:
        path.append(current_pose)
        current_pose = pred[current_pose]
    path.append(start)
    path.reverse()
    frontier = {}
    for i in OPEN.queue:
        frontier[i[2]] = i[1]
    
    # G
    print("G cost:", G)
    
    return path, frontier, costs
#########################

def main(screenshot=False):
    # initialize PyBullet
    connect(use_gui=True)
    # load robot and obstacle resources
    time.sleep(1)
    robots, obstacles = load_env('pr2doorway.json')
    # define active DoFs
    base_joints = [joint_from_name(robots['pr2'], name) for name in PR2_GROUPS['base']]

    collision_fn = get_collision_fn_PR2(robots['pr2'], base_joints, list(obstacles.values()))
    
    start_config = tuple(get_joint_positions(robots['pr2'], base_joints))
    goal_config = (2.6, -1.3, -np.pi/2)
    
    start_time = time.time()
    ### YOUR CODE HERE ###
    global S
    S = start_config
    path, frontier, costs = ANA(collision_fn, start_config, goal_config)
    print("Path length: ", len(path))
    print("Frontier size: ", len(frontier))

    ######################
    print(log)
    print("Planner run time: ", time.time() - start_time)

    sphere_radius = 0.05
    red = (1, 0, 0, 0.5)
    blue = (0, 1, 1, 0.5)
    black = (0, 0, 0)
    for item in dis2.items():
        pos, occ = item
        x, y = pos
        sphere_position = (x * dx + start_config[0], y * dy + start_config[1], 0.1)
        sphere_color = blue if occ == 0 else red
        draw_sphere_marker(sphere_position, sphere_radius, sphere_color)
    for i in range(len(path)-1):
        line_start = (path[i][0], path[i][1], 0.2)
        line_end = (path[i+1][0], path[i+1][1], 0.2)
        line_width = 200
        draw_line(line_start, line_end, line_width, black)
    # Execute planned path
    execute_trajectory(robots['pr2'], base_joints, path, sleep=0.2)
    # Keep graphics window opened
    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main()