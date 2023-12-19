import numpy as np
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker, draw_line
from pybullet_tools.utils import connect, wait_for_user, disconnect, get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
import time
### YOUR IMPORTS HERE ###
from queue import PriorityQueue
import itertools
def hu(m, n):
    # return np.sqrt((m[0]-n[0])**2+(m[1]-n[1])**2+min(abs(m[2]-n[2]), 2*np.pi - abs(m[2]-n[2]))**2)
    return np.sqrt((m[0]-n[0])**2+ 0.6 * (m[1]-n[1])**2+ 4 * np.sin(m[2]-n[2])**2)

def distance(m, n):
    return np.sqrt((m[0]-n[0])**2+(m[1]-n[1])**2+min(abs(m[2]-n[2]), 2*np.pi - abs(m[2]-n[2]))**2)

#########################

def main(screenshot=False):
    # initialize PyBullet
    connect(use_gui=True)
    # load robot and obstacle resources
    robots, obstacles = load_env('pr2doorway.json')
    # define active DoFs
    base_joints = [joint_from_name(robots['pr2'], name) for name in PR2_GROUPS['base']]

    collision_fn = get_collision_fn_PR2(robots['pr2'], base_joints, list(obstacles.values()))
    
    start_config = tuple(get_joint_positions(robots['pr2'], base_joints))
    # print(start_config)
    goal_config = (2.6, -1.3, -np.pi/2)
    path = []
    start_time = time.time()
    ### YOUR CODE HERE ###
    dx, dy, dtheta = 0.15, 0.15, np.pi/4
    d4 = [(dx, 0, 0), (-dx, 0, 0), (0, dy, 0), (0, -dy, 0), (0, 0, dtheta), (0, 0,-dtheta)]
    d8 = [i for i in itertools.product([-dx, 0, dx], [-dy, 0, dy], [-dtheta, 0, dtheta]) if i != (0,0,0)]
    neighbour = d8
    # initialize the priority queue
    q = PriorityQueue()
    dis = {}
    dis2 = {}
    backtrace_from = None
    # root
    q.put((hu(start_config, goal_config), start_config, 0))
    dis[start_config] = (start_config, 0) # pose of parent, distance to current place
    dis2[(0,0)] = 0 # distance to current place
    while not q.empty():
        _, next_pose, next_acc = q.get()
        if distance(next_pose, goal_config) < 0.1:
            # find the path
            backtrace_from = next_pose
            print("Found cost:", next_acc)
            break
        else:
            for d in neighbour:
                # get the new pose
                new_theta = next_pose[2] + d[2]
                if new_theta > np.pi:
                    new_theta -= 2*np.pi
                elif new_theta < -np.pi:
                    new_theta += 2*np.pi
                new_pose = (next_pose[0]+d[0], next_pose[1]+d[1], new_theta)
                if collision_fn(new_pose):
                    pos = (round((new_pose[0]-start_config[0])/dx), round((new_pose[1]-start_config[1])/dy))
                    if pos in dis2:
                        dis2[pos] *= 1
                    else:
                        dis2[pos] = 1
                    continue
                # check if visited
                if new_pose in dis:
                    _, old_acc = dis[new_pose]
                else:
                    old_acc = -1
                # add to queue
                new_acc = next_acc + distance(new_pose, next_pose)
                if old_acc == -1 or new_acc < old_acc:
                    q.put((new_acc + hu(new_pose, goal_config), new_pose, new_acc))
                    dis[new_pose] = (next_pose,  new_acc)
                    dis2[(round((new_pose[0]-start_config[0])/dx), round((new_pose[1]-start_config[1])/dy))] = 0
    if backtrace_from is not None:   
        # backtrace
        while backtrace_from != start_config:
            path.append(backtrace_from)
            backtrace_from = dis[backtrace_from][0]
        path.append(start_config)
        path.reverse()
    print("Path length: ", len(path))
    ######################
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