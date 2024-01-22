"""Robotic Arm Learning Module.

This module contains functions and utilities for implementing and managing the learning process of a robotic arm
using reinforcement learning techniques, specifically the n-step SARSA algorithm. It includes functions for
pausing execution with optional messages, choosing actions based on the epsilon-greedy strategy, performing n-step
SARSA learning on a single robot arm, and managing learning across multiple processes in parallel.

The module also provides functionality for monitoring the learning process using a queue-based communication
system, helpful in a multi-threaded or multi-process environment. Additionally, it includes utilities for
visualizing the learning progress, saving and loading learned values, and stitching learned sections together
for a comprehensive robotic arm movement strategy.

The core of the module revolves around the `learn` and `n_step_sarsa` functions, which implement the learning
algorithm. The `learn_parallel` function extends this by managing multiple learning processes in parallel.
Monitoring and visualization are handled by `monitor_queue` and `print_sections` functions, respectively.

Dependencies:
- numpy
- matplotlib
- sys
- time
- multiprocessing
- threading
- Robot_Arm (a custom module for robotic arm manipulation)

Example Usage:
The module is designed to be used in a sequential manner, starting with initializing learning parameters,
followed by parallel learning with `learn_parallel`, and finally stitching together the learned sections
for a complete learning strategy.

Note:
This module assumes the existence of a `Robot_Arm` class within the `environment` directory, which is
responsible for the actual robotic arm manipulation and learning algorithm implementations. It also uses
multi-threading and multi-processing for parallel execution and monitoring.

Authors: F. M. Sokol, N. M. Hahn, M. Ubbelohde
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('./environment')
import time
import Robot_Arm as bot
import gc
import multiprocessing
import threading

# suppress scientific notation
np.set_printoptions(suppress=True)

def debug_pause(string_to_print=None):
    """
    Pause the program execution and optionally print a message.

    This function pauses the execution of the program until the user presses Enter.
    It can optionally print a message before pausing.

    :param string_to_print: A string message to be printed before pausing.
                            If None, no message is printed.
    :type string_to_print: str, optional

    :return: None
    :rtype: NoneType
    """
    if string_to_print is not None:
        print(string_to_print)
    input("Press Enter to continue...")

def get_action_epsilon_greedy(robot, epsilon: float = 0.1, verbosity_level=0) -> int:
    """
    Explore or exploit to choose an action for the robot based on epsilon-greedy strategy.

    This function decides whether to explore a new action randomly or exploit the best known action
    from the robot's Q-values. The choice between exploration and exploitation is made based on the
    epsilon value provided. If exploring, a random action is chosen. If exploiting, the best action
    based on current Q-values is selected.

    :param robot: The robot on which the action is to be performed.
    :type robot: Six_Axis_Robot_Arm object

    :param epsilon: The probability of choosing to explore rather than exploit.
                    It ranges from 0 (always exploit) to 1 (always explore).
    :type epsilon: float

    :param verbosity_level: Level of verbosity for output messages.
                            Higher values produce more detailed output.
    :type verbosity_level: int

    :return: A tuple containing the action in the format of six float values and the action index.
    :rtype: (tuple of float, int)
    """
    if np.random.uniform(0, 1) < epsilon:
        # Explore: select a random action
        action = robot.get_random_action()
        if verbosity_level >= 2:
            print(f"  Explore with action: {action}")
        return action
    else:
        # Exploit: select the best action based on current Q-values
        current_qs = robot.get_current_qs()
        action = np.argmax(current_qs)
        action_tuple = (robot.get_action_from_dict(action), action)
        if verbosity_level >= 2:
            print(f"  Exploiting with action: {action_tuple}")
        return action_tuple

def n_step_sarsa(robot, num_episodes, alpha=0.1, gamma=0.99, epsilon=0.1, verbosity_level=0,
                 queue=None, section=None, episode_start=0):
    """
    Perform the n-step SARSA algorithm on a given robot for a specified number of episodes.

    This function implements the n-step SARSA learning algorithm. It iteratively updates the Q-values
    based on the observed rewards and the chosen actions using an epsilon-greedy policy. The function
    also allows for dynamic adjustment of the learning rate alpha and supports message passing through
    a queue for parallel execution.

    :param robot: The robot object to apply the n-step SARSA algorithm on.
    :type robot: Robot_Arm object

    :param num_episodes: The number of episodes to run the algorithm.
    :type num_episodes: int

    :param alpha: The learning rate (0 < alpha <= 1).
    :type alpha: float

    :param gamma: The discount factor (0 < gamma <= 1).
    :type gamma: float

    :param epsilon: The exploration probability (0 <= epsilon <= 1).
    :type epsilon: float

    :param verbosity_level: Level of verbosity for output messages.
    :type verbosity_level: int

    :param queue: A queue object for inter-process communication.
    :type queue: Queue

    :param section: The section or segment of the task being processed.
    :type section: any

    :param episode_start: The starting episode number.
    :type episode_start: int

    :return: Tuple containing episode lengths, algorithm name, and the final alpha value.
    :rtype: (list of int, str, float)
    """
    # n step sarsa, n = 5
    n = 5
    # list to save each episodes length
    episode_lengths = []
    # when queue is given, only write into queue every 4 seconds as to not overwhelm the queue
    # last queue update saves the time of the last update of the queue
    last_queue_update = 0

    # do num_episode episodes
    for episode in range(num_episodes):

        # decrease alpha every 100 episodes by a factor of 0.98 and as long as alpha is > 0.001
        if (alpha > 0.001) and ((episode+episode_start) % 100 == 0) and ((episode+episode_start) != 0):
            alpha *= 0.98

        # time measurement
        start_time = time.time()

        # Initialize the starting state S (choosing from the starting positions)
        robot.reset()

        # debug print
        if verbosity_level >= 2:
            print(f"Robot initialized, starting SARSA, Starting angles: {robot.get_joint_angles_rad()}, Starting position: {robot.rob.end_effector_position()} Choosing action!")
            print(f"   Initial Qs for starting_voxel:\n{robot.Q[robot.voxels_index_dict[(-500, 0, 0)]]}")

        # current state contains the current TCP of the robot
        current_state = robot.get_tcp()

        # Choose the first action A from S using Q (epsilon-greedy)
        current_action, current_action_index = get_action_epsilon_greedy(robot, epsilon, verbosity_level=verbosity_level)
        if verbosity_level >= 2:
            print('current_action: ', current_action)
            print('current_action_index: ', current_action_index)

        # number of iterations in current episode
        i = 0

        # lists to save last n values
        states = []
        actions = []
        actions_index = []
        rewards = []

        # save first state
        states.append(current_state)
        actions.append(current_action)
        actions_index.append(current_action_index)

        # weightet sum of last rewards
        G = 0

        # initialize best reward for debug printing
        best_reward = -100

        # Send debug information into queue
        if queue is not None:
            if time.time() - last_queue_update > 4:
                queue.put((section, episode, "episode"))
                last_queue_update = time.time()

        # Loop for each step of the episode
        done = False

        # Until all episodes are done
        while not done:

            # debug print
            if verbosity_level >= 2:
                print(f"\nNew loop, iteration = {i}, current Voxel: {robot.current_voxel}")
                debug_pause(f" Qs for current Voxel:\n {robot.Q[robot.voxels_index_dict[robot.current_voxel]]}")

            # Save Q
            last_q = robot.get_current_q(actions_index[0])
            if verbosity_level >= 2:
                print(f"  Q for current action {current_action} with index {current_action_index} at voxel {robot.current_voxel}: {last_q}. Doing SAR")

            # Take action A, observe R, S'
            new_pos, reward, done = robot.do_move(current_action_index)

            # debug print
            if verbosity_level >= 2:
                print(f"  New position: {new_pos}, reward: {reward}, new Voxel: {robot.current_voxel}. Doing SA")
                print(f"  Qs for new position:\n{robot.Q[robot.voxels_index_dict[robot.current_voxel]]}")

            if reward > best_reward:
                best_reward = reward

            # Choose A' from S' using Q (epsilon-greedy)
            new_action, new_action_index = get_action_epsilon_greedy(robot, epsilon, verbosity_level=verbosity_level)

            # Append states and rewards to lists
            rewards.append(reward)
            states.append(new_pos)
            actions.append(new_action)
            actions_index.append(new_action_index)

            if (len(rewards) == n):
                # calculate n-step reward
                for rew in range(len(rewards)-1):
                    G += (gamma**rew)*rewards[rew]

                last_n_q = robot.get_last_n_q(actions_index[0])

                new_q = robot.get_current_q(new_action_index)

                # debug print
                if verbosity_level >= 2: print(f"  Q for action {new_action} with index {new_action_index} at voxel {robot.current_voxel}: {new_q}. Updating Q values.")

                # Update the Q-value for the current state and action pair
                last_n_q = last_n_q + alpha * (G+gamma**n*new_q - last_n_q)

                robot.set_last_n_q(actions_index[0], last_n_q)

                # only store the last n iterations
                del (rewards[0])
                del (states[0])
                del (actions[0])
                del (actions_index[0])

            # debug print
            if verbosity_level >= 2:
                print(f"  Updated Qs for last Voxel:\n{robot.Q[robot.voxels_index_dict[robot.last_voxel]]}")

            # S <- S' <- done in robot.do_move(); A <- A'
            current_action_index = new_action_index
            current_action = new_action

            # increase number of iterations in current episode
            i += 1
            # reset G
            G = 0

            # debug print every 10000 iterations
            if (i % 10000 == 0) and verbosity_level >= 1:
                print(f"    Current_iteration {i} in episode {episode}, current_pos {new_pos}, Out of bounds: {robot.out_of_bounds_counter}")
                print(f"    Move / out of bound ratio: {robot.out_of_bounds_counter/i}, best reward: {best_reward}")
                print(f"    alpha = {alpha} gamma = {gamma} epsilon = {epsilon}")

        # time calculation
        end_time = time.time()

        # concatenate episode to episode_length for return
        episode_lengths.append(i)

        # debug print
        if verbosity_level >= 1:
            print(f"    Episode {episode} ended with length {i}. Time for eposiode: {end_time-start_time} s.  Alpha = {alpha}                            ",
                                       end='\r')

    # return algorithm name
    algo = 'sarsa_n_step'

    # debug print
    if verbosity_level >= 1:
        print(f"")

    # return length of each episodes, algorithm name and last alpha
    return episode_lengths, algo, alpha

def learn(section_length, section, min_num_episodes, alpha, gamma, epsilon, queue=None, load=False,
          max_num_cycles=5, draw_robot=False, starting_pos=None, save_plot=True):
    """
    Conduct learning on a robotic arm using the n-step SARSA algorithm and save the results.

    This function initializes a robot arm and performs learning cycles using the n-step SARSA algorithm.
    It supports loading pre-learned values, adjusting the starting position, and visualizing the robot's
    movements. The learning process continues until the robot arm successfully completes the task or
    reaches the maximum number of cycles. The learning results are saved to a file, and the learning
    progress can be plotted.

    :param section_length: Length of each section of the robot arm.
    :type section_length: int

    :param section: The specific section of the task to be learned.
    :type section: any

    :param min_num_episodes: Minimum number of episodes for each learning cycle.
    :type min_num_episodes: int

    :param alpha: The learning rate.
    :type alpha: float

    :param gamma: The discount factor.
    :type gamma: float

    :param epsilon: The exploration probability.
    :type epsilon: float

    :param queue: Queue for inter-process communication, if applicable.
    :type queue: Queue

    :param load: Whether to load pre-learned values.
    :type load: bool

    :param max_num_cycles: Maximum number of learning cycles.
    :type max_num_cycles: int

    :param draw_robot: Flag to indicate if the robot should be visualized during the process.
    :type draw_robot: bool

    :param starting_pos: Starting position of the robot, if specified.
    :type starting_pos: list or tuple

    :param save_plot: Flag to indicate if the learning plot should be saved.
    :type save_plot: bool

    :return: The final angles of the robot arm after learning completion.
    :rtype: list or tuple

    :return: None
    :rtype: NoneType

    :return: None
    :rtype: NoneType
    """
    # Learn section and save to file
    arm = bot.Robot_Arm(section_length=section_length, helix_section=section,
                        voxel_volume=2, num_axis=num_axis)

    if load is True:
        arm.load_learned_from_file()

    if starting_pos is not None:
        arm.set_starting_angles_rad(starting_pos)

    if queue is None and draw_robot is True:
        arm.show(draw_path=True, draw_voxels=True, zoom_path=True)

    # Learn until the Q values lead the arm into the finish
    episode_lengths = []
    for cycle in range(max_num_cycles):
        if queue is not None:
            queue.put((section, cycle, "cycle"))
            sarsa_verbosity_level = 0
        else:
            print(f"section: {section}, cycle {cycle}")
            sarsa_verbosity_level = 1
        eps, _, alpha = n_step_sarsa(arm, min_num_episodes, alpha, gamma, epsilon,
                                     verbosity_level=sarsa_verbosity_level, queue=queue,
                                     section=section, episode_start=(cycle*min_num_episodes))
        episode_lengths = episode_lengths + eps
        finishing_angles_last_section = arm.get_finishing_angles_rad()
        if (finishing_angles_last_section[0] == "Success") or (cycle == max_num_cycles-1):
            if queue is not None:
                queue.put((section, cycle, "done"))
            else:
                print(f"    section: {section}, cycle {cycle}, DONE!")
            break

    arm.save_learned_to_file()

    if save_plot is True:
        fig, ax = plt.subplots(figsize=(10, 10))
        #ax.set_yscale('log')
        ax.plot(episode_lengths[0:150])

        ax.set_xlabel('Episodes')
        ax.set_ylabel('Episode length')
        plt.savefig(f'learned_values_{num_axis}_axis/section_{section}_plot.png')

    if queue is None and len(episode_lengths) > 0:
        print(f"    average length of the last 100 episodes: {np.average(episode_lengths[-100:len(episode_lengths)])}")
        print(f"    last 10 episode lengths: {episode_lengths[-10:len(episode_lengths)]}\n")

    if draw_robot is True:
        arm.animate_move_along_q_values(draw_path=True, draw_voxels=True, zoom_path=True)

    # Wait a moment so all queue data can be processed and process can return
    time.sleep(2)

    return finishing_angles_last_section[1]

def print_sections(num_sections):
    """
    Print the header for sections in a formatted manner.

    This function prints a formatted header showing section numbers for a given number of sections.
    It is designed to visually organize output related to different sections in a structured format.

    :param num_sections: The total number of sections to display in the header.
    :type num_sections: int

    :return: None
    :rtype: NoneType
    """
    print("\n         ", end="")
    for i in range(int(num_sections/2)-1):
        print(f"     ", end="")
    print("Sections")
    print("        ", end="")
    for i in range(num_sections):
        print(f" {(i+section_start):03d} ", end="")
    print("\n========", end="")
    for i in range(num_sections):
        print(f"=====", end="")
    print("\nCycle  |\nEpisode|", end="\r")

def monitor_queue(total_num_sections, queue):
    """
    Monitor and display the progress of tasks using data from a queue.

    This function continuously monitors a queue for messages indicating the progress of various tasks.
    It updates the display to reflect the current status of each section in the process. The function
    handles different types of messages such as cycle updates, episode updates, and completion
    notifications. It's designed to provide a real-time progress overview in a multi-process or
    multi-threaded environment.

    :param total_num_sections: Total number of sections in the task.
    :type total_num_sections: int

    :param queue: Queue object used for inter-process or inter-thread communication.
    :type queue: Queue

    :return: None
    :rtype: NoneType

    Note: This function assumes that 'print_sections' is defined and accessible in the same scope.
    """
    print_sections(total_num_sections)
    while True:
        if not queue.empty():
            result = queue.get()
            if result[2] == "cycle":
                print("\033[A\033[C\033[C\033[C\033[C\033[C\033[C\033[C\033[C", end="")
                for i in range(result[0]-section_start):
                    print("\033[C\033[C\033[C\033[C\033[C", end="")
                print(f" {result[1]:03d}", end="\r")
                print("\033[B", end="")
            if result[2] == "episode":
                print("\033[C\033[C\033[C\033[C\033[C\033[C\033[C\033[C", end="")
                for i in range(result[0]-section_start):
                    print("\033[C\033[C\033[C\033[C\033[C", end="")
                print(f" {result[1]:04d}", end="\r")
            if result[2] == "done":
                print("\033[C\033[C\033[C\033[C\033[C\033[C\033[C\033[C", end="")
                for i in range(result[0]-section_start):
                    print("\033[C\033[C\033[C\033[C\033[C", end="")
                print(f" done", end="\r")
            if result[2] == "next":
                iteration += 1
                print(f"\n")
                print(f"Iteration: {iteration}")
                print_sections(total_num_sections)
        time.sleep(0.05)

def learn_parallel(num_episodes, alpha, gamma, epsilon, num_processes=32, use_learned=False):
    """
    Initiate parallel learning processes for a robotic task.

    This function sets up and initiates multiple parallel learning processes for a robotic task. Each process
    runs a learning algorithm on a different section of the task, with the possibility of utilizing previously
    learned values. The function manages process creation, execution, and monitoring. A queue is used for
    inter-process communication to monitor the progress of each section. The function ensures that the number
    of sections is a power of two and starts a separate monitoring thread to track and display the progress of
    the learning tasks.

    :param num_episodes: Number of episodes for each learning process.
    :type num_episodes: int

    :param alpha: Learning rate for the learning algorithm.
    :type alpha: float

    :param gamma: Discount factor for the learning algorithm.
    :type gamma: float

    :param epsilon: Exploration probability for the learning algorithm.
    :type epsilon: float

    :param num_processes: Number of parallel processes to be created for learning.
    :type num_processes: int, optional

    :param use_learned: Flag to indicate whether to use previously learned values.
    :type use_learned: bool, optional

    :return: None
    :rtype: NoneType

    Note: This function assumes that 'learn' and 'monitor_queue' functions are defined and accessible.
    """
    print("\nLearning in Parallel")

    processes = []

    queue = multiprocessing.Queue()

    total_sections = num_sections

    if not (learn_sections > 0 and (learn_sections & (learn_sections - 1)) == 0):
        print(f"Number of parallel sections not a power of 2 ({learn_sections}). Aborting.")
        return

    # Start the queue monitoring thread
    monitor_thread = threading.Thread(target=monitor_queue, args=(learn_sections, queue,))
    monitor_thread.start()

    show_robot = False
    max_num_cycles=10

    section_length = 1/num_processes

    # Create and start processes
    num_processes = total_sections

    for i in range(section_start, section_start+learn_sections):
        args = (section_length, i, num_episodes, alpha, gamma, epsilon, queue, use_learned, max_num_cycles, show_robot, None, True)
        p = multiprocessing.Process(target=learn, args=args)
        p.start()
        processes.append(p)
        time.sleep(0.2)

    # Wait for all processes to finish
    all_processes_done = False
    while all_processes_done is False:
        all_processes_done = True
        delete_processes = []
        for p in processes:
            if not p.is_alive():
                delete_processes.append(p)
            else:
                all_processes_done = False
        processes = [p for p in processes if p not in delete_processes]

    # Kill monitoring thread
    monitor_thread.join()


starting_time = time.time()

# Number of episodes per section
num_episodes = 10000
alpha = 0.1
gamma = 0.99
epsilon = 0.1

# Number of axis used by the robot arm. Either 3 or 6
num_axis = 3

num_sections = 32
section_start = 0
learn_sections = 32

learn_parallel(num_episodes, alpha, gamma, epsilon, num_processes=num_sections, use_learned=True)

print("\n\n    Parallel learning done, now going sequentially through each section and stitching them together.\n")

num_episodes = 20
alpha = 0.01
finishing_angles = None

for i in range(learn_sections):
    finishing_angles = learn(1/num_sections, i, num_episodes, alpha, gamma, epsilon, load=True,
                             max_num_cycles=100, draw_robot=False, starting_pos=finishing_angles,
                             save_plot=False)

total_time = time.time()-starting_time

arm = bot.Robot_Arm(section_length=1/num_sections, helix_section=0, voxel_volume=2, num_axis=num_axis)
arm.load_learned_from_file()

for i in range(learn_sections-1):
    arm.stitch_from_file()

print(f"MSE: {arm.calc_mse(support_points=1500)} (only correct when learning the whole helix)")
arm.animate_move_along_q_values(draw_path=True, draw_voxels=True, zoom_path=True)
arm.show(draw_path=True, draw_voxels=False, zoom_path=True, draw_q_path=True)
print(f"Total time: {total_time} seconds")
