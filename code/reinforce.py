import numpy as np
import matplotlib.pyplot as plt
import sys
import time
sys.path.append('./environment')
#import Six_Axis_Robot_Arm as bot
import Three_Axis_Robot_Arm as bot
import gc
import multiprocessing
import threading
import cProfile

# suppress scientific notation
np.set_printoptions(suppress=True)

def debug_pause(string_to_print=None):
    if string_to_print is not None:
        print(string_to_print)
    input("Press Enter to continue...")

def get_action_epsilon_greedy(robot, epsilon: float = 0.1, verbosity_level=0) -> int:
    """Explore / Exploit, choose action from Q.

    :param robot: Robot to do the action on
    :type robot: Six_Axis_Robot_Arm object

    :param epsilon: Probability to explore
    :type epsilon: float

    :return: tuple with action index from robot action dict and action
    :rtype: ((float, float, float, float, float, float), int)
    """
    if np.random.uniform(0, 1) < epsilon:
        # Explore: select a random action
        action = robot.get_random_action()
        if verbosity_level >= 2: print(f"  Explore with action: {action}")
        return action
    else:
        # Exploit: select the best action based on current Q-values
        current_qs = robot.get_current_qs()
        action = np.argmax(current_qs)
        action_tuple = (robot.get_action_from_dict(action), action)
        if verbosity_level >= 2: print(f"  Exploiting with action: {action_tuple}")
        return action_tuple



def sarsa(robot, num_episodes, alpha=0.1, gamma=1.0, epsilon=0.1, verbosity_level=0):
    # Assume robot is initialized
    episode_lengths = []
    for episode in range(num_episodes):
        start_time = time.time()
        # Initialize the starting state S (choosing from the starting positions)
        robot.reset()
        if verbosity_level >= 2: print(f"Robot initialized, starting SARSA, Starting angles: {robot.get_joint_angles_rad()}, Starting position: {robot.rob.end_effector_position()} Choosing action!")
        if verbosity_level >= 2: print(f"   Initial Qs for starting_voxel:\n{robot.Q[robot.voxels_index_dict[(-500, 0, 0)]]}")

        # Choose the first action A from S using Q (epsilon-greedy)
        current_action, current_action_index = get_action_epsilon_greedy(robot, epsilon, verbosity_level=verbosity_level)

        # Loop for each step of the episode
        done = False
        i = 0

        # remember best reward
        best_reward = -1

        while not done:
            if verbosity_level >= 2: print(f"\nNew loop, iteration = {i}, current Voxel: {robot.current_voxel}")
            if verbosity_level >= 2: debug_pause(f" Qs for current Voxel:\n {robot.Q[robot.voxels_index_dict[robot.current_voxel]]}")

            # Save Q
            last_q = robot.get_current_q(current_action_index)

            if verbosity_level >= 2: print(f"  Q for current action {current_action} with index {current_action_index} at voxel {robot.current_voxel}: {last_q}. Doing SAR")

            # Take action A, observe R, S'
            new_pos, reward, done = robot.do_move(current_action_index)

            if verbosity_level >= 2: print(f"  New position: {new_pos}, reward: {reward}, new Voxel: {robot.current_voxel}. Doing SA")
            if verbosity_level >= 2: print(f"  Qs for new position:\n{robot.Q[robot.voxels_index_dict[robot.current_voxel]]}")
            if reward > best_reward:
                best_reward = reward


            # Choose A' from S' using Q (epsilon-greedy)
            new_action, new_action_index = get_action_epsilon_greedy(robot, epsilon, verbosity_level=verbosity_level)

            new_q = robot.get_current_q(new_action_index)

            if verbosity_level >= 2: print(f"  Q for action {new_action} with index {new_action_index} at voxel {robot.current_voxel}: {new_q}. Updating Q values.")

            # Update the Q-value for the current state and action pair
            new_q = last_q + alpha * (reward + gamma * new_q - last_q)
            robot.set_last_q(current_action_index, new_q)


            if verbosity_level >= 2: print(f"  Updated Q value: {new_q}")

            if verbosity_level >= 2: print(f"  Updated Qs for last Voxel:\n{robot.Q[robot.voxels_index_dict[robot.last_voxel]]}")

            # S <- S' <- done in robot.do_move(); A <- A'
            current_action_index = new_action_index
            current_action = new_action
            i += 1
            if (i % 10000 == 0) and verbosity_level >= 1:
                print(f"Current_iteration {i} in episode {episode}, current_pos {new_pos}, Out of bounds: {robot.out_of_bounds_counter}")
                print(f"Move / out of bound ratio: {robot.out_of_bounds_counter/i}, best reward: {best_reward}")
                print(f"Percentage of track done in furthest move: {round((best_reward + 1)*100, 3)}%")
                print(f"alpha = {alpha} gamma = {gamma} epsilon = {epsilon}")
                #robot.show(draw_path=True, draw_voxels=True, zoom_path=True)

        end_time = time.time()
        episode_lengths.append(i)
        print(f"Episode {episode} ended with length {i}. Time for eposiode: {end_time-start_time} s.")
        algo = 'sarsa' 
    return episode_lengths, algo



def n_step_sarsa(robot, num_episodes, alpha=0.1, gamma=0.99, epsilon=0.1, verbosity_level=0, queue=None, section=None, episode_start=0):

    # Assume robot is initialized
    n=5
    episode_lengths = []
    last_queue_update = 0

    alpha_incr = 1
    for episode in range(num_episodes):

        if (alpha > 0.001) and ((episode+episode_start) % 100 == 0) and ((episode+episode_start) != 0):
            alpha_incr += 1
            alpha *= 0.98

        start_time = time.time()
        # Initialize the starting state S (choosing from the starting positions)
        robot.reset()

        if verbosity_level >= 2: print(f"Robot initialized, starting SARSA, Starting angles: {robot.get_joint_angles_rad()}, Starting position: {robot.rob.end_effector_position()} Choosing action!")
        if verbosity_level >= 2: print(f"   Initial Qs for starting_voxel:\n{robot.Q[robot.voxels_index_dict[(-500, 0, 0)]]}")

        current_state = robot.get_tcp()

        # Choose the first action A from S using Q (epsilon-greedy)
        current_action, current_action_index = get_action_epsilon_greedy(robot, epsilon, verbosity_level=verbosity_level)
        if verbosity_level >= 2: print('current_action: ', current_action)
        if verbosity_level >= 2: print('current_action_index: ', current_action_index)

        # Loop for each step of the episode
        done = False
        i = 0

        states = []
        actions = []
        actions_index = []
        rewards = []

        states.append(current_state)
        actions.append(current_action)
        actions_index.append(current_action_index)

        G=0
        best_reward = -100

        # Send information into queue
        if queue is not None:

            if time.time() - last_queue_update > 4:
                queue.put((section, episode, "episode"))
                last_queue_update = time.time()

        while not done:
            if verbosity_level >= 2: print(f"\nNew loop, iteration = {i}, current Voxel: {robot.current_voxel}")
            if verbosity_level >= 2: debug_pause(f" Qs for current Voxel:\n {robot.Q[robot.voxels_index_dict[robot.current_voxel]]}")

            # Save Q
            last_q = robot.get_current_q(actions_index[0])
            if verbosity_level >= 2: print(f"  Q for current action {current_action} with index {current_action_index} at voxel {robot.current_voxel}: {last_q}. Doing SAR")


            # Take action A, observe R, S'
            new_pos, reward, done = robot.do_move(current_action_index)


            if verbosity_level >= 2: print(f"  New position: {new_pos}, reward: {reward}, new Voxel: {robot.current_voxel}. Doing SA")
            if verbosity_level >= 2: print(f"  Qs for new position:\n{robot.Q[robot.voxels_index_dict[robot.current_voxel]]}")
            if reward > best_reward:
                best_reward = reward

            # Choose A' from S' using Q (epsilon-greedy)
            new_action, new_action_index = get_action_epsilon_greedy(robot, epsilon, verbosity_level=verbosity_level)

            rewards.append(reward)
            states.append(new_pos)
            actions.append(new_action)
            actions_index.append(new_action_index)


            if(len(rewards)==n):
                # calculate n-step reward
                for rew in range(len(rewards)-1):
                    G += (gamma**rew)*rewards[rew]

                last_n_q = robot.get_last_n_q(actions_index[0])

                new_q = robot.get_current_q(new_action_index)
                if verbosity_level >= 2: print(f"  Q for action {new_action} with index {new_action_index} at voxel {robot.current_voxel}: {new_q}. Updating Q values.")

                # Update the Q-value for the current state and action pair
                last_n_q = last_n_q + alpha * (G+gamma**n*new_q - last_n_q)

                robot.set_last_n_q(actions_index[0], last_n_q)


                #if verbosity_level >= 2: print(f"  Updated Q value: {new_q}")
                # only store the last n iterations
                del(rewards[0])
                del(states[0])
                del(actions[0])
                del(actions_index[0])


            if verbosity_level >= 2: print(f"  Updated Qs for last Voxel:\n{robot.Q[robot.voxels_index_dict[robot.last_voxel]]}")
            # S <- S' <- done in robot.do_move(); A <- A'
            current_action_index = new_action_index
            current_action = new_action

            i += 1
            G=0

            if (i % 10000 == 0) and verbosity_level >= 1:
                print(f"Current_iteration {i} in episode {episode}, current_pos {new_pos}, Out of bounds: {robot.out_of_bounds_counter}")
                print(f"Move / out of bound ratio: {robot.out_of_bounds_counter/i}, best reward: {best_reward}")
                #print(f"Percentage of track done in furthest move: {round((best_reward + 1)*100, 3)}%")
                print(f"alpha = {alpha} gamma = {gamma} epsilon = {epsilon}")
                #robot.show(draw_path=True, draw_voxels=True, zoom_path=True)

        end_time = time.time()
        episode_lengths.append(i)
        if verbosity_level >= 1: print(f"Episode {episode} ended with length {i}. Time for eposiode: {end_time-start_time} s.  Alpha = {alpha}                            ",
                                       end='\r')
        algo = 'sarsa_n_step'

    if verbosity_level >= 1: print(f"")
    return episode_lengths, algo, alpha

def learn(section_length, section, min_num_episodes, alpha, gamma, epsilon, queue=None, load=False,
          stitch=False, max_num_cycles=5, stitch_section=1, use_norm_rewarding=False, draw_robot=False,
          starting_pos=None, save_plot=True):
    # Learn section and save to file
    arm = bot.Three_Axis_Robot_Arm(section_length=section_length, helix_section=section, voxel_volume=2, stitch_section=1, use_norm_rewarding=use_norm_rewarding)

    if load is True:
        arm.load_learned_from_file()

    if stitch is True:
        if stitch_section == "all":
            total_sections = int(1/section_length)
            for i in range(total_sections):
                arm.stitch_from_file()
        else:
            arm.stitch_from_file()

    if starting_pos is not None:
        arm.set_starting_angles_rad(starting_pos)

    if queue is None and draw_robot is True:
        arm.show(draw_path=True, draw_voxels=True, zoom_path=True)

    #if queue is None:
    #    arm.animate_move_along_q_values(draw_path=True, draw_voxels=True, zoom_path=True)

    # Learn until the Q values lead the arm into the finish
    episode_lengths = []
    for cycle in range(max_num_cycles):
        if queue is not None:
            queue.put((section, cycle, "cycle"))
            sarsa_verbosity_level = 0
        else:
            print(f"section: {section}, cycle {cycle}")
            sarsa_verbosity_level = 1
        eps, _, alpha = n_step_sarsa(arm, min_num_episodes, alpha, gamma, epsilon, verbosity_level=sarsa_verbosity_level, queue=queue, section=section, episode_start=(cycle*min_num_episodes))
        episode_lengths = episode_lengths + eps
        #queue.put(f"\n\n********************\nLearning Section {section}\n********************\n\n")
        #print(n_step_sarsa(arm, min_num_episodes, alpha, gamma, epsilon, verbosity_level=sarsa_verbosity_level, queue=queue, section=section, episode_start=(cycle*min_num_episodes))[0])
        finishing_angles_last_section = arm.get_finishing_angles_rad()
        #print(f"Done: finishing_angles_section_0: {finishing_angles_last_section}")
        #if queue is None:
        #    arm.animate_move_along_q_values(draw_path=True, draw_voxels=True, zoom_path=True)
        if (finishing_angles_last_section[0] == "Success") or (cycle == max_num_cycles-1):
            if queue is not None:
                queue.put((section, cycle, "done"))
            else:
                print(f"section: {section}, cycle {cycle}, DONE!")
            break

    if stitch is False:
        arm.save_learned_to_file(recalculate_rewards=False)
    else:
        arm.save_learned_to_file(recalculate_rewards=False)

    if save_plot is True:
        fig, ax = plt.subplots(figsize=(10, 10))
        #ax.set_yscale('log')
        ax.plot(episode_lengths[0:150])

        ax.set_xlabel('Episodes')
        ax.set_ylabel('Episode length')
        if stitch_section is True:
            plt.savefig(f'section_{section}_to_{stitch_section}_plot.png')
        else:
            plt.savefig(f'section_{section}_plot.png')

    if queue is None and len(episode_lengths) > 0:
        print(f"average length of the last 100 episodes: {np.average(episode_lengths[-100:len(episode_lengths)])}")
        print(f"last 10 episode lengths: {episode_lengths[-10:len(episode_lengths)]}")
    #plt.show()
    # Wait a moment so all queue data can be processed and process can return
    if draw_robot is True:
        arm.animate_move_along_q_values(draw_path=True, draw_voxels=True, zoom_path=True)

    return finishing_angles_last_section[1]

def print_sections(num_sections):
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
    iteration = 0
    print(f"Iteration: {iteration}")
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

def learn_parallel(num_episodes, alpha, gamma, epsilon, num_processes=64, use_learned=False):
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

    stitch = False
    use_norm_rewarding=True
    show_robot = False
    max_num_cycles=10

    section_length = 1/num_processes

    # Create and start processes
    num_processes = total_sections

    for i in range(section_start, section_start+learn_sections):
        p = multiprocessing.Process(target=learn, args=(section_length, i, num_episodes, alpha, gamma, epsilon, queue, use_learned, False, max_num_cycles, 1, False, False))
        p.start()
        processes.append(p)
        time.sleep(0.2)

    # Wait for all processes to finish
    all_processes_done = False
    while all_processes_done == False:
        all_processes_done = True
        delete_processes = []
        for p in processes:
            if not p.is_alive():
                #print(f"\nDeleting process: {p}\n")
                delete_processes.append(p)
            else:
                all_processes_done = False
        processes = [p for p in processes if p not in delete_processes]

starting_time = time.time()

# Number of episodes per section
num_episodes = 10000
alpha = 0.1
gamma = 0.99
epsilon = 0.1

# Adjust print options
np.set_printoptions(threshold=np.inf)

# Learn section and save to file
#arm = bot.Three_Axis_Robot_Arm(section_length=1/64, helix_section=1, voxel_volume=1, stitch_section=1, use_norm_rewarding=True)
#
#arm.load_learned_from_file()
#arm.animate_move_along_q_values(draw_path=True, draw_voxels=True, zoom_path=True)
#arm.stitch_from_file()

# Learn section and save to file

#learn(1/num_sections, 0, num_episodes, alpha, gamma, epsilon, load=True, max_num_cycles=5)


#learn(1/num_sections, 1, num_episodes, alpha, gamma, epsilon, load=False, max_num_cycles=5, use_norm_rewarding=False)
num_sections=32
section_start=0
learn_sections=32

#learn_parallel(num_episodes, alpha, gamma, epsilon, num_processes=num_sections, use_learned=True)

num_episodes = 20
alpha = 0.01

finishing_angles = None

#for i in range(32):
#    finishing_angles = learn(1/num_sections, i, num_episodes, alpha, gamma, epsilon, load=True, stitch=False, stitch_section=1, max_num_cycles=100, use_norm_rewarding=False, draw_robot=False, starting_pos=finishing_angles, save_plot=False)

#finishing_angles = learn(1/num_sections, 0, num_episodes, alpha, gamma, epsilon, load=True, stitch=False, stitch_section=1, max_num_cycles=5, use_norm_rewarding=False, draw_robot=True)
#learn(1/num_sections, 1, num_episodes, alpha, gamma, epsilon, load=True, stitch=False, stitch_section=1, max_num_cycles=5, use_norm_rewarding=False, draw_robot=True, starting_pos=finishing_angles)

total_time = time.time()-starting_time

arm = bot.Three_Axis_Robot_Arm(section_length=1/num_sections, helix_section=0, voxel_volume=2, stitch_section=1, use_norm_rewarding=False)
arm.load_learned_from_file()
for i in range(32):
    arm.stitch_from_file()

print(f"MSE: {arm.calc_mse(support_points=1500)}")
#arm.animate_move_along_q_values(draw_path=True, draw_voxels=True, zoom_path=True)
arm.show(draw_path=True, draw_voxels=False, zoom_path=True, draw_q_path=True)

print(f"Total time: {total_time} seconds")
