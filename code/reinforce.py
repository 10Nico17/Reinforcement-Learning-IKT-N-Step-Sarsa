import numpy as np
import matplotlib.pyplot as plt
import sys
import time
sys.path.append('./environment')
import Six_Axis_Robot_Arm as bot
#import Three_Axis_Robot_Arm as bot
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

    :param verbosity_level: Debug print verbosity level:
                                - <= 1: no output
                                - >= 2: print action
    :type verbosity_level: int

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

def n_step_sarsa(robot, num_episodes, alpha=0.1, gamma=0.99, epsilon=0.1, verbosity_level=0, queue=None, section=None, episode_start=0):
    """Do n-step SARSA algorithm.

    Use and train the Q values given by in the robot environment.

    :param robot: Robot to do the action on
    :type robot: Six_Axis_Robot_Arm object

    :param num_episodes: Number of episodes to do
    :type num_episodes: int

    :param alpha: Reenforcement learning alpha value
    :type alpha: float

    :param gamma: Reenforcement learning gamma value
    :type gamma: float

    :param epsilon: Probability to explore
    :type epsilon: float

    :param verbosity_level: Debug print verbosity level:
                                - <= 1: no output
                                - >= 2: print action
    :type verbosity_level: int

    :param episode_start: The actual episode the robot is in
    :type episode_start: int
    """
    # Assume robot is initialized
    n=5
    episode_lengths = []
    last_queue_update = 0

    alpha_incr = 1
    for episode in range(num_episodes):

        if (alpha > 0.007) and ((episode+episode_start) % 100 == 0) and ((episode+episode_start) != 0):
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

        states = []
        actions = []
        actions_index = []
        rewards = []

        states.append(current_state)
        actions.append(current_action)
        actions_index.append(current_action_index)

        i = 0
        G = 0
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
            G = 0

            # Do debug printing every 10k episodes
            if (i % 10000 == 0) and verbosity_level >= 1:
                print(f"Current_iteration {i} in episode {episode}, current_pos {new_pos}, Out of bounds: {robot.out_of_bounds_counter}")
                print(f"Move / out of bound ratio: {robot.out_of_bounds_counter/i}, best reward: {best_reward}")
                #print(f"Percentage of track done in furthest move: {round((best_reward + 1)*100, 3)}%")
                print(f"alpha = {alpha} gamma = {gamma} epsilon = {epsilon}")
                print(f"Current Checkpoint: {robot.current_checkpoint}")
                #robot.show(draw_path=True, draw_voxels=True, zoom_path=True)

        end_time = time.time()
        episode_lengths.append(i)
        if verbosity_level >= 1: print(f"Episode {episode} ended with length {i}. Time for eposiode: {end_time-start_time} s.  Alpha = {alpha}                            ",
                                       end='\r')
        algo = 'sarsa_n_step'

    if verbosity_level >= 1: print(f"")
    return episode_lengths, algo, alpha

starting_time = time.time()

# Number of episodes per cycle
num_episodes = 25000
alpha = 0.1
gamma = 0.99
epsilon = 0.1

# Learn section and save to file
arm = bot.Six_Axis_Robot_Arm(voxel_volume=2, checkpoints=True, num_checkpoints=256)

arm.load_learned_from_file()

episode_lengths = []
max_num_cycles = 500
section = 0

for cycle in range(max_num_cycles):

    print(f"cycle {cycle}")

    eps, _, alpha = n_step_sarsa(arm, num_episodes, alpha, gamma, epsilon, verbosity_level=1, queue=None, section=0, episode_start=(cycle*num_episodes))
    episode_lengths = episode_lengths + eps
    finishing_angles = arm.get_finishing_angles_rad(max_steps=2000)

    arm.save_learned_to_file()

    if (finishing_angles[0] == "Success") or (cycle == max_num_cycles-1):
        break

    #bot.set_use_checkpoints(False)

    num_episodes = 2
    alpha = 0.008
    gamma = 0.99
    epsilon = 0.1

print(f"section: {section}, cycle {cycle}, DONE!, {finishing_angles[0]}")

fig, ax = plt.subplots(figsize=(10, 10))

ax.plot(episode_lengths)

ax.set_xlabel('Episodes')
ax.set_ylabel('Episode length')

print(f"average length of the last 100 episodes: {np.average(episode_lengths[-100:len(episode_lengths)])}")
print(f"last 10 episode lengths: {episode_lengths[-10:len(episode_lengths)]}")
plt.savefig(f'Sarsa_n_step_plot.png')

plt.show()

arm.animate_move_along_q_values(draw_path=True, draw_voxels=False, zoom_path=True, max_steps=2000, inverse_zoom_factor=5)

total_time = time.time()-starting_time

print(f"Total time: {total_time} seconds")
