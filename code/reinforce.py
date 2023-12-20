import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('./environment')
#import Six_Axis_Robot_Arm as bot
import Three_Axis_Robot_Arm as bot

# suppress scientific notation
np.set_printoptions(suppress=True)

def debug_pause(string_to_print=None):
    if string_to_print is not None:
        print(string_to_print)
    input("Press Enter to continue...")

def get_action_epsilon_greedy(robot, epsilon: float = 0.1, verbosity_level = 0) -> int:
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
        action = np.argmax(robot.get_current_qs())
        action_tuple = (robot.get_action_from_dict(action), action)
        if verbosity_level >= 2: print(f"  Exploiting with action: {action_tuple}")
        return action_tuple

def sarsa(robot, num_episodes, alpha=0.1, gamma=1.0, epsilon=0.1, verbosity_level=0):
    # Assume robot is initialized
    episode_lengths = []
    for episode in range(num_episodes):
        # Initialize the starting state S (choosing from the starting positions)
        robot.reset()

        if verbosity_level >= 2: print(f"Robot initialized, starting SARSA, Starting angles: {robot.get_joint_angles_rad()}, Starting position: {robot.rob.end_effector_position()} Choosing action!")
        if verbosity_level >= 3: print(f"   Initial Qs for starting_voxel:\n{robot.Q[robot.voxels_index_dict[(-5000, 0, 0)]]}")

        # Choose the first action A from S using Q (epsilon-greedy)
        current_action, current_action_index = get_action_epsilon_greedy(robot, epsilon, verbosity_level=verbosity_level)

        # Loop for each step of the episode
        done = False
        i = 0

        # remember best reward
        best_reward = -1

        while not done:
            if verbosity_level >= 2: debug_pause(f"New loop, iteration = {i}")
            # Save Q
            last_q = robot.get_current_q(current_action_index)
            if verbosity_level >= 2: print(f"  Q for action {current_action} with index {current_action_index} at voxel {robot.current_voxel}: {last_q}. Doing SAR")

            # Take action A, observe R, S'
            new_pos, reward, done = robot.do_move(current_action_index)
            if verbosity_level >= 2: print(f"  New angles: {robot.get_joint_angles_rad()}, new position: {new_pos}")
            if verbosity_level >= 2: print(f"  reward: {reward}, done: {done}. Doing SA")
            if reward > best_reward:
                best_reward=reward

            # Choose A' from S' using Q (epsilon-greedy)
            new_action, new_action_index = get_action_epsilon_greedy(robot, epsilon, verbosity_level=verbosity_level)
            new_q = robot.get_current_q(new_action_index)
            if verbosity_level >= 2: print(f"  Q for action {new_action} with index {new_action_index} at voxel {robot.current_voxel}: {new_q}. Updating Q values.")

            # Update the Q-value for the current state and action pair
            new_q = last_q + alpha * (reward + gamma * new_q - last_q)
            robot.set_last_q(current_action_index, new_q)
            if verbosity_level >= 2: print(f"  Updated Q value: {new_q}")

            if verbosity_level >= 3: print(f"  Qs for starting_voxel:\n{robot.Q[robot.voxels_index_dict[(-5000, 0, 0)]]}")

            # S <- S' <- done in robot.do_move(); A <- A'
            current_action_index = new_action_index
            current_action = new_action
            i+=1
            if (i % 10000 == 0) and verbosity_level >= 1:
                print(f"Current_iteration {i} in episode {episode}, current_pos {new_pos}, Out of bounds: {robot.out_of_bounds_counter}")
                print(f"Move / out of bound ratio: {robot.out_of_bounds_counter/i}, best reward: {best_reward}")
                print(f"Percentage of track done in furthest move: {round((best_reward + 1)*100, 3)}%")
                print(f"alpha = {alpha} gamma = {gamma} epsilon = {epsilon}")
                #robot.show(draw_path=True, draw_voxels=True, zoom_path=True)

        episode_lengths.append(i)
        print(f"Episode {episode} ended with length {i}")
    return episode_lengths

#arm = bot.Six_Axis_Robot_Arm()
arm = bot.Three_Axis_Robot_Arm()
#arm.show(draw_path=True, draw_voxels=True, zoom_path=True)

num_episodes = 1000
alpha = 1/25
gamma = 0.99
epsilon = 0.1

episode_lengths = sarsa(arm, num_episodes, alpha, gamma, epsilon, verbosity_level=1)

fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(episode_lengths)
print(f"average length of the last 100 episodes: {np.average(episode_lengths[-100:len(episode_lengths)])}")
print(f"last 10 episode lengths: {episode_lengths[-10:len(episode_lengths)]}")
plt.show()
