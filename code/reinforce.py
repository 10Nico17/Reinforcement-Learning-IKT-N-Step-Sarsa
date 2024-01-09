import numpy as np
import matplotlib.pyplot as plt
import sys
import time
sys.path.append('./environment')
#import Six_Axis_Robot_Arm as bot
import Three_Axis_Robot_Arm as bot
import gc

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
        print(f"Episode {episode} ended with length {i}. Time for eposiode: {end_time-start_time} s")
        algo = 'sarsa' 
    return episode_lengths, algo



def n_step_sarsa(robot, num_episodes, alpha=0.1, gamma=0.99, epsilon=0.1, verbosity_level=0):

    # Assume robot is initialized
    n=5
    episode_lengths = []
    for episode in range(num_episodes):

        # Überprüfe, ob die aktuelle Episode ein Vielfaches von 25.000 ist
        if episode % 25000 == 0:
            # Multipliziere den aktuellen Wert von alpha mit 0.1
            alpha *= 0.1

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
        best_reward = -1

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
                print(f"Percentage of track done in furthest move: {round((best_reward + 1)*100, 3)}%")
                print(f"alpha = {alpha} gamma = {gamma} epsilon = {epsilon}")
                #robot.show(draw_path=True, draw_voxels=True, zoom_path=True)

        end_time = time.time()
        episode_lengths.append(i)
        if verbosity_level >= 1: print(f"Episode {episode} ended with length {i}. Time for eposiode: {end_time-start_time} s                            ",
                                       end='\r')
        algo = 'sarsa_n_step'

    if verbosity_level >= 1: print(f"")
    return episode_lengths, algo


"""
helix_section = 0
section_length = 1/64

# Todo: Match starting angles to ending angles
arm_1 = bot.Three_Axis_Robot_Arm(section_length=section_length, helix_section=helix_section)

#arm_1.load_learned_from_file()
#finishing_angles_section_0 = arm_1.get_finishing_angles_rad()
#arm.stitch_from_file()

#arm_1.show(draw_path=True, draw_voxels=True, zoom_path=True)

# next section
helix_section = 1

# Todo: Match starting angles to ending angles
arm_2 = bot.Three_Axis_Robot_Arm(section_length=section_length, helix_section=helix_section)

#arm_2.load_learned_from_file()

#arm.stitch_from_file()

#arm_2.show(draw_path=True, draw_voxels=True, zoom_path=True)



while(True):
    episode_lengths, algo = n_step_sarsa(arm_1, num_episodes, alpha, gamma, epsilon, verbosity_level=1)

    finishing_angles_section_0 = arm_1.get_finishing_angles_rad()
    print(f"Done: finishing_angles_section_0: {finishing_angles_section_0}")
    if finishing_angles_section_0[0] == "Success":
        arm_2.set_starting_angles_rad(finishing_angles_section_0[1])
        break

arm_1.save_learned_to_file()

while(True):
    episode_lengths, algo = n_step_sarsa(arm_2, num_episodes, alpha, gamma, epsilon, verbosity_level=1)

    finishing_angles_section_1 = arm_2.get_finishing_angles_rad()
    print(f"Done: finishing_angles_section_1: {finishing_angles_section_0}")
    if finishing_angles_section_0[0] == "Success":
        #arm_2.set_starting_angles_rad(finishing_angles_section_0[1])
        break

arm_2.save_learned_to_file()

#episode_lengths, algo = sarsa(arm, num_episodes, alpha, gamma, epsilon, verbosity_level=1)

# Write learned values to file
#arm_2.save_learned_to_file()
#arm.stitch_from_file()

#fig, ax = plt.subplots(figsize=(10, 10))
#ax.set_yscale('log')
#ax.plot(episode_lengths)
#
#ax.set_title(f"Algorithm: {algo}")
#ax.set_xlabel('Episoden')
#ax.set_ylabel('Episode length')
#
#print(f"average length of the last 100 episodes: {np.average(episode_lengths[-100:len(episode_lengths)])}")
#print(f"last 10 episode lengths: {episode_lengths[-10:len(episode_lengths)]}")
#plt.savefig(f'{algo}_plot.png')
#plt.show()

#arm_1.animate_move_along_q_values(draw_path=True, draw_voxels=True, zoom_path=True)
#arm_2.animate_move_along_q_values(draw_path=True, draw_voxels=True, zoom_path=True)

arm_1.stitch_from_file()

while(True):
    episode_lengths, algo = n_step_sarsa(arm_1, num_episodes, alpha, gamma, epsilon, verbosity_level=1)

    finishing_angles_section_0 = arm_1.get_finishing_angles_rad()
    if finishing_angles_section_0[0] == "Success":
        #arm_2.set_starting_angles_rad(finishing_angles_section_0[1])
        break

arm_1.animate_move_along_q_values(draw_path=True, draw_voxels=True, zoom_path=True)

"""

starting_time = time.time()

sections_to_learn = 4
#start_at_section = 4
section_length = 1/64
voxel_volume = 1

# Number of episodes per section
num_episodes = 1000
alpha = 0.05
gamma = 0.99
epsilon = 0.1

# Learn first section
arm = bot.Three_Axis_Robot_Arm(section_length=section_length, helix_section=0, voxel_volume=voxel_volume)

#arm.show(draw_path=True, draw_voxels=True, zoom_path=True)

# Learn until the Q values lead the arm into the finish
while(True):
    print(f"\n\n********************\nLearning Section 0\n********************\n\n")
    episode_lengths, algo = n_step_sarsa(arm, num_episodes, alpha, gamma, epsilon, verbosity_level=1)

    finishing_angles_last_section = arm.get_finishing_angles_rad()
    #print(f"Done: finishing_angles_section_0: {finishing_angles_last_section}")
    #arm.animate_move_along_q_values(draw_path=True, draw_voxels=True, zoom_path=True)
    if finishing_angles_last_section[0] == "Success":
        break

arm.save_learned_to_file()

# Learn each section using the finishing position from the section before as starting position
for current_section in range(1, sections_to_learn):

    # new arm
    arm_next_section = bot.Three_Axis_Robot_Arm(section_length=section_length, helix_section=current_section, voxel_volume=voxel_volume)

    # set starting angles of next section to finishing angles of section before
    arm_next_section.set_starting_angles_rad(finishing_angles_last_section[1])

    #print(f"Starting angles for learning: {arm_next_section.rob.q0}")

    #arm_next_section.show(draw_path=True, draw_voxels=True, zoom_path=True)

    # Learn until the Q values lead the arm into the finish
    while(True):
        print(f"\n\n********************\nLearning Section {current_section}\n********************\n\n")
        episode_lengths, algo = n_step_sarsa(arm_next_section, num_episodes, alpha, gamma, epsilon, verbosity_level=1)

        finishing_angles_last_section = arm_next_section.get_finishing_angles_rad()

        #print(f"Starting angles for getting finishing angles: {arm_next_section.rob.q0}")
        print(f"{num_episodes} done, Q values lead to: {finishing_angles_last_section[0]}")
        #arm_next_section.animate_move_along_q_values(draw_path=True, draw_voxels=True, zoom_path=True)
        if finishing_angles_last_section[0] == "Success":
            break

    # save learned section
    arm_next_section.save_learned_to_file()
    arm.stitch_from_file()

    #arm.show(draw_path=True, draw_voxels=True, zoom_path=True)

    while(True):
        print(f"\n\n**************************************\nLearning All Current Sections Stitched\n**************************************\n\n")
        episode_lengths, algo = n_step_sarsa(arm, num_episodes, alpha, gamma, epsilon, verbosity_level=1)

        finishing_angles_section_0 = arm.get_finishing_angles_rad()
        #arm.animate_move_along_q_values(draw_path=True, draw_voxels=True, zoom_path=True)
        if finishing_angles_section_0[0] == "Success":
            #arm_2.set_starting_angles_rad(finishing_angles_section_0[1])
            break


num_episodes = 10000

#arm.show(draw_path=True, draw_voxels=True, zoom_path=True)

while(True):
    print(f"\n\n***************************************************\nLearning All Sections Stitched for {num_episodes} episodes again\n***************************************************\n\n")
    episode_lengths, algo = n_step_sarsa(arm, num_episodes, alpha, gamma, epsilon, verbosity_level=1)

    finishing_angles_section_0 = arm.get_finishing_angles_rad()
    if finishing_angles_section_0[0] == "Success":
        #arm_2.set_starting_angles_rad(finishing_angles_section_0[1])
        # Only do 1000 episodes each iteration for now
        num_episodes = 1000
        break

# Save learned Q values
arm.save_learned_to_file()

total_time = time.time()-starting_time

print(f"Learned {sections_to_learn} out of {int(1/section_length)} sections in a total time of {total_time} seconds")

arm.animate_move_along_q_values(draw_path=True, draw_voxels=True, zoom_path=True)
