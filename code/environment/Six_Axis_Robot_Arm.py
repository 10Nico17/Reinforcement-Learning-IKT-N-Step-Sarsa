"""Robot environment for the task."""

import numpy as np
import matplotlib.pyplot as plt
#import mplot3d as plt3d
# maddux for robot visualization
from maddux.robots.link import Link
from maddux.robots.arm import Arm
from maddux.environment import Environment
import time
from path import Path
import itertools


class Six_Axis_Robot_Arm:
    """Class simulating the robot arm."""

    def __init__(self, starting_pos: (float, float, float) = (-500, 0, 0)) -> None:
        """Initialize robot arm.

        :param initial_angles: Tuple with the initial angles of the robot joints in degrees
        :type initial_angles: (float, float, float, float, float, float)

        :param path: Coordinates of path to draw
        :type path: list of touples of ints

        :return: None
        """
        # Create path for the robot
        path = Path(helix_start=starting_pos, max_distance=1)
        self.voxels, self.winning_voxels, self.rewards = path.get_helix_voxels()
        self.voxel_size = 1
        self.path = path.get_helix_data()
        amount_voxels = len(self.voxels)

        # Create hashtable of voxels with a unique index for each voxel
        self.voxels_index_dict = {value: index for index, value in enumerate(self.voxels)}

        # Create a series of links (each link has one joint)
        # (theta, offset, length, twist, q_lim=None)
        L1 = Link(0, 151.85, 0, 1.570796327, link_size=5)
        L2 = Link(0, 0, -243.55, 0, link_size=4)
        L3 = Link(0, 0, -213.2, 0, link_size=4)
        L4 = Link(0, 131.05, 0, 1.570796327, link_size=3)
        L5 = Link(0, 85.35, 0, -1.570796327, link_size=2)
        L6 = Link(0, 92.1, 0, 0, link_size=0.1)
        links = np.array([L1, L2, L3, L4, L5, L6])

        # Caculate starting angles from starting position and
        # convert degrees of starting position to radiants

        # Initial arm angles
        q0 = np.array((0, 0, 0, 0, 0, 0))

        # Create arm
        self.rob = Arm(links, q0, '1-link')

        # Do inverse kinematics for the starting position and
        # create a new arm with the starting position
        self.starting_angles = self.__limit_angles(self.rob.ikine((self.path[0][0], self.path[1][0], self.path[2][0])))
        # Create arm
        self.rob = Arm(links, self.starting_angles, '1-link')

        self.env = Environment(dimensions=[1500.0, 1500.0, 1500.0],
                               robot=self.rob)

        # Create all possible actions
        # Define possible actions for each joint in deg
        # For now 1 degree per action, as the robot will take forever otherwise
        joint_actions_deg = [-0.1, 0, 0.1]
        joint_actions_rad = np.array([self.__deg_to_rad(action) for action in joint_actions_deg])

        # Generate all possible action combinations for the 6 joints
        action_combinations = list(itertools.product(joint_actions_rad, repeat=6))
        total_amount_actions = len(action_combinations)

        # Create a dictionary to map each combination to a unique integer
        self.actions_dict = {i: action for i, action in enumerate(action_combinations)}
        self.inv_actions_dict = {v: k for k, v in self.actions_dict.items()}

        # Create Q
        self.Q = -np.random.rand(amount_voxels, total_amount_actions)

        # Set ending positions to zero in Q
        for winning_voxel in self.winning_voxels:
            self.Q[self.voxels_index_dict[winning_voxel]] = np.zeros(total_amount_actions)

        # Create variable for voxel of current TCP position, so it only needs to be calculated
        # when the TCP is changed
        self.current_voxel = self.__get_tcp_voxel_position()
        # Save last voxel to be able to set last Q
        self.last_voxel = None

    def __deg_to_rad(self, deg: float) -> float:
        """Convert degree to radiants.

        :param deg: Degrees to convert to radiants
        :type deg: float

        :return: Radiants
        :rtype: float
        """
        return deg*np.pi/180

    def __rad_to_deg(self, rad: float) -> float:
        """Convert degree to radiants.

        :param rad: Ratiants to convert to radiants
        :type rad: float

        :return: Degrees
        :rtype: float
        """
        return rad*180/np.pi

    def __limit_angle(self, angle: float) -> float:
        """Limit angle (in rad) to +-pi (+-180°).

        :param rad: Angle in radiants
        :type rad: float

        :return: Limited angle in radiants
        :rtype: float
        """
        if angle > np.pi:
            return np.pi
        if angle < -np.pi:
            return -np.pi
        return angle

    def __limit_angles(self, angles: (float, float, float, float, float, float)
                       ) -> (float, float, float, float, float, float):
        """Limit angles of the robot (in rad) to +-pi (+-180°).

        :param rad: Angles in radiants
        :type rad: float

        :return: Limited angles in radiants
        :rtype: float
        """
        return np.array([self.__limit_angle(angle) for angle in angles])

    def __get_tcp_voxel_position(self) -> (int, int, int):
        """Get voxel the TCP is in.

        :return: Tuple containing the x, y and z coordinates of the voxel
        :rtype: (int, int, int)
        """
        # Compute forward kinematics to receive current TCP
        tcp = self.rob.fkine()
        x = tcp[0, 3]
        y = tcp[1, 3]
        z = tcp[2, 3]
        return (int(round(x, 0)), int(round(y, 0)), int(round(z, 0)))

    def __check_win(self) -> bool:
        """Check if the current position is in a winning voxel.

        :return: Bool indicating if the TCP is in a winning voxel
        :rtype: bool
        """
        return self.current_voxel in self.winning_voxels

    def __check_in_voxels(self) -> bool:
        """Check if the current position is in a voxel.

        :return: Bool indicating if the TCP is in a voxel
        :rtype: bool
        """
        return self.current_voxel in self.voxels_index_dict

    def __get_reward(self) -> int:
        """Get the reward for the current position.

        :return: Reward, -1 when at the start, getting 0 linearly when getting closer to the target
        :rtype: int
        """
        # Get index of voxel and return reward
        try:
            current_voxel_index = self.voxels_index_dict[self.current_voxel]
            return self.rewards[current_voxel_index]
        except IndexError as e:
            print(f"Index Error in Six_Axis_Robot_Arm.get_reward(): {e}")
            return None
        except ValueError as e:
            print(f"Value Error in Six_Axis_Robot_Arm.get_reward(): {e}")
            return None

    def get_joint_angles(self) -> (float, float, float, float, float, float):
        """Return current joint angles.

        :return: Tuple with the current angles of the robot joints in degrees
        :rtype: (float, float, float, float, float, float)

        :return: None
        """
        return np.array([self.__rad_to_deg(angle) for angle in self.rob.get_current_joint_config()])

    def set_joint_angles_degrees(self, angles: (float, float, float, float, float, float)) -> None:
        """Set joint angles.

        :param angles: Tuple with the angles for the robot joints in degrees
        :type angles: (float, float, float, float, float, float)

        :return: None
        """
        # Convert degrees of angles to radiants
        angles_rad = np.array([self.__deg_to_rad(angle) for angle in angles])
        self.set_joint_angles_rad(angles_rad)

    def set_joint_angles_rad(self, angles: (float, float, float, float, float, float)) -> None:
        """Set joint angles.

        :param angles: Tuple with the angles for the robot joints in radiants
        :type angles: (float, float, float, float, float, float)

        :return: None
        """
        # Limit angles to +-180°
        angles_rad = self.__limit_angles(angles)
        self.rob.update_angles(angles_rad, save=True)
        self.last_voxel = self.current_voxel
        self.current_voxel = self.__get_tcp_voxel_position()

    def reset(self) -> None:
        """Reset robot to starting state.

        :return: None
        """
        self.rob.reset(save=False)
        self.out_of_bounds_counter = 0
        self.current_voxel = self.__get_tcp_voxel_position()

    def get_random_action(self) -> ((float, float, float, float, float, float), int):
        """Get a random action from all actions.

        :return: Tuple containing the action and the unique integer representing the action in the
                 actions dict
        :rtype: ((float, float, float, float, float, float), int)
        """
        x = np.random.randint(len(self.actions_dict))
        return self.actions_dict[x], x

    def get_current_qs(self) -> list[float]:
        """Get the q values for the current state.

        :return: List of q values
        :rtype: list of float
        """
        # Return the value of Q at the index of the current voxel in the index dict
        return self.Q[self.voxels_index_dict[self.current_voxel]]

    def get_current_q(self, action: int) -> float:
        """Get the q value for the current state and a specific action.

        :param action: Action index for the action dict
        :type action: int

        :return: Q value
        :rtype: float
        """
        # Return the value of Q at the index of the current voxel in the index dict
        return self.Q[self.voxels_index_dict[self.current_voxel]][action]

    def set_current_q(self, action: int, q: float) -> None:
        """Set a q value for the current state.

        :param action: Action index for the action dict
        :type action: int

        :param new_q: new Q value to set
        :type new_q: float

        :return: None
        """
        # Set the value of Q at the index of the current voxel in the index dict
        self.Q[self.voxels_index_dict[self.current_voxel]][action] = q

    def set_last_q(self, action: int, q: float) -> None:
        """Set a q value for the state before the current state.

        :param action: Action index for the action dict
        :type action: int

        :param new_q: new Q value to set
        :type new_q: float

        :return: None
        """
        # Set the value of Q at the index of the current voxel in the index dict
        self.Q[self.voxels_index_dict[self.last_voxel]][action] = q

    def get_action_dict(self) -> dict:
        """Get the dict containing all actions (action_number : action).

        :return: Dict with all actions
        :rtype: dict
        """
        return self.actions_dict

    def get_action_from_dict(self, action: int) -> (float, float, float, float, float):
        """Get action from the actions dict.

        :param action: Action index for the action dict
        :type action: int

        :return: Tuple with action
        :rtype: (float, float, float, float, float)
        """
        return self.actions_dict[action]

    def get_inverse_action_dict(self) -> dict:
        """Get the inverse dict containing all actions (action : action_number).

        :return: Dict with all actions
        :rtype: dict
        """
        return self.inv_actions_dict

    def do_move(self, action: int) -> ((int, int, int), int, bool):
        """Move the robot based on the action.

        :param action: Action index for the action dict
                       Action dict can be obtained with get_action_dict()
                       Inverse action dict can be obtained with get_inverse_action_dict()
        :type action: int

        :return: tuple containing the new TCP (x, y, z), reward and win indication
        :rtype: ((int, int, int), int, bool)
        """
        win = False
        # calculate the new joint angles
        new_angles = self.rob.get_current_joint_config() + self.actions_dict[action]
        # Move robot into new position
        self.set_joint_angles_rad(new_angles)

        # Check for boundaries and reset if neccessary
        if self.__check_in_voxels() is False:
            # Go back to starting position
            self.set_joint_angles_rad(self.starting_angles)
            self.last_voxel = self.current_voxel
        # Check for win
        if self.__check_win() is True:
            win = True
        reward = self.__get_reward()

        # Forward kinematics for TCP coordinate calculation
        tcp_matrix = self.rob.fkine()
        # TCP Coordinates as (x, y, z)
        tcp_coordinates = (tcp_matrix[0, 3], tcp_matrix[1, 3], tcp_matrix[2, 3])
        return tcp_coordinates, reward, win

    def show(self, draw_path=False, draw_voxels=False, zoom_path=False) -> None:
        """Open window and draw robot arm.

        :param draw_path: Draw the path the robot is supposed to learn
        :type draw_path: bool

        :param draw_voxels: Draw the voxels
        :type draw_voxels: bool

        :param zoom_path: Fit drawing to the path
        :type zoom_path: bool

        :return: None
        """
        if zoom_path is False:
            ax = self.env.plot(show=False)
        else:
            ax = self.env.plot(show=False,
                               xlim=[np.min(self.path[0])-10, np.max(self.path[0])+10],
                               ylim=[np.min(self.path[1])-10, np.max(self.path[1])+10],
                               zlim=[np.min(self.path[2])-10, np.max(self.path[2])+10],
                               )

        if draw_path is True:
            ax.plot(self.path[0], self.path[1], self.path[2], color='red')

        if draw_voxels is True:
            x = []
            y = []
            z = []
            for voxel in self.voxels:
                x.append(voxel[0])
                y.append(voxel[1])
                z.append(voxel[2])
            ax.scatter(x, y, z, marker=".", s=2, color='cyan')

            x = []
            y = []
            z = []
            for voxel in self.winning_voxels:
                x.append(voxel[0])
                y.append(voxel[1])
                z.append(voxel[2])
            ax.scatter(x, y, z, marker=".", s=2.5, color='red')

        plt.show()

    def animate(self) -> None:
        """Animate robot.

        Needs to be called last.

        :return: None
        """
        self.env.animate()
