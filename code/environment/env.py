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
        self.voxels, self.winning_voxels = path.get_helix_voxels()
        self.voxel_size = 1
        self.path = path.get_helix_data()

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
        q0 = self.__limit_angles(self.rob.ikine((self.path[0][0], self.path[1][0], self.path[2][0])))
        print(f"initial angles: {q0}")
        # Create arm
        self.rob = Arm(links, q0, '1-link')

        self.env = Environment(dimensions=[1500.0, 1500.0, 1500.0],
                               robot=self.rob)

        # Create all possible actions
        # Define possible actions for each joint in deg
        # For now 1 degree per action, as the robot will take forever otherwise
        joint_actions_deg = [-1, 0, 1]
        joint_actions_rad = np.array([self.__deg_to_rad(action) for action in joint_actions_deg])

        # Generate all possible action combinations for the 6 joints
        action_combinations = list(itertools.product(joint_actions_rad, repeat=6))

        # Create a dictionary to map each combination to a unique integer
        self.actions_dict = {i: action for i, action in enumerate(action_combinations)}
        self.inv_actions_dict = {v: k for k, v in self.actions_dict.items()}
        #print(self.actions_dict)

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
        voxel_pos = self.__get_tcp_voxel_position()
        return voxel_pos in self.winning_voxels

    def __check_in_voxels(self) -> bool:
        """Check if the current position is in a voxel.

        :return: Bool indicating if the TCP is in a voxel
        :rtype: bool
        """
        voxel_pos = self.__get_tcp_voxel_position()
        return voxel_pos in self.voxels

    def __get_reward(self) -> int:
        """Get the reward for the current position.

        :return: Reward, -1 when we are not in a winning voxel, otherwise 0
        :rtype: int
        """
        if self.__check_win() is True:
            return 0
        else:
            return -1

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
        # Limit angles to +-180°
        angles_rad = self.__limit_angles(angles_rad)
        self.rob.update_angles(angles_rad, save=True)

    def set_joint_angles_rad(self, angles: (float, float, float, float, float, float)) -> None:
        """Set joint angles.

        :param angles: Tuple with the angles for the robot joints in radiants
        :type angles: (float, float, float, float, float, float)

        :return: None
        """
        # Limit angles to +-180°
        angles_rad = self.__limit_angles(angles)
        self.rob.update_angles(angles_rad, save=True)

    def get_random_action(self) -> ((float, float, float, float, float, float), int):
        """Get a random action from all actions.

        :return: Tuple containing the action and the unique integer representing the action in the
                 actions dict
        :rtype: ((float, float, float, float, float, float), int)
        """
        x = np.random.randint(len(self.actions_dict))
        return self.actions_dict[x], x

    def get_action_dict(self) -> dict:
        """Get the dict containing all actions (action_number : action).

        :return: Dict with all actions
        :rtype: dict
        """
        return self.actions_dict

    def get_inverse_action_dict(self) -> dict:
        """Get the inverse dict containing all actions (action : action_number).

        :return: Dict with all actions
        :rtype: dict
        """
        return self.inv_actions_dict

    def do_move(self, action: int) -> ((int, int, int), int, bool):
        """Move the robot based on the action.

        :param action: Action from the action dict.
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
        #if self.__check_in_voxels() is False:
        #    # Go back to starting position
        #    print(f" not in voxels! ")
        #    new_angles = self.rob.ikine((self.path[0][0], self.path[1][0], self.path[2][0]))
        #    self.set_joint_angles_rad(new_angles)
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
        """Animate robot. Needs to be called last

        :return: None
        """
        self.env.animate()


#arm = Six_Axis_Robot_Arm()
#move = (3**6-1) / 2 + 2
#print(f"doing move: {arm.get_action_dict()[move]}")
#for i in range(100):
    #print(f"action: {arm.get_random_action()}")
    #action, action_number = arm.get_random_action()
#    arm.do_move(move)
#arm.show(draw_path=True, draw_voxels=True, zoom_path=True)
#print("\n\nDone\n\n")
#arm.animate()

# Optimize space:
# 1) All voxels with every decision at every voxel
# 2) All vocels only with evry decision at the needed voxels
# 3) Only define the coordinate of the voxels needed in an array, sort that array and use binary search for every decision in that array