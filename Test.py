#!/usr/bin/env python3
import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import *
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose, Twist
from std_msgs.msg import String
import tensorflow as tf
from tensorflow.keras import layers
from sensor_msgs.msg import LaserScan
import subprocess
import math
import tf_conversions
rospy.init_node('robot_control')
goal_x= 2.2
goal_y= 0
last_lidar_msg = None
# Define the LiDAR callback function
def lidar_callback(msg):
    global last_lidar_msg
    # Access the range values from the LIDAR scan
    ranges = msg.ranges
    
    # Check for any range value that is below a threshold indicating a collision
    collision_threshold = 0.2  # Adjust this threshold based on your environment
    collision_detected = any(distance < collision_threshold for distance in ranges)
    last_lidar_msg=msg
    if collision_detected:
        print("Collision detected!")
        return True
    else:
        return False

# Create a subscriber for the LIDAR scan
rospy.Subscriber('/scan', LaserScan, lidar_callback)

def odometry_callback(msg):
    global robot_x, robot_y, robot_yaw

    # Access the robot's position estimate
    robot_x = msg.pose.pose.position.x
    robot_y = msg.pose.pose.position.y
    quaternion = (
        msg.pose.pose.orientation.x,
        msg.pose.pose.orientation.y,
        msg.pose.pose.orientation.z,
        msg.pose.pose.orientation.w,
    )
    euler = tf_conversions.transformations.euler_from_quaternion(quaternion)
    robot_yaw = euler[2]  # Get the yaw component from the Euler angles

    # Print the robot's position estimate
    # print("Robot Position (x, y, yaw):", robot_x, robot_y, robot_yaw)

    # Publish the initial pose estimate
    initial_pose = PoseWithCovarianceStamped()
    initial_pose.header.stamp = rospy.Time.now()
    initial_pose.header.frame_id = "map"
    initial_pose.pose.pose.position.x = robot_x
    initial_pose.pose.pose.position.y = robot_y
    initial_pose.pose.pose.orientation = msg.pose.pose.orientation

    pose_pub.publish(initial_pose)

rospy.Subscriber('/odom', Odometry, odometry_callback)
pose_pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=1)


def send_goal(position_x, position_y, timeout):
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position.x = position_x
    goal.target_pose.pose.position.y = position_y
    goal.target_pose.pose.orientation.w = 1.0
    #print("Sending Goal Location: ({}, {})".format(position_x, position_y))
    ac.send_goal(goal)
    # Wait for the goal result with the specified timeout
    finished = ac.wait_for_result(rospy.Duration.from_sec(timeout))
    
ac = actionlib.SimpleActionClient('move_base', MoveBaseAction)
ac.wait_for_server()

# Define the check_crash_condition function
def check_crash_condition():
    if last_lidar_msg is not None:
        # Call the lidar_callback function with the last received LIDAR message
        collision_detected = lidar_callback(last_lidar_msg)
        if collision_detected:
            # Add your crash handling code here
            print("Turtlebot crashed!")
            return True

    return False
# Define the RobotInterface class
class RobotInterface:
    def __init__(self):
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

    def publish_twist(self, twist_msg):
        self.cmd_vel_pub.publish(twist_msg)

# Initialize CvBridge
bridge = CvBridge()
CAMERA_SCALING_FACTOR = 2.9 / 2.48

# Variables to store the robot position, velocity, and depth
robot_position = None
robot_velocity = None
estimated_distance = None
robot_orentation= None

# Create a publisher for the robot's velocity commands
cmd_vel_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

# Create a publisher for the /gazebo topic
gazebo_publisher = rospy.Publisher('/gazebo', String, queue_size=10)

# Callback function for the depth image
def depth_image_callback(msg):
    try:
        # Convert the ROS depth image message to OpenCV image
        depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')

        # Get the dimensions of the depth image
        height, width = depth_image.shape

        # Get the depth value at the center of the image
        center_depth = depth_image[int(height/2), int(width/2)]

        # Convert the depth value to distance (based on camera intrinsic parameters)
        distance = center_depth * CAMERA_SCALING_FACTOR

        # Calculate the median depth value
        median_depth = np.nanmedian(depth_image)

        # Convert the median depth value to distance (based on camera intrinsic parameters)
        median_distance = median_depth * CAMERA_SCALING_FACTOR

        # Update the estimated distance
        global estimated_distance
        estimated_distance = (median_distance + distance) / 2
        # Create a message to publish on the /gazebo topic
        message = "Hello, Gazebo!"

        # Publish the message on the /gazebo topic
        gazebo_publisher.publish(message)

    except Exception as e:
        rospy.logerr(e)

def robot_position_callback(data):
    global robot_position,robot_orientation
    # Get the index of the turtlebot3 model in the model states list
    model_index = data.name.index("turtlebot3_waffle")
    
    # Get the position of the turtlebot3
    robot_position = data.pose[model_index].position
    robot_orientation = data.pose[model_index].orientation

def robot_velocity_callback(data):
    global robot_velocity
    # Get the index of the turtlebot3 model in the model states list
    model_index = data.name.index("turtlebot3_waffle")

    # Get the velocity of the turtlebot3
    robot_velocity = data.twist[model_index]

# Create subscribers
rospy.Subscriber('/camera/depth/image_raw', Image, depth_image_callback)
rospy.Subscriber('/gazebo/model_states', ModelStates, robot_position_callback)
rospy.Subscriber('/gazebo/model_states', ModelStates, robot_velocity_callback)

# Define the DQNAgent class
class DQNAgent:
    def __init__(self, state_shape, num_actions):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.memory = []
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.models.load_model('/home/sri2/AIE20006/src/lab7_wanderer/src/modified/30')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.num_actions)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = np.random.choice(len(self.memory), batch_size, replace=False)
        for idx in minibatch:
            state, action, reward, next_state, done = self.memory[idx]
            target = reward
            if not done:
                target = reward + np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def perform_action(self, action):
        # Execute the selected action using your robot's interface
        if robot_velocity.linear.x>0:
            print('ahead')
            action=0
        elif robot_velocity.linear.x<0:
            print('behind')
            action=1
        elif robot_velocity.angular.z>0:
            print('turn')
            action=2
        elif robot_velocity.angular.z<0:
            print('turn')
            action=3
        elif robot_velocity.linear.x==0 and robot_velocity.angular.z==0:
            print('stop')
            action=4
        send_goal(goal_x, goal_y, timeout=2.0)
        return action

        
        
# Define the RobotNavigationEnv class
class RobotNavigationEnv:
    def __init__(self):
        # Initialize the environment parameters
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.x = -2.5
        self.y = 0
        self.orientation = 0.0
        self.robot_depth = 0.2
        self.robot_linear_x = 0.0
        self.robot_angular_z = 0.0

    def reset(self):
        # Create the command string to set the model state
        model_name = "turtlebot3_waffle"
        command = 'rosservice call /gazebo/set_model_state \'{model_state: { model_name: "' + model_name + '", pose: { position: { x: -2.5, y: 0, z: 0 }, orientation: { x: 0, y: 0, z: 0, w: 1 } }, twist: { linear: { x: 0, y: 0, z: 0 }, angular: { x: 0, y: 0, z: 0 } }, reference_frame: "world" }}\''
        # Execute the command using subprocess
        subprocess.call(command, shell=True)
        # Reset the environment to the initial state
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.x = -2.5
        self.y = 0
        self.orientation = 0.0
        self.robot_depth = 0.2
        self.robot_linear_x = 0.0
        self.robot_angular_z = 0.0
        return self.get_state()

    def step(self, action):
        # Perform the given action in the environment
        action=agent.perform_action(action)
        # Update the state based on the action
        self.update_state()
        # Calculate the reward based on the new state
        reward = self.get_reward()
        # Check if the termination condition is met
        done = self.check_termination()
        return self.get_state(), reward, done,action

    def get_state(self):
        # Return the current state of the environment
        state = [
            self.goal_x,
            self.goal_y,
            self.x,
            self.y,
            self.orientation,
            self.robot_depth,
            self.robot_linear_x,
            self.robot_angular_z
        ]
        return np.array(state).reshape(1, -1)



    def get_reward(self):
        distance_x = abs(self.x - self.goal_x)
        distance_y = abs(self.y - self.goal_y)
        
        # Get the orientation quaternion from Gazebo
        gazebo_orientation = {'x': 0, 'y': 0, 'z': self.orientation, 'w': 1}
        
        quaternion = [gazebo_orientation['x'], gazebo_orientation['y'], gazebo_orientation['z'], gazebo_orientation['w']]
        euler = tf_conversions.transformations.euler_from_quaternion(quaternion)
        # Extract the yaw angle from the Euler angles
        yaw = euler[2]
        
        goal_angle = math.atan2(self.goal_y - self.y, self.goal_x - self.x)
        
        # Calculate the absolute difference between the goal angle and the robot's orientation
        theta_diff = abs(goal_angle - yaw)
        
        # Check if the robot has reached the goal
        if distance_x < 0.1 and distance_y < 0.1:
            reward = 1000  # Reached the goal
        # Check if the robot has collided
        elif check_crash_condition():
            reward = -1000  # Collided
        elif self.robot_angular_z==0.0 and self.robot_linear_x==0 and self.robot_depth>0.5 or self.robot_linear_x==0:
            reward = -10 
        else:
            # Calculate the reward based on the distance, orientation, velocity, and depth
            reward = 1 / ((distance_x**2 + distance_y**2 + 1) + (theta_diff + 1))
        
        if self.robot_depth<0.3:
            reward=reward-5
        else:
            reward=reward+5
        
        return reward

    def check_termination(self):
        # Check if the termination condition is met
        if not check_crash_condition():
            distance_x = abs(self.x - self.goal_x)
            distance_y = abs(self.y - self.goal_y)
            return distance_x < 0.1 and distance_y < 0.1
        else:
            return True

    def update_state(self):
        self.x = robot_position.x
        self.y = robot_position.y
        self.orientation = robot_orientation.z
        self.robot_depth = estimated_distance
        self.robot_linear_x = robot_velocity.linear.x
        self.robot_angular_z = robot_velocity.angular.z


state_shape = (8,)  # Shape of the state vector
num_actions = 5  # Number of possible actions

# Create the DQNAgent and RobotNavigationEnv instances
agent = DQNAgent(state_shape, num_actions)
env = RobotNavigationEnv()
arr=[]
# Training loop
num_epochs = 100
robot_interface = RobotInterface()
batch_size = 32
timeout_duration = rospy.Duration(30)  # 30 seconds timeout

# After training, use the learned policy to navigate the environment
state = env.reset()
done = False
start_time = rospy.Time.now()

while not done:
    # Check if the timeout duration has been exceeded
    if rospy.Time.now() - start_time > timeout_duration:
        twist_msg = Twist()
        twist_msg.linear.x = 0.0
        twist_msg.angular.z = 0.0
        robot_interface.publish_twist(twist_msg)
        ac.cancel_goal()
        print("Timeout reached. Navigation terminated.")
        done = True
         
        break

    action = agent.act(state)
    next_state, reward, done,action = env.step(action)
    state = next_state
