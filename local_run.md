# Running Multi-Robot Autonomous Exploration Locally

Follow these instructions to build and run the simulation using your `distrobox` virtual machine without needing any symbolic links or new workspaces.

## Step 1: Enter your Virtual Machine
Run this in your host terminal to enter the ROS 2 environment:
```bash
distrobox enter ros-humble
```
*(You must do this for every new terminal you open!)*

---

## Step 2: Install System Dependencies
*(If you haven't already run these)*
```bash
sudo apt update
sudo apt install ros-humble-turtlebot3-gazebo ros-humble-turtlebot3-description -y
sudo apt install ros-humble-nav2-bringup ros-humble-navigation2 -y
sudo apt install ros-humble-slam-toolbox -y
sudo apt install python3-pip -y
pip install numpy opencv-python
```

---

## Step 3: Setup Environment Variables
Append the required Gazebo and TurtleBot variables to your `~/.bashrc`:
```bash
echo 'export TURTLEBOT3_MODEL=burger' >> ~/.bashrc
echo 'export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/opt/ros/humble/share/turtlebot3_gazebo/models' >> ~/.bashrc
source ~/.bashrc
```

---

## Step 4: Build the Package
You can run `colcon build` right where your code already is.

```bash
# 1. Navigate to your project's root folder
cd /home/shiv/Autonomous-Swarm-Exploration

# 2. Source the base ROS 2 installation
source /opt/ros/humble/setup.bash

# 3. Build the package
colcon build

# 4. Source your local build
source install/setup.bash
```

---

## Step 5: Run the Simulation (Requires Multiple Terminals)
You will need to open **5 separate terminal windows/tabs**. 

For **EVERY new terminal**, you must enter the distrobox, navigate to your folder, and source the setups:
```bash
distrobox enter ros-humble
cd /home/shiv/Autonomous-Swarm-Exploration
source /opt/ros/humble/setup.bash
source install/setup.bash
export TURTLEBOT3_MODEL=burger
```

Once that is run in the terminal, launch the following commands one by one, waiting for each to fully load before starting the next:

**Terminal 1 (Simulation Environment & Robots):**
```bash
ros2 launch multi_robot_exploration spawn_two_turtlebots.launch.py
```

**Terminal 2 (Multi-Robot SLAM):**
```bash
ros2 launch multi_robot_exploration multi_robot_slam.launch.py
```

**Terminal 3 (Map Merging):**
```bash
ros2 launch multi_robot_exploration map_merge.launch.py
```

**Terminal 4 (Nav2 Navigation Stacks):**
```bash
ros2 launch multi_robot_exploration nav2_bringup_multi.launch.py
```

**Terminal 5 (Frontier Exploration Brain):**
*(Wait until Terminal 4 nodes are fully active)*
```bash
ros2 launch multi_robot_exploration frontier_exploration.launch.py
```

---

## Step 6 (Optional): Visualization with RViz2
To visualize the map merging and robots, open a **6th terminal**, run the setup commands (distrobox enter, cd, source), and then run:
```bash
rviz2 -d $(ros2 pkg prefix multi_robot_exploration)/share/multi_robot_exploration/config/multi_robot_exploration_cinematic.rviz
```
