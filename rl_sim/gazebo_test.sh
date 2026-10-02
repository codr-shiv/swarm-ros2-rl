#!/bin/bash
# Headless test: run a frontier-selection policy on the full Gazebo stack.
#
#   rl_sim/gazebo_test.sh <ppo|heuristic|nearest> [sim_seconds=240] [model.zip] [world=42]
#
# Starts Gazebo (no window) + SLAM + map merge + Nav2 on the benchmark arena,
# runs rl_sim/ros_policy_node.py, logs known area over time to
# results/gazebo_runs/<policy>_<time>.csv (launch logs in ~/rl_sim_runs/gazebo_tests/),
# then stops everything.
# Run inside the ros-humble distrobox after `colcon build` (from any directory).

POLICY=${1:?usage: gazebo_test.sh <ppo|heuristic|nearest> [sim_seconds] [model.zip] [world]}
DURATION=${2:-240}
MODEL=${3:-}
WORLD=${4:-42}
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT="$REPO/results/gazebo_runs"
mkdir -p "$OUT"
TAG="${POLICY}_$(date +%Y%m%d_%H%M%S)"
LOGS=~/rl_sim_runs/gazebo_tests/$TAG
mkdir -p "$LOGS"

unset DISPLAY WAYLAND_DISPLAY
source /opt/ros/humble/setup.bash
source "$REPO/install/setup.bash"
export TURTLEBOT3_MODEL=burger GAZEBO_WORLD_SEED=$WORLD
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID_TEST:-77}            # isolated from anything else running
export GAZEBO_MASTER_URI=http://127.0.0.1:11677
export GAZEBO_MODEL_PATH=${GAZEBO_MODEL_PATH:-}:/opt/ros/humble/share/turtlebot3_gazebo/models
export RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}

PGIDS=()
start() {  # start a launch in its own process group
    setsid ros2 launch multi_robot_exploration "$@" > "$LOGS/$1.log" 2>&1 &
    PGIDS+=($!)
}
cleanup() {
    for g in "${PGIDS[@]}"; do kill -INT -"$g" 2>/dev/null; done
    sleep 15
    for g in "${PGIDS[@]}"; do kill -KILL -"$g" 2>/dev/null; done
}
trap cleanup EXIT

echo "[$TAG] starting headless Gazebo ..."
start spawn_two_turtlebots.launch.py gui:=false; sleep 25
start multi_robot_slam.launch.py;                  sleep 10
start map_merge.launch.py;                         sleep 8
start nav2_bringup_multi.launch.py;                sleep 35

ARGS=(--policy "$POLICY" --world "$WORLD" --log "$OUT/$TAG.csv" --max-time "$DURATION")
[ -n "$MODEL" ] && ARGS+=(--model "$MODEL")
echo "[$TAG] running policy for $DURATION sim seconds ..."
python3 "$REPO/rl_sim/ros_policy_node.py" "${ARGS[@]}" 2>&1 | tee "$LOGS/policy.log" \
    | grep --line-buffered -E "Exploration started|EXPLORATION COMPLETE|Traceback|Error"
echo "[$TAG] area log: $OUT/$TAG.csv"
tail -1 "$OUT/$TAG.csv"
