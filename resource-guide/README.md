# Resource Guide

A curated path through the **foundations** behind this project: multi-robot exploration with ROS 2, SLAM, navigation and reinforcement learning. Work through the sections in order before diving into the code in [`../final-product/`](../final-product/). Each entry says what it teaches and why it matters here.

---

## 1. Programming and tools
| Resource | What you learn |
|---|---|
| [The Python Tutorial](https://docs.python.org/3/tutorial/) | Language basics used everywhere in this repo |
| [NumPy: absolute beginners guide](https://numpy.org/doc/stable/user/absolute_beginners.html) | Array operations; occupancy grids are NumPy arrays |
| [OpenCV-Python tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html) | Morphology (dilate, erode) and connected components: the core of frontier detection |
| [Pro Git book](https://git-scm.com/book/en/v2) | Version control for collaborating on this repository |

## 2. ROS 2 (Humble)
| Resource | What you learn |
|---|---|
| [ROS 2 Humble documentation, Tutorials](https://docs.ros.org/en/humble/Tutorials.html) | Nodes, topics, services, actions, launch files, parameters, colcon. **Start here.** |
| [ROS 2 Concepts](https://docs.ros.org/en/humble/Concepts.html) | Middleware (DDS), QoS, namespaces: why `/robot1/...` and `/robot2/...` don't collide |
| [tf2 tutorials](https://docs.ros.org/en/humble/Tutorials/Intermediate/Tf2/Tf2-Main.html) | Coordinate frames and transforms (`map → odom → base_footprint`) |
| [ROBOTIS TurtleBot3 e-Manual](https://emanual.robotis.com/docs/en/platform/turtlebot3/overview/) | The robot, its lidar, and its Gazebo simulation packages |
| [Gazebo Classic tutorials](https://classic.gazebosim.org/tutorials) | Worlds, models, SDF and sensors (the arena is generated as SDF) |

## 3. Probabilistic robotics and SLAM
| Resource | What you learn |
|---|---|
| S. Thrun, W. Burgard, D. Fox, *Probabilistic Robotics*, MIT Press, 2005 | The theory of occupancy-grid mapping, localisation and SLAM. Chapters 9–10 are the essentials |
| Cyrill Stachniss, *Mobile Sensing and Robotics* / SLAM lecture series (YouTube, University of Bonn) | Video lectures covering the same theory, approachable |
| [slam_toolbox (GitHub)](https://github.com/SteveMacenski/slam_toolbox) | The SLAM package used here; its README explains async mapping and parameters |

## 4. Navigation (Nav2)
| Resource | What you learn |
|---|---|
| [Nav2 documentation](https://docs.nav2.org/) | Planners, controllers, costmaps, behaviour trees, lifecycle nodes |
| [Nav2 "Getting Started" and "Navigation Concepts"](https://docs.nav2.org/getting_started/index.html) | How a `NavigateToPose` goal becomes motion: the interface the exploration brain uses |
| S. Macenski et al., "The Marathon 2: A Navigation System", IROS 2020 ([arXiv:2003.00368](https://arxiv.org/abs/2003.00368)) | The design of Nav2 itself |

## 5. Frontier-based and multi-robot exploration
| Resource | What you learn |
|---|---|
| B. Yamauchi, "A frontier-based approach for autonomous exploration", IEEE CIRA 1997 | **The original frontier idea:** drive to the boundary between known and unknown space |
| W. Burgard, M. Moors, C. Stachniss, F. Schneider, "Coordinated multi-robot exploration", IEEE Transactions on Robotics 21(3), 2005 | Assigning frontiers to several robots with utility costs: the idea behind the heuristic coordinator |

## 6. Reinforcement learning
| Resource | What you learn |
|---|---|
| R. Sutton, A. Barto, *Reinforcement Learning: An Introduction*, 2nd ed. ([free online](http://incompleteideas.net/book/the-book-2nd.html)) | MDPs, returns, value functions, policy gradients: the foundation |
| [OpenAI Spinning Up](https://spinningup.openai.com/) | A practical introduction to deep RL and policy-gradient methods, with clear PPO notes |
| J. Schulman et al., "Proximal Policy Optimization Algorithms", 2017 ([arXiv:1707.06347](https://arxiv.org/abs/1707.06347)) | The PPO algorithm used for training |
| J. Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation", 2015 ([arXiv:1506.02438](https://arxiv.org/abs/1506.02438)) | GAE (λ = 0.95 in our training) |
| S. Huang, S. Ontañón, "A Closer Look at Invalid Action Masking in Policy Gradient Algorithms", 2020 ([arXiv:2006.14171](https://arxiv.org/abs/2006.14171)) | Why masking invalid frontier slots makes training work |
| [Gymnasium documentation](https://gymnasium.farama.org/) | The environment API (`reset`, `step`, spaces) that `rl_sim` implements |
| [Stable-Baselines3 documentation](https://stable-baselines3.readthedocs.io/) | Using PPO in practice; the "RL Tips and Tricks" page is essential reading |
| [sb3-contrib: MaskablePPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html) | The masked PPO implementation used here |

## 7. Learned exploration and sim-to-real
| Resource | What you learn |
|---|---|
| D. S. Chaplot et al., "Learning to Explore using Active Neural SLAM", ICLR 2020 ([arXiv:2004.05155](https://arxiv.org/abs/2004.05155)) | A learned global goal-selection policy for exploration, rewarded by coverage: the closest relative of this project |
| J. Tobin et al., "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World", IROS 2017 ([arXiv:1703.06907](https://arxiv.org/abs/1703.06907)) | How to make simulator-trained policies transfer, relevant to the next steps (training on many arenas, real robots) |

---

**Suggested order for a newcomer:**
1. Sections 1–2: get ROS 2 running and the TurtleBot3 simulation working.
2. Sections 3–4: understand how a map is built and how a goal becomes motion.
3. Section 5: understand what the exploration brain decides.
4. Sections 6–7: understand how the RL brain learns it.

Then read [`../final-product/README.md`](../final-product/README.md).
