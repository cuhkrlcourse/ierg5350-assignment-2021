# IERG 5350 Assignment 4: Reinforcement Learning Pipeline in Practice

**Due: 11:59 pm, November 26 (Friday Night), 2021**

**Changelog**:

```diff
+ Nov 22: Update installation guide. Update examplar evaluate script.
+ Nov 22: Add box2d installation guide.
+ Nov 15: Change the installation instruction on MetaDrive. Some windows user experience error if install via pip git+.
- Nov 8: Remove TD3 in CartPole-v0 requirement.
- Nov 6: Remove wrong assertions in BaseTrainer test. dist_entropy should be a tensor with single entry.
```

---


Welcome to assignment 4 of our RL course! In this assignment, we will implement a RL system that allows us to train and evaluate RL agents formally and efficiently. In this assigment, we will **train agent to drive car**!

Different from the previous assignment, this assignment requires you to finish codes mainly in python files instead of in a jupyter notebook. The provided jupyter notebook is only guideline and visualizing toolbox for you to test and play around those python files. 

This assignment is highly flexible. You are free to do hyper-parameter searching or improve your implementation by introducing new components.


### The grading scheme and deliverables

We reuiqred you to finish the following tasks:


* Task 1: Get familiar with the environment, play with MetaDrive **(0/200 points)**
* Task 2: Implement BaseTrainer **(30/200 points)**
* Task 3: Implement A2CTrainer **(30/200 points)**
* Task 4: Implement PPOTrainer **(40/200 points)**
* Task 5: Implement TD3Trainer **(40/200 points)**
* Task 6: Try to demonstrate the generalization curves of different methods **(60/200 points)**

As we said above, the jupyter notebook is only a visualization toolbox. The ultimate deliverables justifying your effort are a set of learning curves. You should use the provided `result.md` and fill the required curves in it before generating a **PDF file** which should be submitted to the blackboard. Please visit the `result.md` for details.

You need to **zip all files** in `assignment4` and upload it **with the pdf file**  and submit **both the zip file and the pdf file** to blackboard.



### File structure

1. `train.py` - Train scripts for A2C and PPO in CartPole-v0 and MetaDrive. Please implement `train`.
2. `core/base_trainer.py` - Please implement `compute_action`.
3. `core/a2c_trainer.py` - A2C algorithm. Please implement `compute_loss` and `update`.
4. `core/ppo_trainer.py` - PPO algorithm. Please implement `compute_loss`.
5. `core/buffer.py` - Supporting data structure for both A2C and PPO (GAE is implemented here). Please implement `compute_returns` for both A2C and PPO.
6. `core/td3_trainer.py` - File which can be directly run to train TD3. Please implement TD3 here.
7. `assignment4.ipynb` - Useful jupyter notebook to walk you through the whole assignment. Unlike previous work, you are not required to fill anything in this notebook (but you can use it to generate necessary images if you like those code). 
8. `result.md` - A **template** file for your final submission. You need to **generate a PDF file** based on it after filling all curves. 

We also provide many useful scripts like the `exp_search_lr.sh`. Since they are not necessary for your implementation so we ignore them here.



### Dependencies
Before start coding, please make sure to install the following packages:

* box2d
* pandas
* scipy
* seaborn
* tabulate

These can be installed via: `pip install pandas scipy seaborn tabulate`

Since we might required `BipedalWalker` environment during testing, you might also need to install box2D engine via `pip install gym[box2d]` or `pip install gym[all]`. 

We also need to install a lightweight driving simulator [MetaDrive](https://github.com/decisionforce/metadrive):
 
 
```bash
git clone https://github.com/decisionforce/metadrive.git
cd metadrive
pip install -e .

# For some windows user, please install metadrive via:
pip install --user -e .
```



### Notes

1. We use multi-processing to enable asynchronous data sampling. Therefore in many places the tensors have shape `[num_steps, num_envs, num_feat]`. This means that `num_envs` environments are running concurrently and each of them samples a fragment of trajectory with length `num_steps`. There are totally `num_steps*num_envs` data points when entering each iteration of training. PPO will split the large batch into many mini batches though.
2. Each process can only have a single MetaDrive environment. If you create a MetaDrive environment and then close it, you can't not launch a new MetaDrive env in this process anymore. We suggest you to restart the program or the notebook in case this happens.
3. The jupyter notebook is used for tutorial and visualization. It is optional for you to use the notebook to train agents or visualize results.
4. If you look for computing resources, you can try out the **Google Colab**, where you can apply for free GPU resources to accelerate the learning of your RL models. Here are some resources as intro to the Colab.
	- YouTube Video: https://www.youtube.com/watch?v=inN8seMm7UI
	- Colab Intro: https://colab.research.google.com/notebooks/intro.ipynb (you may need to login with your google account)
5. We learned that install MetaDrive in Windows via `pip install git+https://github.com/decisionforce/metadrive.git` is problematic. Please use the above way to install, thanks!








------

*2021-2022 1st term, IERG 5350: Reinforcement Learning. Department of Information Engineering, The Chinese University of Hong Kong. Course Instructor: Professor ZHOU Bolei. Assignment author: PENG Zhenghao.*

