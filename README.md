# SE(3) Linear Parameter Varying Dynamical Systems for Globally Asymptotically Stable End-Effector Control

Implementation of SE(3) Linear Parameter Varying Dynamical Systems (LPVDS) for pose control. The ```main``` branch contains the coupled learning of pose trajectory, whereas the decoupled variant of this framework is maintained in the ```main-decoupled``` branch, and please refer to its README.md for usage.


## Dataset
The available dataset contains the following:
1. clfd_data: collection of pose trajectory from the [clfd framework](https://github.com/sayantanauddy/clfd)
2. kinesthetic_demo: pose trajectory collected from kinesthetic teaching
3. UMI_demo: single pose trajectory collected from the UMI gripper real-time data collection


## Usage
Create a virtual envrionemnt and install the required packages:
```
virtualenv -p /path/to/python3.8 venv
source venv/bin/activate
pip install -r requirements.txt
```

Begin the learning process and evaluate the result:
```
python main.py
```

