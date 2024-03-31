# Quaternion Dynamical System Learning

Quaternion-DS



---

### Double cover of quaternion
- It turns out that though -q and q represent the same rotation, they yield greatly different results when dealing with riemannian logarithm.
For example, when projecting a decaying sequence of angular velocity (in quaternion) wrt identity, one should expect the projected angular velocity
converge to zero; i.e. the zero rotation is equivalent to the identity quaternion. However, if the point of tangency is negative identity, which is
still identity and representing zero rotation, the projected angular velocity is now DIVERGING! 
- Solution could be: 
    1. force cannonical representation of quaternions; need to ensure smoothness of trajectory in both orientation and angular velocity on real data
    2. double-cover invariant in riemannian logarithm





### Update
11/10
- plot comparison between original and reproduced


11/9 
- deal with double cover of quaternion
- four tests in test folder


11/7
- archived all the outdated function and files
- centralize optimize into one single function
- vectorize parallel transport in `quat_tools.py`



9/8
- extend to pose
- test two framework: one fuse pos and ori, another one separately but ori dependent on pos in gamma (hierarchical model)
- explore the correlation in 7 by 7 covariance matrix


9/7
- generate angular velocity given only orientation(world and body frame)
- test all four tasks from pos_ori dataset



9/2
- ~~scale the result angular velcotiy~~
- ~~extend to multiple DS learning~~

9/1
- ~~propose a feasible dimension reduction to tackle singular covariance matrix resulting from unit vectors~~
- ~~construct double DS learning~~
- start on quaternion clustering


8/31
- ~~verify if system stable in tangent space~~
- ~~verify the equivalence between tan and quat~~


8/29 
- ~~finshi plot_tools.py~~
- ~~optimize_single_system~~
- ~~retrieve mean and covariance~~
- ~~construct quat normal dist~~
- ~~plot sequence of rotation and show clusters in time series~~

---
thoughts
- angular velocity directly acted on current orientation(world frame vs. local frame)
- is cannocal quaternion always the right option? maybe should choose the one closest to the att out of two