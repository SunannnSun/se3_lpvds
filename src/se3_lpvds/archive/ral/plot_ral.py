
import matplotlib.pyplot as plt
import numpy as np
import random



colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
"#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]


def plot_reference_trajectories_DS2(Data, att, vel_sample, vel_size):
    fig = plt.figure(figsize=(6, 6))
    M = len(Data) / 2 

    ax = fig.add_subplot(111)

    L = Data.shape[0]

    # for l in range(L):
    #     data_l = Data[l, 0]
    #     ax.plot(data_l[0, :], data_l[1, :], 'k', alpha=0.65, linewidth=1)

    for l in range(L):

        if l == 2:
            cut = 350
        else:
            cut = 400


        data_l = Data[l, 0]

        ax.plot(data_l[0, :cut], data_l[1, :cut], 'slateblue' , alpha=0.55, linewidth=1)
        ax.plot(data_l[0, cut:], data_l[1, cut:],  'crimson' , alpha=0.55, linewidth=1)

    # ax.scatter(-25, 40, marker=(6, 2, 0), s=100, c='black')
    # ax.quiver(-25, 40, 1, -1, color='k', scale= 10)



    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    ax.set_xlabel(r'$x_1$', fontsize=34)
    ax.set_ylabel(r'$x_2$', fontsize=34)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig('damm.png', dpi=600, transparent=True)


    plt.show()



def plot_damm_vis(Data):
    sub_samp = 1



    M = int(Data.shape[1]/2)

    Data= Data[::sub_samp, :]

    

    plt.figure()
    ax = plt.axes(projection='3d')

    L = Data.shape[0]

    for l in range(L):

        if l == 2:
            cut = 350
        else:
            cut = 400


        data_l = Data[l, 0]

        ax.plot(data_l[0, :cut], data_l[1, :cut], np.zeros((cut, )), 'slateblue' , alpha=0.55, linewidth=1)
        ax.plot(data_l[0, cut:], data_l[1, cut:], np.zeros((1000-cut, )), 'crimson' , alpha=0.55, linewidth=1)



    data1 = [] 
    for l in range(L):
        if l == 2:
            cut = 350
        else:
            cut = 400

        data_l = Data[l, 0][0:2, :cut]
        if l == 0:
            data1 = data_l
        else:
            data1 = np.hstack((data1, data_l))




    data1_cov = np.cov(data1)
    data1_cov_new = 10 * np.eye(3)

    data1_cov_new[:2, :2] = data1_cov
    Mu  = np.mean(data1, axis= 1).reshape(-1, 1)
    Mu = np.vstack((Mu, np.array([0])))



    _, s, rotation = np.linalg.svd(data1_cov_new)  # find the rotation matrix and radii of the axes
    radii = np.sqrt(s) * 1.7                        # set the scale factor yourself
    u = np.linspace(0.0, 2.0 * np.pi, 60)
    v = np.linspace(0.0, np.pi, 60)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))   
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + Mu[:, 0]
    ax.plot_surface(x, y, z, rstride=3, cstride=3, color="slateblue", linewidth=0.1, alpha=0.3, shade=True) 




    data2 = [] 
    for l in range(L):
        if l == 2:
            cut = 350
        else:
            cut = 400

        data_l = Data[l, 0][0:2, cut:]
        if l == 0:
            data2 = data_l
        else:
            data2 = np.hstack((data2, data_l))




    data2_cov = np.cov(data2)
    data2_cov_new = 10 * np.eye(3)

    data2_cov_new[:2, :2] = data2_cov
    Mu  = np.mean(data2, axis= 1).reshape(-1, 1)
    Mu = np.vstack((Mu, np.array([0])))



    _, s, rotation = np.linalg.svd(data2_cov_new)  # find the rotation matrix and radii of the axes
    radii = np.sqrt(s) * 1.7                        # set the scale factor yourself
    u = np.linspace(0.0, 2.0 * np.pi, 60)
    v = np.linspace(0.0, np.pi, 60)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))   
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + Mu[:, 0]
    ax.plot_surface(x, y, z, rstride=3, cstride=3, color="crimson", linewidth=0.1, alpha=0.3, shade=True) 






    ax.set_xlabel(r'$\xi_1$', labelpad=-8)
    ax.set_ylabel(r'$\xi_2$', labelpad=-8)
    ax.set_zlabel(r'$\xi_3$')

    labels = [item.get_text() for item in ax.get_xticklabels()]

    empty_string_labels = ['']*len(labels)
    ax.set_xticklabels(empty_string_labels)


    labels = [item.get_text() for item in ax.get_yticklabels()]

    empty_string_labels = ['']*len(labels)
    ax.set_yticklabels(empty_string_labels)


    empty_string_labels = ['0', '', '']
    ax.set_zticklabels(empty_string_labels)


    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    plt.grid(b=None)


    ax.set_xlim(-40, 2)
    ax.set_ylim(5, 45)
    ax.set_zlim(-20, 20)
    ax.set_aspect("equal")

    ax.set_xticks(np.array([ -20, 10]))
    ax.set_yticks(np.array([ 25, 45 ]))
    ax.set_zticks(np.array( [0, 20 ]))

    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)    

    ax.view_init(elev=28., azim=-122)

    plt.savefig("damm_1.png", dpi=600, transparent=True)






def plot_damm_vis_blue(Data):
    sub_samp = 1



    M = int(Data.shape[1]/2)

    Data= Data[::sub_samp, :]

    

    plt.figure()
    ax = plt.axes(projection='3d')

    L = Data.shape[0]

    for l in range(L):

        if l == 2:
            cut = 350
        else:
            cut = 400


        data_l = Data[l, 0]

        ax.plot(data_l[0, :cut], data_l[1, :cut], np.zeros((cut, )), 'slateblue' , alpha=0.55, linewidth=1)
        ax.plot(data_l[0, cut:], data_l[1, cut:], np.zeros((1000-cut, )), 'black' , alpha=0.55, linewidth=1)




    data1 = [] 
    for l in range(L):
        if l == 2:
            cut = 350
        else:
            cut = 400

        data_l = Data[l, 0][0:2, :cut]
        if l == 0:
            data1 = data_l
        else:
            data1 = np.hstack((data1, data_l))




    data1_cov = np.cov(data1)
    data1_cov_new = 10 * np.eye(3)

    data1_cov_new[:2, :2] = data1_cov
    Mu  = np.mean(data1, axis= 1).reshape(-1, 1)
    Mu = np.vstack((Mu, np.array([0])))



    _, s, rotation = np.linalg.svd(data1_cov_new)  # find the rotation matrix and radii of the axes
    radii = np.sqrt(s) * 1.7                        # set the scale factor yourself
    u = np.linspace(0.0, 2.0 * np.pi, 60)
    v = np.linspace(0.0, np.pi, 60)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))   
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + Mu[:, 0]
    ax.plot_surface(x, y, z, rstride=3, cstride=3, color="slateblue", linewidth=0.1, alpha=0.3, shade=True) 




    ##############################################################################################################

    ax.scatter(-25, 40, 12, marker=(6, 2, 0), s=40, c='black')

    ax.scatter(-25, 40, 0, marker=(6, 2, 0), s=40, c='black', alpha= 0.5)

    ax.plot([-25, -25], [40, 40], [0, 12], '--',   c='black', alpha= 0.7)


    q = ax.quiver(-25, 40, 12, 1, -1, 0, length=10, normalize=True,colors='k', arrow_length_ratio=0.1)

    q = ax.quiver(-25, 40, 12, 1, 2, 0, length=10, normalize=True,colors='slateblue', arrow_length_ratio=0.1)


    ax.set_xlabel(r'$\xi_1$', labelpad=-8)
    ax.set_ylabel(r'$\xi_2$', labelpad=-8)
    ax.set_zlabel(r'$\xi_3$')



    labels = [item.get_text() for item in ax.get_xticklabels()]

    empty_string_labels = ['']*len(labels)
    ax.set_xticklabels(empty_string_labels)


    labels = [item.get_text() for item in ax.get_yticklabels()]

    empty_string_labels = ['']*len(labels)
    ax.set_yticklabels(empty_string_labels)


    empty_string_labels = ['0', '', '']
    ax.set_zticklabels(empty_string_labels)


    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    plt.grid(b=None)


    ax.set_xlim(-40, 2)
    ax.set_ylim(5, 45)
    ax.set_zlim(-20, 20)
    ax.set_aspect("equal")

    ax.set_xticks(np.array([ -20, 10]))
    ax.set_yticks(np.array([ 25, 45 ]))
    ax.set_zticks(np.array( [0, 20 ]))

    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)    

    ax.view_init(elev=28., azim=-122)

    plt.savefig("damm_1.png", dpi=600, transparent=True)













def plot_damm_vis_red(Data):
    sub_samp = 1



    M = int(Data.shape[1]/2)

    Data= Data[::sub_samp, :]

    

    plt.figure()
    ax = plt.axes(projection='3d')

    L = Data.shape[0]

    for l in range(L):

        if l == 2:
            cut = 350
        else:
            cut = 400


        data_l = Data[l, 0]

        ax.plot(data_l[0, :cut], data_l[1, :cut], np.zeros((cut, )), 'black' , alpha=0.55, linewidth=1)
        ax.plot(data_l[0, cut:], data_l[1, cut:], np.zeros((1000-cut, )), 'crimson' , alpha=0.55, linewidth=1)





    data2 = [] 
    for l in range(L):
        if l == 2:
            cut = 350
        else:
            cut = 400

        data_l = Data[l, 0][0:2, cut:]
        if l == 0:
            data2 = data_l
        else:
            data2 = np.hstack((data2, data_l))




    data2_cov = np.cov(data2)
    data2_cov_new = 10 * np.eye(3)

    data2_cov_new[:2, :2] = data2_cov
    Mu  = np.mean(data2, axis= 1).reshape(-1, 1)
    Mu = np.vstack((Mu, np.array([0])))



    _, s, rotation = np.linalg.svd(data2_cov_new)  # find the rotation matrix and radii of the axes
    radii = np.sqrt(s) * 1.7                        # set the scale factor yourself
    u = np.linspace(0.0, 2.0 * np.pi, 60)
    v = np.linspace(0.0, np.pi, 60)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))   
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + Mu[:, 0]
    ax.plot_surface(x, y, z, rstride=3, cstride=3, color="crimson", linewidth=0.1, alpha=0.3, shade=True) 



    ax.scatter(-25, 40, 7, marker=(6, 2, 0), s=40, c='black')

    ax.scatter(-25, 40, 0, marker=(6, 2, 0), s=40, c='black')

    ax.plot([-25, -25], [40, 40], [0, 7], '--',   c='black', alpha= 0.5)


    q = ax.quiver(-25, 40, 7, 1, -1, 0, length=10, normalize=True,colors='k', arrow_length_ratio=0.1)

    q = ax.quiver(-25, 40, 7, 1, -0.65, 0, length=10, normalize=True,colors='crimson', arrow_length_ratio=0.1)





    ax.set_xlabel(r'$\xi_1$', labelpad=-8)
    ax.set_ylabel(r'$\xi_2$', labelpad=-8)
    ax.set_zlabel(r'$\xi_3$')


    labels = [item.get_text() for item in ax.get_xticklabels()]

    empty_string_labels = ['']*len(labels)
    ax.set_xticklabels(empty_string_labels)


    labels = [item.get_text() for item in ax.get_yticklabels()]

    empty_string_labels = ['']*len(labels)
    ax.set_yticklabels(empty_string_labels)


    empty_string_labels = ['0', '', '']
    ax.set_zticklabels(empty_string_labels)


    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    plt.grid(b=None)


    ax.set_xlim(-40, 2)
    ax.set_ylim(5, 45)
    ax.set_zlim(-20, 20)
    ax.set_aspect("equal")

    ax.set_xticks(np.array([ -20, 10]))
    ax.set_yticks(np.array([ 25, 45 ]))
    ax.set_zticks(np.array( [0, 20 ]))

    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)    

    ax.view_init(elev=28., azim=-122)


    plt.savefig("damm_1.png", dpi=600, transparent=True)






def plot_PS_vis(Data):
    fig = plt.figure(figsize=(6, 6))
    M = len(Data) / 2 

    ax = fig.add_subplot(111)

    L = Data.shape[0]

    L = Data.shape[0]

    for l in range(L):

        # if l == 2:
        #     cut = 350
        # else:
        cut = 400
        cut2 = 580
        cut3 = 700
    
        data_l = Data[l, 0]

        ax.plot(data_l[0, :cut], data_l[1, :cut], 'crimson' ,  linewidth=1)
        ax.plot(data_l[0, cut:cut2], data_l[1, cut:cut2], 'slateblue' ,  linewidth=1)
        ax.plot(data_l[0, cut2:cut3], data_l[1, cut2:cut3], 'darkmagenta' ,  linewidth=1)
        ax.plot(data_l[0, cut3:], data_l[1, cut3:], 'black' ,  linewidth=1)



    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    ax.set_xlabel(r'$\xi_1$', fontsize=34)
    ax.set_ylabel(r'$\xi_2$', fontsize=34)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig('damm.png', dpi=600, transparent=True)


    plt.show()


def plot_PS_launch_init(Data):
    fig = plt.figure(figsize=(6, 6))
    M = len(Data) / 2 

    ax = fig.add_subplot(111)

    L = Data.shape[0]

    L = Data.shape[0]

    random_array = np.random.randint(2, size=400)

    color_mapping = np.take(colors, random_array)

    

    for l in range(L):
        cut = 400
        cut2 = 580
        cut3 = 700
    
        data_l = Data[l, 0]

        # ax.plot(data_l[0, :cut], data_l[1, :cut], 'crimson' , alpha=0.65, linewidth=1)

        ax.scatter(data_l[0, :cut], data_l[1, :cut], s=0.2,  c=color_mapping)

        
        ax.plot(data_l[0, cut:cut2], data_l[1, cut:cut2], 'slateblue'  , linewidth=1)
        ax.plot(data_l[0, cut2:cut3], data_l[1, cut2:cut3], 'darkmagenta' , linewidth=1)
        ax.plot(data_l[0, cut3:], data_l[1, cut3:], 'black' , linewidth=1)



    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    ax.set_xlabel(r'$\xi_1$', fontsize=34)
    ax.set_ylabel(r'$\xi_2$', fontsize=34)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig('damm.png', dpi=600, transparent=True)


    plt.show()






def plot_PS_launch(Data):
    fig = plt.figure(figsize=(6, 6))
    M = len(Data) / 2 

    ax = fig.add_subplot(111)

    L = Data.shape[0]

    L = Data.shape[0]

    random_array = np.random.randint(2, size=400)

    color_mapping = np.take(colors, random_array)

    

    for l in range(L):
        if l == 5 or l==4 or l==3 or l==2:
            cut = 240
        else:    
            cut = 270
        cut1 = 400
        cut2 = 580
        cut3 = 700
    
        data_l = Data[l, 0]

        # ax.plot(data_l[0, :cut], data_l[1, :cut], 'crimson' , alpha=0.65, linewidth=1)

        ax.scatter(data_l[0, :cut], data_l[1, :cut], s=0.2,  c="crimson")
        ax.scatter(data_l[0, cut:cut1], data_l[1, cut:cut1], s=0.2,  c="darkseagreen")

        
        ax.plot(data_l[0, cut1:cut2], data_l[1, cut1:cut2], 'slateblue'  , linewidth=1)
        ax.plot(data_l[0, cut2:cut3], data_l[1, cut2:cut3], 'darkmagenta' , linewidth=1)
        ax.plot(data_l[0, cut3:], data_l[1, cut3:], 'black' , linewidth=1)



    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    ax.set_xlabel(r'$\xi_1$', fontsize=34)
    ax.set_ylabel(r'$\xi_2$', fontsize=34)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig('damm.png', dpi=600, transparent=True)


    plt.show()


