# import sys
import numpy as np

import seaborn as sns
import skimage

import cv2
import matplotlib
# matplotlib.use('tkagg')
# if sys.platform == 'darwin':
#     matplotlib.use("tkagg")
# else:
#     matplotlib.use('Agg')
import matplotlib.pyplot as plt



def visualize(fig, ax, img, grid, pos, gt_pos, dump_dir, rank, ep_no, t,
              visualize, print_images, vis_style):
    for i in range(2):
        ax[i].clear()
        ax[i].set_yticks([])
        ax[i].set_xticks([])
        ax[i].set_yticklabels([])
        ax[i].set_xticklabels([])

    ax[0].imshow(img)
    ax[0].set_title("Observation", family='sans-serif',
                    fontname='Helvetica',
                    fontsize=20)

    if vis_style == 1:
        title = "Predicted Map and Pose"
    else:
        title = "Ground-Truth Map and Pose"

    ax[1].imshow(grid)
    ax[1].set_title(title, family='sans-serif',
                    fontname='Helvetica',
                    fontsize=20)

    # Draw GT agent pose
    agent_size = 8
    x, y, o = gt_pos
    x, y = x * 100.0 / 5.0, grid.shape[1] - y * 100.0 / 5.0

    dx = 0
    dy = 0
    fc = 'Grey'
    dx = np.cos(np.deg2rad(o))
    dy = -np.sin(np.deg2rad(o))
    ax[1].arrow(x - 1 * dx, y - 1 * dy, dx * agent_size, dy * (agent_size * 1.25),
                head_width=agent_size, head_length=agent_size * 1.25,
                length_includes_head=True, fc=fc, ec=fc, alpha=0.9)

    # Draw predicted agent pose
    x, y, o = pos
    x, y = x * 100.0 / 5.0, grid.shape[1] - y * 100.0 / 5.0

    dx = 0
    dy = 0
    fc = 'Red'
    dx = np.cos(np.deg2rad(o))
    dy = -np.sin(np.deg2rad(o))
    ax[1].arrow(x - 1 * dx, y - 1 * dy, dx * agent_size, dy * agent_size * 1.25,
                head_width=agent_size, head_length=agent_size * 1.25,
                length_includes_head=True, fc=fc, ec=fc, alpha=0.6)

    for _ in range(5):
        plt.tight_layout()

    if visualize:
        plt.gcf().canvas.flush_events()
        fig.canvas.start_event_loop(0.001)
        plt.gcf().canvas.flush_events()

    if print_images:
        fn = '{}/episodes/{}/{}/{}-{}-Vis-{}.png'.format(
            dump_dir, (rank + 1), ep_no, rank, ep_no, t)
        plt.savefig(fn)


def insert_circle(mat, x, y, value):
    mat[x - 2: x + 3, y - 2:y + 3] = value
    mat[x - 3:x + 4, y - 1:y + 2] = value
    mat[x - 1:x + 2, y - 3:y + 4] = value
    return mat


def fill_color(colored, mat, color):
    for i in range(3):
        colored[:, :, 2 - i] *= (1 - mat)
        colored[:, :, 2 - i] += (1 - color[i]) * mat
    return colored


def get_colored_map(mat, collision_map, visited, visited_gt, goal,
                    explored, gt_map, gt_map_explored):
    m, n = mat.shape
    colored = np.zeros((m, n, 3))
    pal = sns.color_palette("Paired")

    current_palette = [(0.9, 0.9, 0.9)]
    colored = fill_color(colored, gt_map, current_palette[0])

    current_palette = [(235. / 255., 243. / 255., 1.)]
    colored = fill_color(colored, explored, current_palette[0])

    green_palette = sns.light_palette("green")
    colored = fill_color(colored, mat, pal[2])

    current_palette = [(0.6, 0.6, 0.6)]
    colored = fill_color(colored, gt_map_explored, current_palette[0])

    colored = fill_color(colored, mat * gt_map_explored, pal[3])

    red_palette = sns.light_palette("red")

    colored = fill_color(colored, visited_gt, current_palette[0])
    colored = fill_color(colored, visited, pal[4])
    colored = fill_color(colored, visited * visited_gt, pal[5])

    colored = fill_color(colored, collision_map, pal[2])

    current_palette = sns.color_palette()

    selem = skimage.morphology.disk(4)
    goal_mat = np.zeros((m, n))
    goal_mat[goal[0], goal[1]] = 1
    goal_mat = 1 - skimage.morphology.binary_dilation(
        goal_mat, selem) != True

    colored = fill_color(colored, goal_mat, current_palette[0])

    current_palette = sns.color_palette("Paired")

    colored = 1 - colored
    colored *= 255
    colored = colored.astype(np.uint8)
    return colored



def get_observed_colored_map(obs_map, explored, visited, goal, gt_map=None, gt_visited=None, pose=None, save_fig=False, fig_name=None):
    m, n = obs_map.shape
    colored = np.zeros((m, n, 3))
    pal = sns.color_palette("Paired")

    if gt_map is not None:
        current_palette = [(0.8, 0.8, 0.8)]
        colored = fill_color(colored, gt_map, current_palette[0])

    current_palette = [(235. / 255., 243. / 255., 1.)]
    colored = fill_color(colored, explored, current_palette[0])

    green_palette = sns.light_palette("green")
    colored = fill_color(colored, obs_map, pal[2])
    if gt_map is not None:
        current_palette = [(0.6, 0.6, 0.6)]
        colored = fill_color(colored, obs_map * gt_map, pal[3])

    current_palette = [(0.9, 0.9, 0.9)]
    map_explored = explored * obs_map
    colored = fill_color(colored, explored, current_palette[0])
    colored = fill_color(colored, obs_map * map_explored, pal[3])

    red_palette = sns.light_palette("red")
    if gt_visited is not None:
        colored = fill_color(colored, gt_visited, current_palette[0])
    colored = fill_color(colored, visited, pal[4])
    if gt_visited is not None:
        colored = fill_color(colored, visited * gt_visited, pal[5])


    current_palette = sns.color_palette()

    selem = skimage.morphology.disk(4)
    goal_mat = np.zeros((m, n))
    goal_mat[goal[0], goal[1]] = 1
    goal_mat = 1 - skimage.morphology.binary_dilation(
        goal_mat, selem) != True

    colored = fill_color(colored, goal_mat, current_palette[0])

    current_palette = sns.color_palette("Paired")

    colored = 1 - colored
    colored *= 255
    colored = colored.astype(np.uint8)

    if save_fig:
        agent_size = 8
        x, y, o = pose
        x, y = x / 5.0, colored.shape[1] - y / 5.0

        figure, ax = plt.subplots(1, 1, figsize=(6 * 16 / 9, 6), facecolor="whitesmoke")

        ax.clear()
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_yticklabels([])
        ax.set_xticklabels([])

        ax.imshow(colored, interpolation='none', origin='lower')

        dx = 0
        dy = 0
        fc = 'Grey'
        # dx = np.cos(np.deg2rad(o))
        # dy = -np.sin(np.deg2rad(o))
        dx = np.cos(o)
        dy = np.sin(o)
        ax.arrow(x - 1 * dx, y - 1 * dy, dx * agent_size, dy * (agent_size * 1.25),
                    head_width=agent_size, head_length=agent_size * 1.25,
                    length_includes_head=True, fc=fc, ec=fc, alpha=0.9)

        if fig_name is None:
            fig_name = 'test'
        plt.savefig(fig_name + '.png')

    return colored



def visualize_obs_and_map(img, depth, colored, pose, gt_pose=None, fig_name=None, save_as_png=False, save_as_video=False):
    figure, ax = plt.subplots(1, 3, figsize=(6 * 16 / 9, 6), facecolor="whitesmoke")
    for i in range(3):
        ax[i].clear()
        ax[i].set_yticks([])
        ax[i].set_xticks([])
        ax[i].set_yticklabels([])
        ax[i].set_xticklabels([])

    ax[0].imshow(img)
    ax[0].set_title("Observation", family='sans-serif',
                    fontname='Helvetica',
                    fontsize=20)

    ax[1].imshow(depth)
    ax[1].set_title("Depth", family='sans-serif',
                    fontname='Helvetica',
                    fontsize=20)

    title = "Local Map and Pose"

    ax[2].imshow(colored, interpolation='none', origin='lower')
    ax[2].set_title(title, family='sans-serif',
                    fontname='Helvetica',
                    fontsize=20)

    if gt_pose is not None:
        # Draw GT agent pose
        agent_size = 8
        x, y, o = gt_pose
        x, y = x / 5.0, y / 5.0

        fc = 'Grey'
        dx = np.cos(o)
        dy = np.sin(o)
        ax[2].arrow(x - 1 * dx, y - 1 * dy, dx * agent_size, dy * (agent_size * 1.25),
                    head_width=agent_size, head_length=agent_size * 1.25,
                    length_includes_head=True, fc=fc, ec=fc, alpha=0.9)

    # Draw agent pose
    agent_size = 8
    x, y, o = pose
    x, y = x / 5.0, y / 5.0


    fc = 'Red'
    dx = np.cos(o)
    dy = np.sin(o)
    ax[2].arrow(x - 1 * dx, y - 1 * dy, dx * agent_size, dy * (agent_size * 1.25),
             head_width=agent_size, head_length=agent_size * 1.25,
             length_includes_head=True, fc=fc, ec=fc, alpha=0.9)

    for _ in range(5):
        plt.tight_layout()

    if fig_name is None:
        fig_name = 'test'

    if save_as_png:
        plt.savefig(fig_name + '.png')
    if save_as_video:
        figure.canvas.draw()
        out_img = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        out_img = out_img.reshape(figure.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return out_img
    plt.close()
    return None

