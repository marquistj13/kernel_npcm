# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from algorithms import npcm_kernel_zero
from sklearn.datasets import make_blobs

colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']
plt.style.use('classic')

from moviepy.video.io.ffmpeg_reader import FFMPEG_VideoReader
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
import os

n_samples = 800


def _generateFig():
    """
    Two close clusters, one big and the other small,
    :return:
    """

    n_true_clusters = 3
    ##画图吧！
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=100, facecolor='white')  # 坐标轴刻度一致便于对比，但是若加了
    dpi = fig.dpi  # 取出上面设定的dpi值，注意这个dpi越大，字体越大

    np.random.seed(1)
    d = np.random.uniform(low=-6, high=6, size=(1000, 2))
    d1a = d[(d[:, 0] ** 2 + d[:, 1] ** 2 < 36) & (d[:, 0] ** 2 + d[:, 1] ** 2 > 25), :]
    ya = np.zeros(len(d1a))
    d = np.random.uniform(low=-4, high=4, size=(800, 2))
    # d1b = d[(d[:,0]**2 + d[:,1]**2 <9) & (d[:,0]**2 + d[:,1]**2 > 4),:]
    d1c = d[(d[:, 0] ** 2 + d[:, 1] ** 2 <= 3), :]
    yc = np.zeros(len(d1c)) + 1
    X = np.vstack((d1a, d1c))
    y = np.hstack((ya, yc))
    print("Cluster sizes %s " % [d1a.shape[0], d1c.shape[0]])  # , d1c.shape[0]])
    return X, y


if __name__ == '__main__':
    X, y = _generateFig()
    marker_size = 4
    dpi = 90
    fig_size = (8, 6)
    # plot ori data and save
    fig1 = plt.figure(figsize=fig_size, dpi=dpi, num=1)
    ax_fig1 = fig1.gca()
    ax_fig1.grid(True)
    for label in range(4):
        ax_fig1.plot(X[y == label][:, 0], X[y == label][:, 1], '.',
                     color=colors[label], markersize=marker_size, label="Cluster %d" % (label + 1))
    ax_fig1.set_xlim(-6, 6)
    ax_fig1.set_ylim(-6, 6)
    lg = ax_fig1.legend(loc='upper left', fancybox=True, framealpha=0.5, prop={'size': 8})
    ax_fig1.set_title("Original Dataset")
    plt.savefig(r".\video\fig8_ori.png", dpi=dpi, bbox_inches='tight')
    # plot animation and save
    fig2 = plt.figure(figsize=fig_size, dpi=dpi, num=2)
    ax = fig2.gca()
    ax.grid(True)
    n_cluster, sigma_v, alpha_cut = 10, 1, 0.1
    n_cluster, sigma_v, alpha_cut = 10, 0.5, 0.1
    n_cluster, sigma_v, alpha_cut = 10, 1, 0.5
    n_cluster, sigma_v, alpha_cut = 10, 1, 0.1
    ini_save_name = r".\video\fig8_ini_%d.png" % n_cluster
    last_frame_name = r'.\video\fig8_n_%d_sigmav_%.1f_alpha_%.1f_last_frame.png' % (n_cluster, sigma_v, alpha_cut)
    tmp_video_name = r'.\video\fig8_n_%d_sigmav_%.1f_alpha_%.1f_tmp.mp4' % (n_cluster, sigma_v, alpha_cut)
    video_save_newFps_name = r'.\video\fig8_n_%d_sigmav_%.1f_alpha_%.1f.mp4' % (n_cluster, sigma_v, alpha_cut)
    clf = npcm_kernel_zero(X, n_cluster, alpha_cut=alpha_cut, ax=ax, x_lim=(-6, 6), y_lim=(-6, 6),
                      ini_save_name=ini_save_name, last_frame_name=last_frame_name)
    # we should set "blit=False,repeat=False" or the program would fail. "init_func=clf.init_animation" plot the
    # background of each frame There is not much point to use blit=True, if most parts of your plot should be
    # refreshed. see http://stackoverflow.com/questions/14844223/python-matplotlib-blit-to-axes-or-sides-of-the
    # -figure
    # To begin with, if you're chaining the ticks, etc, there isn't much point in using blitting. Blitting is
    #  just a way to avoid re-drawing everything if only some things are changing. If everything is changing,
    # there's no point in using blitting. Just re-draw the plot.
    anim = animation.FuncAnimation(fig2, clf, frames=clf.fit,
                                   init_func=clf.init_animation, interval=1500, blit=True, repeat=False)
    # anim.save(tmp_video_name, fps=1, extra_args=['-vcodec', 'libx264'], dpi='figure')
    # new_fps = 24
    # play_slow_rate = 1.5  # controls how many times a frame repeats.
    # movie_reader = FFMPEG_VideoReader(tmp_video_name)
    # movie_writer = FFMPEG_VideoWriter(video_save_newFps_name, movie_reader.size, new_fps)
    # print "n_frames:", movie_reader.nframes
    # # the 1st frame of the saved video can't be directly read by movie_reader.read_frame(), I don't know why
    # # maybe it's a bug of anim.save, actually, if we look at the movie we get from anim.save
    # # we can easilly see that the 1st frame just close very soon.
    # # so I manually get it at time 0, this is just a trick, I think.
    # tmp_frame = movie_reader.get_frame(0)
    # [movie_writer.write_frame(tmp_frame) for _ in range(int(new_fps * play_slow_rate))]
    # # for the above reason, we should read (movie_reader.nframes-1) frames so that the last frame is not
    # # read twice (not that get_frame(0) alread read once)
    # # However, I soon figure out that it should be (movie_reader.nframes-2). The details: we have actually
    # # 6 frames, but (print movie_reader.nframes) is 7. I read the first frame through movie_reader.get_frame(0)
    # # then are are 5 left. So I should use movie_reader.nframes - 2. Note that in fig1_pcm_fs2.py
    # # in the case of: original fps=1
    # # new_fps = 24, play_slow_rate = 1.5 the result is: 1st frame last 1.8s, others 1.5s, i.e., the 1st frame
    # # has more duration. This is messy.
    # for i in range(movie_reader.nframes - 2):
    #     tmp_frame = movie_reader.read_frame()
    #     [movie_writer.write_frame(tmp_frame) for _ in range(int(new_fps * play_slow_rate))]
    #     pass
    # movie_reader.close()
    # movie_writer.close()
    # os.remove(tmp_video_name)
    plt.show()
    pass
