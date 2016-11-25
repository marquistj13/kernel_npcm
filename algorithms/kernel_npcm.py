# -*- coding: utf-8 -*-
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import logging
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils import check_random_state
from kernel_kmeans import KernelKMeans

logging.captureWarnings(True)

colors = ['c', 'orange', 'g', 'r', 'b', 'm', 'y', 'k', 'Brown', 'ForestGreen'] * 30
plt.style.use('classic')


def exp_marginal(d, v0, sigma_v0):
    v_square = 0.5 * v0 ** 2 + sigma_v0 * d + 0.5 * v0 * np.sqrt(v0 ** 2 + 4 * sigma_v0 * d)
    return np.exp(-d ** 2 / v_square)


v_exp_marginal = np.vectorize(exp_marginal)


class kernel_npcm(BaseEstimator, ClusterMixin):
    def __init__(self, X, m_ini, alpha_cut=0.1, ax=None, x_lim=None, y_lim=None, error=1e-5, maxiter=10000,
                 ini_save_name="", last_frame_name="", save_figsize=(8, 6), random_state=None,
                 kernel="linear", gamma=None, degree=3, coef0=1, kernel_params=None, verbose=0):
        """
        :param X: scikit-learn form, i.e., pf shape (n_samples, n_features)
        :param m_ini: NO.of initial clusters
        :param sig_v0:
        :return:
        """
        self.x = X
        self.m = m_ini
        self.m_ori = m_ini  # the original number of clusters specified
        self.ax = ax
        self.x_lim = x_lim  # tuple
        self.y_lim = y_lim
        self.alpha_cut = alpha_cut
        self.save_figsize = save_figsize
        # alpha_cut can't be exactly 0, because we will use it to caculate sig_vj via 0.2* ita/ sqrt(log(alpha_cut))
        if abs(self.alpha_cut) < 1e-5:
            self.alpha_cut += 1e-5
        # alpha_cut also shouldn't be too large, for the same reason as above
        if abs(1 - self.alpha_cut) < 1e-5:
            self.alpha_cut -= 1e-5
        self.error = error
        self.maxiter = maxiter
        self.ini_save_name = ini_save_name
        self.last_frame_name = last_frame_name
        self.log = logging.getLogger('algorithm.npcm')
        self.random_state = random_state
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.verbose = verbose
        # use fcm to initialise the clusters
        # self.init_theta_ita()
        pass

    def init_animation(self):
        ax = self.ax
        # initialise the lines to update (each line represents a cluster)
        # this idea comes from http://stackoverflow.com/questions/19519587/python-matplotlib-plot-multi-lines-array-and-animation
        # i.e.,o animate N lines, you just need N Line2D objects
        self.lines = [ax.plot([], [], '.', color=colors[label])[0] for label in range(self.m_ori)]
        # centers
        # self.line_centers=[ax.plot([],[], 's',color=colors[label])[0] for label in range(self.m_ori) ]
        self.line_centers = [ax.plot([], [], 'rs')[0] for _ in range(self.m_ori)]
        # the title
        self.text = ax.text(0.02, 0.75, '', transform=ax.transAxes)
        # the limit of axixes
        ax.set_xlim(self.x_lim)
        ax.set_ylim(self.y_lim)
        # add circles to indicate the standard deviation, i.e., ita_j
        self.inner_circles = [
            ax.add_patch(plt.Circle((0, 0), radius=0, color='k', fill=None, lw=3.5, linestyle='dotted'))
            for _ in range(self.m_ori)]
        # add circles to indicatethe the radius at which the membership dereases to alpha when sigma_v=0
        self.circles = [ax.add_patch(plt.Circle((0, 0), radius=0, color='k', fill=None, lw=2, linestyle='solid')) for _
                        in range(self.m_ori)]
        # outer circles to indicate the radius at which the membership dereases to alpha
        self.outer_circles = [ax.add_patch(plt.Circle((0, 0), radius=0, color='k', fill=None, lw=2, linestyle='dashed'))
                              for _ in range(self.m_ori)]
        # remember to add the needs-to-update elments to the return list
        return self.lines + self.line_centers + self.inner_circles + self.circles + [self.text] + self.outer_circles

    def init_theta_ita(self):
        """
        This initialization is criticle because pcm based algorithms all rely on this initial 'beautiful' placement
        of the cluster prototypes.

        As we know, pcm is good at mode-seeking, and the algorithm works quite intuitively: you specify the location
        of lots of prototypes, then the algorithm tries to seek the dense region around the prototypes.

        Note that the prototypes has very little probability to move far from there initial locations. This fact
        quit annoys me because it reveals the mystery secret of clustering and makes clustering a trival work. This
        fact also makes clustering unattractive any more.

        Recall that the motivation of pcm is to remove the strong constraint imposed on the memberships as in fcm, and this modification
        does have very good results, that is, the resulting memberships finally have the interpretation of typicality
        which is one of the most commonly used interpretations of memberships in applications of fuzzy set theory,
        and beacuse of this improvment, the algorithm behaves more wisely under noisy environment.

        I start to doubt the foundations of the clustering community.

        :return:
        """
        clf = KernelKMeans(self.m_ori, random_state=45).fit(self.x)
        # hard classification labels
        labels = clf.labels_
        # u
        u = np.zeros((np.shape(self.x)[0], self.m_ori))
        u[xrange(len(u)), labels] = 1
        self.u = u
        # theta
        self.theta = [np.average(self.x[labels == i], axis=0) for i in xrange(self.m_ori)]
        # get eta
        self.eta = clf.eta

        self.log.debug("Initialize bandwidth via KMeans")
        for cntr_index in range(self.m_ori):
            self.log.debug("%d th cluster, eta:%3f" % (cntr_index, self.eta[cntr_index]))

        # plot the fcm initialization result
        fig = plt.figure("KMeans_init", dpi=300, figsize=self.save_figsize)
        ax = fig.gca()
        for label in range(self.m):
            ax.plot(self.x[labels == label][:, 0], self.x[labels == label][:, 1], '.',
                    color=colors[label])
            ax.text(self.theta[label][0], self.theta[label][1], "%d" % label, size='xx-large')
        ax.set_xlim(self.x_lim)
        ax.set_ylim(self.y_lim)
        ax.grid(True)
        # ax.set_title('KMeans initialization:%2d clusters' % self.m)
        plt.savefig(self.ini_save_name, dpi=fig.dpi, bbox_inches='tight')
        plt.close("KMeans_init")

        # eliminate noise clusters
        density_list = []  # store density each cluster
        for index in range(self.m):
            no_of_pnts = np.sum(labels == index)
            density = no_of_pnts / np.power(self.eta[index], np.shape(self.x)[1])
            density_list.append(density)
        index_delete = []  # store the cluster index to be deleted
        p = 0
        max_density = max(density_list)  # the maximum density
        for index in range(self.m):
            if density_list[index] < 0.1 * max_density:
                index_delete.append(index)
                p += 1
        for index in range(self.m):
            self.log.debug("%d th cluster, ita:%.3f, density:%.3f", index, self.eta[index], density_list[index])
        self.log.debug("Noise cluster delete list:%s", index_delete)
        self.theta = np.delete(self.theta, index_delete, axis=0)
        self.eta = np.delete(self.eta, index_delete, axis=0)
        self.m -= p

        pass

    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma,
                      "degree": self.degree,
                      "coef0": self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel,
                                filter_params=True, **params)

    def _compute_dist_alpha(self, K, dist, within_distances, update_alpha):
        """Compute a n_samples x n_clusters distance matrix using the
        kernel trick."""
        # labels = np.argmax(self.u, axis=1)
        for j in xrange(self.m):
            mask = self.u[:, j] >= self.alpha_cut
            u_j = self.u[mask, j]
            KK = K[mask][:, mask]  # K[mask, mask] does not work.
            if update_alpha:
                dist_j = np.sum(np.outer(u_j, u_j) * KK / np.square(np.sum(u_j)))
                within_distances[j] = dist_j
                dist[:, j] += dist_j
            else:
                dist[:, j] += within_distances[j]
            # u_j is broadcasted to (n_sample,n_u_j)
            dist[:, j] += np.diag(K) - 2 * np.sum(u_j * K[:, mask], axis=1) / np.sum(u_j)
        dist = np.sqrt(dist)

    def update_u_theta(self):
        # update u (membership matrix)
        self.log.debug("Update parameters")
        self.ita_alpha_sigmaV = []
        self.ita_alpha_ori = []
        u = np.zeros((np.shape(self.x)[0], self.m))

        dist = np.zeros((self.x.shape[0], self.m))
        self.within_distances = np.zeros(self.m)
        self._compute_dist_alpha(self.K, dist, self.within_distances, update_alpha=True)
        for cntr_index in range(self.m):
            dist_2_cntr = dist[:, cntr_index]
            # caculate sigma_vj for each cluster
            tmp_sig_vj = 0.2 * self.eta[cntr_index] / np.sqrt(-np.log(self.alpha_cut))
            # tmp_sig_vj = 1
            # caculate the d_\alpha of each cluster, i.e., the outer bandwidth circle
            tmp_ita_alpha = np.sqrt(-np.log(self.alpha_cut)) * (self.eta[cntr_index] + np.sqrt(-np.log(self.alpha_cut))
                                                                * tmp_sig_vj)
            tmp_ita_ori = np.sqrt(-np.log(self.alpha_cut)) * self.eta[cntr_index]
            self.ita_alpha_sigmaV.append(tmp_ita_alpha)
            self.ita_alpha_ori.append(tmp_ita_ori)
            self.log.debug("%d th cluster, ita:%3f, sig_v:%3f, d_alpha_corrected:%3f, d_alpha_ori:%3f" %
                           (cntr_index, self.eta[cntr_index], tmp_sig_vj, tmp_ita_alpha, tmp_ita_ori))
            u[:, cntr_index] = v_exp_marginal(dist_2_cntr, self.eta[cntr_index], tmp_sig_vj)
        self.u = u
        # update theta (centers), maybe for plot use. Note that in the high-dimensional feature space, we can't know
        # the theta because we can't know the mapping \phi generally
        for cntr_index in range(self.m):
            # only those without too much noise can be used to calculate centers
            samples_mask = u[:, cntr_index] >= self.alpha_cut
            if np.any(samples_mask):  # avoid null value for the following calculation
                self.theta[cntr_index] = np.sum(u[samples_mask][:, cntr_index][:, np.newaxis]
                                                * self.x[samples_mask], axis=0) / sum(u[samples_mask][:, cntr_index])

        pass

    def cluster_elimination(self):
        labels = np.argmax(self.u, axis=1)
        p = 0
        index_delete = []  # store the cluster index to be deleted
        for cntr_index in range(self.m):
            if np.any(labels == cntr_index):
                continue
            else:
                p += 1
                index_delete.append(cntr_index)
        # remove the respective center related quantities
        if p > 0:
            self.u = np.delete(self.u, index_delete, axis=1)
            self.theta = np.delete(self.theta, index_delete, axis=0)
            self.eta = np.delete(self.eta, index_delete, axis=0)
            self.ita_alpha_ori = np.delete(self.ita_alpha_ori, index_delete, axis=0)
            self.ita_alpha_sigmaV = np.delete(self.ita_alpha_sigmaV, index_delete, axis=0)
            self.m -= p
            self.log.critical("/******************************In cluster_elimination******************************/")
            self.log.critical(" %d clusters eliminated!", p)
            self.log.critical("/******************************End elimination******************************/")

        pass

    def _compute_dist_eta(self, K):
        """Compute a n_samples x n_clusters distance matrix using the
        kernel trick."""
        eta = []
        labels = np.argmax(self.u, axis=1)
        for j in xrange(self.m):
            eta_j = 0
            mask_eta = np.logical_and(self.u[:, j] >= 0.01, labels == j)
            if np.any(mask_eta):
                mask_theta = self.u[:, j] >= self.alpha_cut
                KK = K[mask_theta][:, mask_theta]  # K[mask, mask] does not work.
                u_j = self.u[mask_theta, j]
                dist_j = np.sum(np.outer(u_j, u_j) * KK / np.square(np.sum(u_j)))
                eta_j += dist_j
                # u_j is broadcasted to (n_eta_j,n_u_j)
                eta_j += np.diag(K[mask_eta][:, mask_eta]) - 2 * np.sum(u_j * K[mask_eta][:, mask_theta],
                                                                        axis=1) / np.sum(u_j)
                eta_j = np.average(eta_j)
                eta.append(eta_j)
            else:
                eta.append(0)
        return eta

    def adapt_eta(self):
        """
        in the hard partition, if no point belongs to cluster i then it will be removed.
        :return:
        """
        p = 0
        index_delete = []  # store the cluster index to be deleted

        self.eta = self._compute_dist_eta(self.K)
        for cntr_index in range(self.m):
            if np.isclose(self.eta[cntr_index], 0):
                p += 1
                index_delete.append(cntr_index)
                # remove the respective center related quantities
        if p > 0:
            self.u = np.delete(self.u, index_delete, axis=1)
            self.theta = np.delete(self.theta, index_delete, axis=0)
            self.eta = np.delete(self.eta, index_delete, axis=0)
            self.ita_alpha_ori = np.delete(self.ita_alpha_ori, index_delete, axis=0)
            self.ita_alpha_sigmaV = np.delete(self.ita_alpha_sigmaV, index_delete, axis=0)
            self.m -= p
        pass

    def save_last_frame(self, p):
        fig = plt.figure("last frame", dpi=300, figsize=self.save_figsize)
        ax = fig.gca()
        ax.grid(True)
        # the limit of axixes
        ax.set_xlim(self.x_lim)
        ax.set_ylim(self.y_lim)
        tmp_text = "Iteration times:%2d\n" % p
        tmp_text += r"$\alpha={:.2f}$".format(self.alpha_cut) + "\n"
        tmp_text += "Initial    number:%2d\nCurrent number:%2d" % (self.m_ori, self.m)
        ax.text(0.02, 0.75, tmp_text, transform=ax.transAxes)
        # ax.set_title("Clustering Finished")
        labels = np.argmax(self.u, axis=1)
        for label in range(self.m):
            ax.plot(self.x[labels == label][:, 0], self.x[labels == label][:, 1], '.', color=colors[label], zorder=1)
            ax.plot(self.theta[label][0], self.theta[label][1], 'rs', zorder=2)
            ax.add_patch(plt.Circle((self.theta[label][0], self.theta[label][1]), zorder=3,
                                    radius=self.eta[label], color='k', fill=None, lw=3.5, linestyle='dotted'))
            ax.add_patch(plt.Circle((self.theta[label][0], self.theta[label][1]), zorder=3,
                                    radius=self.ita_alpha_ori[label], color='k', fill=None, lw=2, linestyle='solid'))
            ax.add_patch(plt.Circle((self.theta[label][0], self.theta[label][1]), zorder=3,
                                    radius=self.ita_alpha_sigmaV[label], color='k', fill=None, lw=2,
                                    linestyle='dashed'))
        plt.figure("last frame")
        plt.savefig(self.last_frame_name, dpi=300, bbox_inches='tight')
        plt.close("last frame")
        pass

    def fit(self):
        """
         # This re-initialization is necessary if we use animation.save. The reason is: FuncAnimation needs a
        # save_count parameter to know the  mount of frame data to keep around for saving movies. So the animation
        # first runs the fit() function to get the number of runs of the algorithm and save the movie, then this number
        #  is  the run times for the next animation run. This second run is the one we see, not the one we save.
        #  So we should make sure that the second run of fit() has exactly the same enviroment as the first run.
        :return:
        """
        # The main loop
        p = 0
        self.m = self.m_ori
        self.init_animation()
        self.init_theta_ita()
        self.K = self._get_kernel(self.x)
        while p < self.maxiter:
            theta_ori = self.theta.copy()
            self.update_u_theta()
            self.cluster_elimination()
            self.adapt_eta()
            if (len(theta_ori) == len(self.theta)) and (np.linalg.norm(self.theta - theta_ori) < self.error):
                self.save_last_frame(p)
                break
            p += 1
            yield p  # here the current iteration result has been recorded in the class, the result is ready for plotting.
            # note that the yield statement returns p as an argument to the callback function __call__(self, p) which is called by the
            # animation process

    def __call__(self, p):
        """
        (refer to 74.4 animation example code: bayes_update.py from Matplotlib, Release 1.4.3 page1632)
        :param p:
        :return:
        """
        self.log.info("/******************************%d th iteration******************************/", p)
        tmp_text = "Iteration times:%2d\n" % p
        tmp_text += r"$\alpha={:.2f}$".format(self.alpha_cut) + "\n"
        labels = np.argmax(self.u, axis=1)
        # the following logic is as this: if the final cluster number is equal to the specified value
        # then draw all the clusters, otherwise, the deleted clusters are not plotted
        if self.m == self.m_ori:
            for label, line, line_center, inner_circle, circle, outer_circle \
                    in zip(range(self.m), self.lines, self.line_centers, self.inner_circles, self.circles,
                           self.outer_circles):
                line.set_data(self.x[labels == label][:, 0], self.x[labels == label][:, 1])
                line_center.set_data(self.theta[label][0], self.theta[label][1])
                inner_circle.center = self.theta[label][0], self.theta[label][1]
                inner_circle.set_radius(self.eta[label])
                circle.center = self.theta[label][0], self.theta[label][1]
                circle.set_radius(self.ita_alpha_ori[label])
                outer_circle.center = self.theta[label][0], self.theta[label][1]
                outer_circle.set_radius(self.ita_alpha_sigmaV[label])
                # print label, self.ita[label], len(self.ita)
        else:
            for label, line, line_center, inner_circle, circle, outer_circle \
                    in zip(range(self.m), self.lines[:self.m], self.line_centers[:self.m], self.inner_circles[:self.m],
                           self.circles[:self.m], self.outer_circles[:self.m]):
                line.set_data(self.x[labels == label][:, 0], self.x[labels == label][:, 1])
                line_center.set_data(self.theta[label][0], self.theta[label][1])
                inner_circle.center = self.theta[label][0], self.theta[label][1]
                inner_circle.set_radius(self.eta[label])
                circle.center = self.theta[label][0], self.theta[label][1]
                circle.set_radius(self.ita_alpha_ori[label])
                outer_circle.center = self.theta[label][0], self.theta[label][1]
                outer_circle.set_radius(self.ita_alpha_sigmaV[label])
                # self.log.info("Total %d clusters, %d th bandwidth %f" % (len(self.ita), label, self.ita[label]))
            for label, line, line_center, inner_circle, circle, outer_circle \
                    in zip(range(self.m, self.m_ori), self.lines[self.m:], self.line_centers[self.m:],
                           self.inner_circles[self.m:], self.circles[self.m:], self.outer_circles[self.m:]):
                line.set_data([], [])
                line_center.set_data([], [])
                inner_circle.set_radius(0)
                circle.set_radius(0)
                outer_circle.set_radius(0)
        tmp_text += "Initial    number:%2d\nCurrent number:%2d" % (self.m_ori, self.m)
        self.text.set_text(tmp_text)
        # remember to add the needs-to-update elments to the return list
        return self.lines + self.line_centers + self.inner_circles + self.circles + [self.text] + self.outer_circles
