"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Visualize beam search in the knowledge graph.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def visualize_step(action_dist, e, action_space, plot):
    action_space_size = len(action_space)
    plt_obj = plot.imshow(np.expand_dims(action_dist, 1), interpolation='nearest', cmap=plt.cm.Blues)
    plt.setp(plot, xticks=[0], xticklabels=[e], yticks=range(action_space_size), yticklabels=action_space)
    plot.xaxis.tick_top()
    plot.yaxis.tick_right()
    plot.tick_params(axis='both', which='major', labelsize=6)
    return plt_obj


def visualize_path(query, path_components, output_path=None):
    """
    :param query: String representation of the query.
    :param path_components:
        List of path component (e, action_space, action_dist)
            e - current node name
            (Numpy array) action_space - names of top k actions in the action space
            (Numpy array) action_dist - probabilities of top k actions in the action space
    :param output_path: Path to save the result graph.

    Visualize probabilities of all actions along a beam search paths and save the plots.
    """
    plt.clf()
    num_steps = len(path_components)
    gridspec_kwargs = dict(top=0.9, bottom=0.1, left=0.1, right=0.9, wspace=8, hspace=0.4)
    f, axarr = plt.subplots(num_steps, 1, gridspec_kw=gridspec_kwargs)
    for i, (e, action_space, action_dist) in enumerate(path_components):
        visualize_step(action_dist, e, action_space, axarr[i])

    plt.suptitle(query, fontsize=6)
    # f.colorbar(plt_obj)
    plt.show()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', format='png')
        print('path visualization saved to {}'.format(output_path))

    plt.close(f)
