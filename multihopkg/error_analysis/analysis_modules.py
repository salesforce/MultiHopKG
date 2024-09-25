"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Error Analysis Modules.
"""

class ModelErrors(object):
    def __init__(self, model_name):
        self.name = model_name
        self.top_1_error_cases = None
        self.top_10_error_cases = None

def compute_venn_areas(model_error_list):
    def intersect(m_e1, m_e2):
        if m_e1.name:
            inter_model_errors = ModelErrors('{} & {}'.format(m_e1.name, m_e2.name))
            inter_model_errors.top_1_error_cases = m_e1.top_1_error_cases & m_e2.top_1_error_cases
            inter_model_errors.top_10_error_cases = m_e1.top_10_error_cases & m_e2.top_10_error_cases
        else:
            inter_model_errors  = m_e2
        return inter_model_errors

    assert(len(model_error_list) > 1)
    all_top_1_errors = set([e for m_e in model_error_list for e in m_e.top_1_error_cases ])
    all_top_10_errors = set([e for m_e in model_error_list for e in m_e.top_10_error_cases])
    all_model_errors = ModelErrors('')
    all_model_errors.top_1_error_cases = all_top_1_errors
    all_model_errors.top_10_error_cases = all_top_10_errors
    subset_overlap = {
        0: all_model_errors,
        1: model_error_list[0]
    }
    num_sets = len(model_error_list)
    j = 1
    for i in range(2, 1 << num_sets):
        if i >= pow(2, j+1):
            j += 1
        res = i - pow(2, j)
        subset_overlap[i] = intersect(subset_overlap[res], model_error_list[j])

    print('Top 1 Error Cases')
    for i in sorted(subset_overlap.keys()):
        print('|{}|: {}'.format(subset_overlap[i].name, len(subset_overlap[i].top_1_error_cases)))
    print()
    print('Top 10 Error Cases')
    for i in sorted(subset_overlap.keys()):
        print('|{}|: {}'.format(subset_overlap[i].name, len(subset_overlap[i].top_10_error_cases)))

    return subset_overlap
