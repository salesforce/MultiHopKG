"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Error Analysis.
"""

import json
import os
import pickle

from src.error_analysis.analysis_modules import ModelErrors
from src.error_analysis.analysis_modules import compute_venn_areas


model_paths = {}
model_paths['umls'] = {
    'conve': "model/umls-conve-RV-xavier-200-200-0.003-32-3-0.3-0.3-0.2-0.1",
    'distmult': "model/umls-distmult-xavier-200-200-0.003-0.3-0.1",
    'complex': "model/umls-complex-xavier-200-200-0.003-0.3-0.1",
    'pg': "model/umls-point-xavier-n/a-200-200-3-0.001-0.3-0.1-0.7-256-0.05",
    'pg+conve': "model/umls-point.rs-xavier-n/a-200-200-3-0.001-0.3-0.1-0.7-256-0.05"
}
model_paths['kinship'] = {
    'conve': "model/kinship-conve-RV-xavier-200-200-0.003-32-3-0.2-0.3-0.2-0.1",
    'distmult': "model/kinship-distmult-xavier-200-200-0.003-0.3-0.1",
    'complex': "model/kinship-complex-RV-xavier-200-200-0.003-0.3-0.1",
    'pg': "model/kinship-point-xavier-n/a-200-200-3-0.001-0.3-0.1-0.9-256-0.05",
    'pg+conve': "model/kinship-point.rs-xavier-n/a-200-200-3-0.001-0.3-0.1-0.9-256-0.05"
}
model_paths['fb15k-237'] = {
    'conve': "model/FB15K-237-conve-RV-xavier-200-200-0.003-32-3-0.3-0.3-0.2-0.1",
    'distmult': "model/FB15K-237-distmult-xavier-200-200-0.003-0.3-0.1",
    'complex': "model/FB15K-237-complex-RV-xavier-200-200-0.003-0.3-0.1",
    'pg': "model/FB15K-237-point-xavier-n/a-200-200-3-0.001-0.3-0.1-0.5-256-0.02",
    'pg+conve': "model/FB15K-237-point.rs-xavier-n/a-200-200-3-0.001-0.3-0.1-0.5-256-0.02"
}
model_paths['wn18rr'] = {
    'conve': 'model/WN18RR-conve-RV-xavier-200-200-0.003-32-3-0.3-0.3-0.2-0.1',
    'distmult': 'model/WN18RR-distmult-xavier-200-200-0.003-0.2-0.1',
    'complex': 'model/WN18RR-complex-xavier-200-200-0.003-0.3-0.1',
    'pg': 'model/WN18RR-point-xavier-n/a-200-200-3-0.001-0.5-0.3-0.5-500-0.0',
    'pg+conve': ''
}
model_paths['nell-995'] = {}

def compare_models(dataset, model_names):

    def read_error_cases(input_dir):
        input_path = os.path.join(input_dir, 'error_cases.txt')
        print('loading error cases from: {}'.format(input_path))
        base_dir = input_dir.split('/')[1]
        if base_dir.startswith('FB15K-237'):
            model_name = base_dir.split('-')[2]
        else:
            model_name = base_dir.split('-')[1]
        model_errors = ModelErrors(model_name.upper())
        with open(input_path, 'rb') as f:
            top_1_ecs, top_10_ecs = pickle.load(f)
            model_errors.top_1_error_cases = set(top_1_ecs)
            model_errors.top_10_error_cases = set(top_10_ecs)
        return model_errors

    model_error_list = []
    for model_name in model_names:
        model_error_list.append(read_error_cases(model_paths[dataset][model_name]))

    subset_overlap = compute_venn_areas(model_error_list)
    experiment = {}
    experiment['dataset'] = dataset.upper()    
    experiment['top-1'] = []
    experiment['top-10'] = []
    for i in sorted(subset_overlap.keys()):
        if subset_overlap[i].name:
            experiment['top-1'].append({'name': '{}'.format(subset_overlap[i].name.replace(' & ', ',')),
                                        'value': len(subset_overlap[i].top_1_error_cases)})
            experiment['top-10'].append({'name': '{}'.format(subset_overlap[i].name.replace(' & ', ',')),
                                        'value': len(subset_overlap[i].top_10_error_cases)})
    print(json.dumps(experiment, indent=4, sort_keys=True))

def main():
    dataset = 'fb15k-237'
    model_names = ['conve', 'distmult', 'complex']
    compare_models(dataset, model_names)

if __name__ == '__main__':
    main()
