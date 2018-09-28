"""
Hyperparameter range specification.
"""

hp_range = {
    "beta": [0., .01, .02, .05, .1],
    "emb_dropout_rate": [0, .1, .2, .3],
    "ff_dropout_rate": [0, .1, .2, .3],
    "action_dropout_rate": [.95],
    "bandwidth": [200, 256, 400, 512],
    "relation_only": [True, False]
}
