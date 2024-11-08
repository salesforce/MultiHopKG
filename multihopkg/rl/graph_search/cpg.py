from numpy import common_type
from torch._C import _cuda_tunableop_set_max_tuning_duration
from multihopkg.utils.ops import int_fill_var_cuda, var_cuda, zeros_var_cuda
from multihopkg.rl.graph_search.pn import GraphSearchPolicy, ITLGraphEnvironment
from multihopkg.utils import ops
import torch
from torch import nn
from multihopkg.knowledge_graph import ITLKnowledgeGraph
from typing import Tuple
import pdb


class ContinuousPolicyGradient(nn.Module):
    # TODO: remove all parameters that are irrelevant here
    def __init__(
        # TODO: remove all parameters that are irrelevant here
        self,
        baseline: str,
        beta: float,
        gamma: float,
        action_dropout_rate: float,
        action_dropout_anneal_factor: float,
        action_dropout_anneal_interval: float,
        num_rollout_steps: int,
        dim_action: int,
        dim_hidden: int,
        dim_observation: int,
    ):
        super(ContinuousPolicyGradient, self).__init__()

        # Training hyperparameters
        self.num_rollout_steps = num_rollout_steps
        self.baseline = baseline
        self.beta = beta  # entropy regularization parameter
        self.gamma = gamma  # shrinking factor
        self.action_dropout_rate = action_dropout_rate
        self.action_dropout_anneal_factor = (
            action_dropout_anneal_factor  # Used in parent
        )
        self.action_dropout_anneal_interval = (
            action_dropout_anneal_interval  # Also used by parent
        )

        ########################################
        # Torch Modules
        ########################################
        self.fc1, self.mu_layer, self.sigma_layer = self._define_modules(
            input_dim=dim_observation, observation_dim=dim_action, hidden_dim=dim_hidden
        )

        # # Inference hyperparameters
        # self.beam_size = beam_size
        # # Analysis
        # self.path_types = dict()
        # self.num_path_types = 0

    def forward(
        self, observations: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Once we do the observations we need to do the sampling
        return self._sample_action(observations)

    def _sample_action(
        self,
        observations: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Will sample batch_len actions given batch_len observations
        args
            observations: torch.Tensor. Shape: (batch_len, path_encoder_dim)
        """
        projections = self.fc1(observations)
        mu = self.mu_layer(projections)
        log_sigma = self.sigma_layer(projections)
        # log_sigma = torch.clamp(log_sigma, min=-20, max=2) # TODO: Check if this is needed
        sigma = torch.exp(log_sigma)

        # Create a normal distribution using the mean and standard deviation
        dist = torch.distributions.Normal(mu, sigma)
        entropy = dist.entropy().sum(dim=-1)  

        # Now Sample from it 
        # TODO: Ensure we are sampling correctly from this 
        actions = dist.rsample()
        log_probs = dist.log_prob(actions).sum(dim=-1)

        return actions,log_probs, entropy


    def _define_modules(self, input_dim:int, observation_dim: int, hidden_dim: int):

        fc1 = nn.Linear(input_dim, hidden_dim)
        
        mu_layer = nn.Linear(hidden_dim, observation_dim)
        sigma_layer = nn.Linear(hidden_dim, observation_dim)

        return fc1, mu_layer, sigma_layer

    def _reparemeteriztion(self, dist, action):
        return dist.log_prob(action).sum(dim=-1)





class ContinuousPolicy:

    def __init__(
        self,
        # Goodness this is ugly:
        action_dropout_anneal_factor: float,
        action_dropout_anneal_interval: float,
        action_dropout_rate: float,
        baseline: str,
        beam_size: int,
        beta: float,
        gamma: float,
        num_rollouts: int,
        num_rollout_steps: int,
        use_action_space_bucketing: bool,
    ):
        # Training hyperparameters
        self.use_action_space_bucketing = use_action_space_bucketing
        self.num_rollouts = num_rollouts
        self.num_rollout_steps = num_rollout_steps
        self.baseline = baseline
        self.beta = beta  # entropy regularization parameter
        self.gamma = gamma  # shrinking factor
        self.action_dropout_rate = action_dropout_rate
        self.action_dropout_anneal_factor = (
            action_dropout_anneal_factor  # Used in parent
        )
        self.action_dropout_anneal_interval = (
            action_dropout_anneal_interval  # Also used by parent
        )

        # TODO: PRepare more stuff here. You erased a lot
        # Inference hyperparameters
        self.beam_size = beam_size

        # Analysis
        self.path_types = dict()
        self.num_path_types = 0

    def reward_fun(self, e1, r, e2, pred_e2):
        # TODO: Soft reward here.
        raise NotImplementedError
        # return (pred_e2 == e2).float()

    #! This function is modified
    def rollout(self, e_s, q, e_t, num_steps, kg, pn):
        #! Changes:
        # * kg is passed as an argument
        # * pn is passed as an argument

        # TODO: Do we need to include the description of every passed parameter?
        # TODO: Do we need to document this method?
        """
        Rollout a batch of episodes.
        """
        assert num_steps > 0

        log_action_probs = []
        action_entropy = []

        r_s = int_fill_var_cuda(e_s.size(), kg.dummy_start_r)
        seen_nodes = int_fill_var_cuda(e_s.size(), kg.dummy_e).unsqueeze(1)
        path_components = []

        # Save some history
        path_trace = [(r_s, e_s)]
        # NOTE:(LG): Must be run as `.reset()` for ensuring environment `pn` is stup
        pn.initialize_path((r_s, e_s), kg)

        for t in range(num_steps):
            last_r, e = path_trace[-1]
            obs = [e_s, q, e_t, t == (num_steps - 1), last_r, seen_nodes]

            # * transit method has been remade, line below is modified
            db_outcomes, inv_offset = pn.transit(
                e, obs, kg, use_action_space_bucketing=self.use_action_space_bucketing
            )

            # * sample_action method has been remade, line below is modified
            sample_outcome, policy_entropy = self.sample_action(db_outcomes, inv_offset)

            action = sample_outcome["action_sample"]
            pn.update_path(action, kg)

            action_prob = sample_outcome["action_prob"]
            log_action_probs.append(ops.safe_log(action_prob))
            action_entropy.append(policy_entropy)
            seen_nodes = torch.cat([seen_nodes, e.unsqueeze(1)], dim=1)
            path_trace.append(action)

            # * If decided to visualize the action probabilities, uncomment
            # if visualize_action_probs:
            #     top_k_action = sample_outcome['top_actions']
            #     top_k_action_prob = sample_outcome['top_action_probs']
            #     path_components.append((e, top_k_action, top_k_action_prob))

        pred_e2 = path_trace[-1][1]
        self.record_path_trace(path_trace)

        return {
            "pred_e2": pred_e2,
            "log_action_probs": log_action_probs,
            "action_entropy": action_entropy,
            "path_trace": path_trace,
            "path_components": path_components,
        }

    def loss(self, mini_batch):

        # TODO: Check if we want to do that
        def stablize_reward(r):
            r_2D = r.view(-1, self.num_rollouts)
            if self.baseline == "avg_reward":
                stabled_r_2D = r_2D - r_2D.mean(dim=1, keepdim=True)
            elif self.baseline == "avg_reward_normalized":
                stabled_r_2D = (r_2D - r_2D.mean(dim=1, keepdim=True)) / (
                    r_2D.std(dim=1, keepdim=True) + ops.EPSILON
                )
            else:
                raise ValueError(
                    "Unrecognized baseline function: {}".format(self.baseline)
                )
            stabled_r = stabled_r_2D.view(-1)
            return stabled_r

        ##################################
        # Here we roll a batch of episodes
        ##################################
        e1, e2, r = format_batch(mini_batch, num_tiles=self.num_rollouts)
        output = self.rollout(e1, r, e2, num_steps=self.num_rollout_steps)

        ##################################
        # Compute metrics from output
        ##################################
        # Compute policy gradient loss
        pred_e2 = output["pred_e2"]
        log_action_probs = output["log_action_probs"]
        action_entropy = output["action_entropy"]

        # Compute discounted reward
        final_reward = self.reward_fun(e1, r, e2, pred_e2)
        if self.baseline != "n/a":
            final_reward = stablize_reward(final_reward)
        cum_discounted_rewards = [0] * self.num_rollout_steps
        cum_discounted_rewards[-1] = final_reward
        R = 0
        for i in range(self.num_rollout_steps - 1, -1, -1):
            R = self.gamma * R + cum_discounted_rewards[i]
            cum_discounted_rewards[i] = R

        # Compute policy gradient
        pg_loss, pt_loss = 0, 0
        for i in range(self.num_rollout_steps):
            log_action_prob = log_action_probs[i]
            pg_loss += -cum_discounted_rewards[i] * log_action_prob
            pt_loss += -cum_discounted_rewards[i] * torch.exp(log_action_prob)

        # Entropy regularization
        entropy = torch.cat([x.unsqueeze(1) for x in action_entropy], dim=1).mean(dim=1)
        pg_loss = (pg_loss - entropy * self.beta).mean()
        pt_loss = (pt_loss - entropy * self.beta).mean()

        loss_dict = {}
        loss_dict["model_loss"] = pg_loss
        loss_dict["print_loss"] = float(pt_loss)
        loss_dict["reward"] = final_reward
        loss_dict["entropy"] = float(entropy.mean())
        if self.run_analysis:
            fn = torch.zeros(final_reward.size())
            for i in range(len(final_reward)):
                if not final_reward[i]:
                    if int(pred_e2[i]) in self.kg.all_objects[int(e1[i])][int(r[i])]:
                        fn[i] = 1
            loss_dict["fn"] = fn

        return loss_dict


def format_batch(batch_data, num_labels=-1, num_tiles=1):
    """
    Convert batched tuples to the tensors accepted by the NN.
    """
    # TODO: Understand why this is needed

    # This is the tiling happening again.
    def convert_to_binary_multi_subject(e1):
        e1_label = zeros_var_cuda([len(e1), num_labels])
        for i in range(len(e1)):
            e1_label[i][e1[i]] = 1
        return e1_label

    def convert_to_binary_multi_object(e2):
        e2_label = zeros_var_cuda([len(e2), num_labels])
        for i in range(len(e2)):
            e2_label[i][e2[i]] = 1
        return e2_label

    batch_e1, batch_e2, batch_r = [], [], []
    for i in range(len(batch_data)):
        e1, e2, r = batch_data[i]
        batch_e1.append(e1)
        batch_e2.append(e2)
        batch_r.append(r)
    batch_e1 = var_cuda(torch.LongTensor(batch_e1), requires_grad=False)
    batch_r = var_cuda(torch.LongTensor(batch_r), requires_grad=False)
    if type(batch_e2[0]) is list:
        batch_e2 = convert_to_binary_multi_object(batch_e2)
    elif type(batch_e1[0]) is list:
        batch_e1 = convert_to_binary_multi_subject(batch_e1)
    else:
        batch_e2 = var_cuda(torch.LongTensor(batch_e2), requires_grad=False)
    # Rollout multiple times for each example
    if num_tiles > 1:
        batch_e1 = ops.tile_along_beam(batch_e1, num_tiles)
        batch_r = ops.tile_along_beam(batch_r, num_tiles)
        batch_e2 = ops.tile_along_beam(batch_e2, num_tiles)
    return batch_e1, batch_e2, batch_r


def define_path_encoder(
    action_dim: int,
    ff_dropout_rate: float,
    history_dim: int,
    history_num_layers: int,
) -> Tuple[nn.Module, nn.Module, nn.Module, nn.Module, nn.Module]:
    """
    We will deterine the input_dim outside of this

    """

    input_dim = action_dim
    W1 = nn.Linear(input_dim, action_dim)
    W2 = nn.Linear(action_dim, action_dim)

    W1Dropout = nn.Dropout(p=ff_dropout_rate)
    W2Dropout = nn.Dropout(p=ff_dropout_rate)

    path_encoder = nn.LSTM(
        input_size=action_dim,
        hidden_size=history_dim,
        num_layers=history_num_layers,
        batch_first=True,
    )
    return W1, W2, W1Dropout, W2Dropout, path_encoder
