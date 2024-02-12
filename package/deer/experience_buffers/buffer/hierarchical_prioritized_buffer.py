# -*- coding: utf-8 -*-
from deer.experience_buffers.buffer.pseudo_prioritized_buffer import *

import numpy as np
from sklearn.cluster import BisectingKMeans
import torch
from ray.rllib.policy.sample_batch import MultiAgentBatch

logger = logging.getLogger(__name__)


class HierarchicalPrioritizedBuffer(PseudoPrioritizedBuffer):
    def __init__(self, **configs):  # O(1)
        super(HierarchicalPrioritizedBuffer, self).__init__(**configs)
        self.clustering = None
        self.cluster_priority_list = []
        self.embedding_fn = None

    def clean(self):  # O(1)
        super().clean()
        self.cluster_priority_list = []

    def _add_type_if_not_exist(self, type_id):  # O(1)
        exists = super()._add_type_if_not_exist(type_id)
        if not exists:
            return False
        self.cluster_priority_list.append(0)
        return True

    def build_clusters(self, embedding_fn):
        self.embedding_fn = embedding_fn
        buffer_item_list = [element
                            for batch in self.batches for element in batch]
        buffer_item_list = [element.policy_batches['default_policy']
                            if isinstance(element, MultiAgentBatch)
                            else element
                            for element in buffer_item_list]
        # buffer_item_list = list(np.random.choice(
        #     buffer_item_list, size=2048, replace=False))
        buffer_embedding_iter = self.embedding_fn(buffer_item_list)
        self.clustering = BisectingKMeans(n_clusters=100)
        buffer_label_list = self.clustering.fit_predict(
            buffer_embedding_iter.detach().numpy()).tolist()
        self.clean()
        for i, l in zip(buffer_label_list, buffer_label_list):
            get_batch_infos(i)['batch_index'] = {}
            self.add(i, type_id=l)

    # self._cluster_labels = ms.labels_
    # self._cluster_centers = ms.cluster_centers_

    def update_beta_weights(self, batch, type_):
        tot_cluster_priority = sum(self.cluster_priority_list)
        ##########
        # Get priority weight
        get_probability = lambda x: self.cluster_priority_list[
                                        x] / tot_cluster_priority
        this_probability = get_probability(
            type_)  # clusters priorities are already > 0
        min_probability = min(map(get_probability, self.type_values))
        weight = min_probability / this_probability
        weight = weight ** self._prioritization_importance_beta
        ##########
        # Add age weight
        # if self._weight_importance_by_update_time:
        # 	weight *= self.get_age_weight(type_, idx) # batches with outdated priorities should have a lower weight, they might be just noise
        ##########
        batch[PRIO_WEIGHTS] = np.full(batch.count, weight, dtype=np.float32)

    def sample(self, n=1, **kwargs):  # O(log)
        _, type_ = self.sample_cluster()
        cluster_buffer = self.batches[type_]
        batch_list = random.choices(cluster_buffer, k=n)
        # Update weights
        if self._prioritization_importance_beta:  # Update weights
            for batch in batch_list:
                self.update_beta_weights(batch, type_)
        return batch_list

    def _cache_priorities(self):
        pass

    def update_priority(self, new_batch, idx, type_id=0):  # O(log)
        normalized_priority = super().update_priority(new_batch, idx, type_id=type_id)
        type_ = self.get_type(type_id)
        self.cluster_priority_list[type_] = normalized_priority ** self._cluster_prioritization_alpha

    def get_cluster_priority(self, segment_tree, min_priority=0):
        assert True, 'This function is not well-defined'

    def get_cluster_priority_dict(self):
        return dict(map(
            lambda x: (str(self.type_keys[x[0]]), x[1]),
            enumerate(self.cluster_priority_list)
        ))

    def add(self, batch, update_prioritisation_weights=False,
            **args):  # O(log)
        if self.clustering:
            if isinstance(batch, MultiAgentBatch):
                batch = batch.policy_batches['default_policy']
            embedding = self.embedding_fn(batch)
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.detach().numpy()
            type_id = self.clustering.predict(embedding).tolist()[
                0]
        else:
            type_id = 0  # add to the same cluster if no clustering is available
        return super().add(
            batch, type_id=type_id,
            update_prioritisation_weights=update_prioritisation_weights)
