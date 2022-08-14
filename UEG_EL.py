import os
import sys
import tensorflow as tf
from utility import Learner, Tool, DataGenerator, configs
from utility.AbstractRecommender import AbstractRecommender
from utility.Tool import timer, ensureDir
from utility.Dataset import Dataset as DATA
from utility.DataIterator import DataIterator
import numpy as np
import random
import logging
from time import time, localtime, strftime
import datetime
import pickle
import faiss
import scipy.sparse as sp

np.random.seed(2022)
random.seed(2022)
tf.set_random_seed(2022)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# os.environ['CUDA_VISIBLE_DEVICES'] = '4'


class PEG(AbstractRecommender):
    def __init__(self, sess, dataset, conf, train2cg=None, train2cg_idx=None):
        super(PEG, self).__init__(dataset, conf)

        # dataset
        self.dataset_name = dataset.dataset_name
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.num_valid_items = dataset.num_valid_items
        self.num_user_features = dataset.num_user_features
        self.num_item_features = dataset.num_item_featuers
        self.num_context_features = dataset.num_context_features
        self.dataset = dataset
        self.user_feature_mat = self.get_feature_matrix('user')
        self.item_feature_mat = self.get_feature_matrix('item')
        self.context_feature_mat = self.get_feature_matrix('context')
        self.num_contexts = len(self.context_feature_mat)
        # share the item id embedding in last purchase
        self.context_feature_wo_last = self.context_feature_mat[:, :-1]
        self.context_feature_last = self.context_feature_mat[:, -1]
        self.insts2userid = dataset.all_data_dict['train_data']['user_id'].to_list()
        self.insts2itemid = dataset.all_data_dict['train_data']['item_id'].to_list()
        self.insts2contextid = dataset.all_data_dict['train_data']['context_id'].to_list()
        self.num_context_fields = self.dataset.side_info['side_info_stats']['num_context_fields']
        self.adj_norm_type = conf.adj_norm_type
        if self.adj_norm_type in ['rs', 'rd', 'db']:
            self.user_neighbor_num, self.item_neighbor_num = self.cnt_neighbour_number(
                dataset.all_data_dict['train_data'])
            self.norm_user_neighbor_num = self.get_inv_neighbor_num(self.user_neighbor_num, self.adj_norm_type)
            self.norm_item_neighbor_num = self.get_inv_neighbor_num(self.item_neighbor_num, self.adj_norm_type)
        assert self.num_valid_items == len(self.item_feature_mat)

        self.train2cg = train2cg
        self.train2cg_idx = train2cg_idx

        # learning hyper-parameters
        self.batch_size = conf.batch_size
        self.test_batch_size = conf.test_batch_size
        self.learning_rate = conf.lr
        self.hidden_factor = conf.hidden_factor
        self.num_epochs = conf.epoch
        self.optimizer_type = conf.optimizer
        self.reg = conf.reg
        self.loss_type = conf.loss_type
        if self.loss_type == 'log_loss':
            self.num_negatives = conf.num_negatives
        self.init_method = conf.init_method
        self.stddev = conf.stddev
        self.num_gcn_layers = conf.num_gcn_layers
        self.gcn_layer_weight = conf.gcn_layer_weight
        self.merge_type = conf.merge_type
        self.decoder_type = conf.decoder_type
        if self.decoder_type == 'MLP':
            self.num_hidden_layers = conf.num_hidden_layers

        # other parameters
        self.pretrain_flag = conf.pretrain
        if self.pretrain_flag:
            self.read_file = args.read_file
        self.save_flag = conf.save_flag
        if self.save_flag:
            self.save_file = args.save_file

        self.sess = sess
        self.best_result = np.zeros([9], dtype=float)
        self.best_epoch = 0

    def get_feature_matrix(self, key_word):
        mat = self.dataset.side_info['%s_feature' % key_word]
        return mat.values if mat is not None else None

    def cnt_neighbour_number(self, df):
        user_neighbor_num = np.zeros([self.num_users], dtype=int)
        item_neighbor_num = np.zeros([self.num_valid_items], dtype=int)
        dict_user_neightbor_num = df['user_id'].value_counts().to_dict()
        dict_item_neightbor_num = df['item_id'].value_counts().to_dict()
        for id, value in dict_user_neightbor_num.items():
            user_neighbor_num[id] = value
        for id, value in dict_item_neightbor_num.items():
            item_neighbor_num[id] = value
        return user_neighbor_num, item_neighbor_num

    def get_inv_neighbor_num(self, data, norm_type):
        if norm_type in ['rs', 'db']:
            d_inv = np.power(data, -0.5).flatten()
        elif norm_type == 'rd':
            d_inv = np.power(data, -1.0).flatten()
        else:
            raise Exception("adj_norm_type is invalid.")
        d_inv[np.isinf(d_inv)] = 0.
        return d_inv

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None], name="user_input")
            self.context_input = tf.placeholder(tf.int32, shape=[None], name='context_input')
            self.item_input = tf.placeholder(tf.int32, shape=[None], name="item_input")
            self.labels = tf.placeholder(tf.float32, shape=[None], name='labels')

    def _create_variables(self):
        with tf.name_scope("embedding"):
            if self.pretrain_flag > 0:
                weight_saver = tf.train.import_meta_graph(self.read_file + '.meta')
                pretrain_graph = tf.get_default_graph()
                user_embeddings = pretrain_graph.get_tensor_by_name('embedding/embedding/user_embeddings:0')
                item_embeddings = pretrain_graph.get_tensor_by_name('embedding/embedding/item_embeddings:0')
                item_feature_embeddings = pretrain_graph.get_tensor_by_name(
                    'embedding/embedding/item_feature_embeddings:0')
                context_feature_embeddings = pretrain_graph.get_tensor_by_name(
                    'embedding/embedding/context_feature_embeddings:0')
                user_bias = pretrain_graph.get_tensor_by_name('embedding/embedding/user_bias:0')
                item_bias = pretrain_graph.get_tensor_by_name('embedding/embedding/item_bias:0')
                item_feature_bias = pretrain_graph.get_tensor_by_name('embedding/embedding/item_feature_bias:0')
                context_feature_bias = pretrain_graph.get_tensor_by_name('embedding/embedding/context_feature_bias:0')
                global_bias = pretrain_graph.get_tensor_by_name('embedding/embedding/global_bias:0')
                att_embed_1 = pretrain_graph.get_tensor_by_name('embedding/embedding/att_embed_1:0')
                att_bias_2 = pretrain_graph.get_tensor_by_name('embedding/embedding/att_bias_2:0')
                att_embed_2 = pretrain_graph.get_tensor_by_name('embedding/embedding/att_embed_2:0')
                att_embed_3 = pretrain_graph.get_tensor_by_name('embedding/embedding/att_embed_3:0')

                if self.user_feature_mat is not None:
                    user_feature_embeddings = pretrain_graph.get_tensor_by_name(
                        'embedding/embedding/user_feature_embeddings:0')
                    user_feature_bias = pretrain_graph.get_tensor_by_name('embedding/embedding/user_feature_bias:0')
                else:
                    user_feature_embeddings = tf.zeros([0, self.hidden_factor])
                    user_feature_bias = tf.zeros([0, 1])

                with tf.Session() as sess:
                    weight_saver.restore(sess, self.read_file)
                    ue, ufe, ie, ife, cfe, ub, ufb, ib, ifb, cfb, gb,att1,bias2,att2,att3 = sess.run(
                        [user_embeddings,
                         user_feature_embeddings,
                         item_embeddings,
                         item_feature_embeddings,
                         context_feature_embeddings,
                         user_bias,
                         user_feature_bias,
                         item_bias,
                         item_feature_bias,
                         context_feature_bias,
                         global_bias,
                         att_embed_1,
                         att_bias_2,
                         att_embed_2,
                         att_embed_3])
                self.user_embeddings = tf.Variable(ue, dtype=tf.float32, name='user_embeddings')
                self.user_feature_embeddings = tf.Variable(ufe, dtype=tf.float32, name='user_feature_embeddings')
                self.item_embeddings = tf.Variable(ie, dtype=tf.float32, name='item_embeddings')
                self.item_feature_embeddings = tf.Variable(ife, dtype=tf.float32, name='item_feature_embeddings')
                self.context_feature_embeddings = tf.Variable(cfe, dtype=tf.float32, name='context_feature_embeddings')
                self.user_bias = tf.Variable(ub, dtype=tf.float32, name='user_bias')
                self.user_feature_bias = tf.Variable(ufb, dtype=tf.float32, name='user_feature_bias')
                self.item_bias = tf.Variable(ib, dtype=tf.float32, name='item_bias')
                self.item_feature_bias = tf.Variable(ifb, dtype=tf.float32, name='item_feature_bias')
                self.context_feature_bias = tf.Variable(cfb, dtype=tf.float32, name='context_feature_bias')
                self.global_bias = tf.Variable(gb, dtype=tf.float32, name='global_bias')
                self.att_embed_1 = tf.Variable(att1,dtype=tf.float32, name='att_embed_1')
                self.att_bias_2 = tf.Variable(bias2,dtype=tf.float32, name='att_bias_2')
                self.att_embed_2 = tf.Variable(att2,dtype=tf.float32, name='att_embed_2')
                self.att_embed_3 = tf.Variable(att3,dtype=tf.float32, name='att_embed_3')
                print("restore network done")
                print("===========================")

            else:
                initializer = Tool.get_initializer(self.init_method, self.stddev)

                self.user_embeddings = tf.Variable(initializer([self.num_users, self.hidden_factor]),
                                                   name='user_embeddings', dtype=tf.float32)  # (users, embedding_size)
                self.user_feature_embeddings = tf.Variable(initializer([self.num_user_features, self.hidden_factor]),
                                                           name='user_feature_embeddings', dtype=tf.float32)
                self.item_embeddings = tf.Variable(initializer([self.num_items, self.hidden_factor]),
                                                   name='item_embeddings', dtype=tf.float32)  # (items, embedding_size)
                self.item_feature_embeddings = tf.Variable(initializer([self.num_item_features, self.hidden_factor]),
                                                           name='item_feature_embeddings', dtype=tf.float32)
                self.context_feature_embeddings = tf.Variable(initializer([self.num_context_features - self.num_items,
                                                                           self.hidden_factor]),
                                                              name='context_feature_embeddings', dtype=tf.float32)
                self.user_bias = tf.Variable(initializer([self.num_users, 1]),
                                             name='user_bias', dtype=tf.float32)  # (users, embedding_size)
                self.user_feature_bias = tf.Variable(initializer([self.num_user_features, 1]),
                                                     name='user_feature_bias', dtype=tf.float32)
                self.item_bias = tf.Variable(initializer([self.num_items, 1]),
                                             name='item_bias', dtype=tf.float32)  # (items, embedding_size)
                self.item_feature_bias = tf.Variable(initializer([self.num_item_features, 1]),
                                                     name='item_feature_bias', dtype=tf.float32)
                self.context_feature_bias = tf.Variable(initializer([self.num_context_features - self.num_items, 1]),
                                                        name='context_feature_bias', dtype=tf.float32)
                self.global_bias = tf.Variable(tf.zeros([1, 1]),
                                               dtype=tf.float32,
                                               name='global_bias')
                # ==================================================================== #
                self.att_embed_1 = tf.Variable(initializer([self.hidden_factor, self.hidden_factor]),
                                               name='att_embed_1', dtype=tf.float32)
                self.att_embed_2 = tf.Variable(initializer([self.hidden_factor, self.hidden_factor]),
                                               name='att_embed_2', dtype=tf.float32)
                self.att_bias_2 = tf.Variable(initializer([1, self.hidden_factor]), name='att_bias_2',
                                              dtype=tf.float32)
                self.att_embed_3 = tf.Variable(initializer([self.hidden_factor, 1]),
                                               name='att_embed_3', dtype=tf.float32)
                # ==================================================================== #
            self.valid_item_embeddings = tf.gather(self.item_embeddings, list(range(self.num_valid_items)))
            if self.adj_norm_type in ['rs', 'rd', 'db']:
                self.norm_user_nn = tf.constant(self.norm_user_neighbor_num, dtype=tf.float32,
                                                name='norm_user_neighbor_number')
                self.norm_item_nn = tf.constant(self.norm_item_neighbor_num, dtype=tf.float32,
                                                name='norm_item_neighbor_number')

            self.all_init_embedding = tf.concat([self.user_embeddings,
                                                 self.user_feature_embeddings,
                                                 self.item_embeddings,
                                                 self.item_feature_embeddings,
                                                 self.context_feature_embeddings],
                                                axis=0,
                                                name='all_init_embeddings')
            self.all_init_bias = tf.concat([self.user_bias,
                                            self.user_feature_bias,
                                            self.item_bias,
                                            self.item_feature_bias,
                                            self.context_feature_bias,
                                            self.global_bias],
                                           axis=0,
                                           name='all_init_bias')

    def _create_inference(self):
        with tf.name_scope("inference"):
            ######################## Encoder ########################
            # user feature
            all_user_embedding = tf.expand_dims(self.user_embeddings, 1)
            if self.user_feature_mat is not None:
                user_feature_embedding = tf.nn.embedding_lookup(self.user_feature_embeddings,
                                                                self.user_feature_mat)  # [num_users, 2, h]
                all_user_embedding = tf.concat([all_user_embedding,
                                                user_feature_embedding], axis=1)  # [num_users, 3, h]

            # item feature
            item_feature_embedding = tf.nn.embedding_lookup(self.item_feature_embeddings,
                                                            self.item_feature_mat)  # [num_valid_items, 3, h]
            all_item_feature_embedding = tf.concat([tf.expand_dims(self.valid_item_embeddings, 1),
                                                    item_feature_embedding], axis=1)  # [num_valid_items, 4, h]
            if self.num_gcn_layers == 0:
                self.encoded_user_embedding = tf.reduce_sum(all_user_embedding, axis=1)
                self.encoded_item_embedding = tf.reduce_sum(all_item_feature_embedding, axis=1)
            else:
                self.encoded_user_embedding = tf.reduce_mean(all_user_embedding, axis=1)
                self.encoded_item_embedding = tf.reduce_mean(all_item_feature_embedding, axis=1)

            # context
            self.context_feature_embedding_wo_last = tf.nn.embedding_lookup(self.context_feature_embeddings,
                                                                            self.context_feature_wo_last)  # [num_contexts, 4, h]
            self.context_feature_embedding_last = tf.nn.embedding_lookup(self.item_embeddings,
                                                                         self.context_feature_last)  # [num_contexts, h]
            self.context_feature_embedding = tf.concat([self.context_feature_embedding_wo_last,
                                                        tf.expand_dims(self.context_feature_embedding_last, 1)],
                                                       axis=1)  # [num_contexts, 5, h]
            self.context_feature_bias_wo_last = tf.nn.embedding_lookup(self.context_feature_bias,
                                                                       self.context_feature_wo_last)  # [num_contexts, 4, 1]
            self.context_feature_bias_last = tf.nn.embedding_lookup(self.item_bias,
                                                                    self.context_feature_last)  # [num_contexts, 1]
            self.context_feature_bias = tf.concat([self.context_feature_bias_wo_last,
                                                   tf.expand_dims(self.context_feature_bias_last, 1)],
                                                  axis=1)  # [num_contexts, 5, 1]
            self.encoded_context_embedding = tf.reduce_mean(self.context_feature_embedding, axis=1)  # [num_contexts, h]

            # event
            user_embedding = tf.nn.embedding_lookup(self.encoded_user_embedding, self.insts2userid)  # [num_insts, h]
            context_embedding = tf.nn.embedding_lookup(self.context_feature_embedding,
                                                       self.insts2contextid)  # [num_insts, 5, h]

            # ==================================================================== #
            attention = []
            for i in range(5):
                temp = tf.matmul(
                    tf.nn.relu(
                        tf.matmul(
                            user_embedding, self.att_embed_1) + (
                                tf.matmul(context_embedding[:, i, :], self.att_embed_2) + self.att_bias_2)
                    ), self.att_embed_3)  # [num_insts, 1]
                attention.append(temp)
            attention = tf.squeeze(tf.stack(attention, 1), 2)  # [num_insts, 5]

            attention = tf.nn.softmax(attention, 1)  # [num_insts, 5]

            ######################## GNN layers ########################
            layer_user_embedding = self.encoded_user_embedding
            layer_item_embedding = self.encoded_item_embedding
            layer_context_embedding = self.context_feature_embeddings
            all_user_embeddings = [layer_user_embedding * self.gcn_layer_weight[0]]
            all_item_embeddings = [layer_item_embedding * self.gcn_layer_weight[0]]
            all_context_embeddings = [layer_context_embedding * self.gcn_layer_weight[0]]

            for k in range(1, self.num_gcn_layers + 1):
                if self.adj_norm_type in ['rs', 'rd', 'db']:
                    layer_user_embedding = tf.multiply(layer_user_embedding, tf.expand_dims(self.norm_user_nn, 1))
                    layer_item_embedding = tf.multiply(layer_item_embedding, tf.expand_dims(self.norm_item_nn, 1))

                insts_user_embedding = tf.nn.embedding_lookup(layer_user_embedding, self.insts2userid)  # [num_insts, h]
                insts_item_embedding = tf.nn.embedding_lookup(layer_item_embedding, self.insts2itemid)  # [num_insts, h]

                # ==================================================================== #
                insts_context_embedding_wo_last = tf.nn.embedding_lookup(
                    layer_context_embedding, self.context_feature_wo_last[self.insts2contextid])  # [num_insts, 4, h]
                insts_context_embedding_last = tf.nn.embedding_lookup(
                    self.item_embeddings, self.context_feature_last[self.insts2contextid])  # [num_insts, h]
                insts_context_embedding = tf.concat([insts_context_embedding_wo_last,
                                                     tf.expand_dims(insts_context_embedding_last, 1)],
                                                    axis=1)  # [num_insts, 5, h]
                insts_event_embedding = tf.squeeze(
                    tf.matmul(tf.expand_dims(attention, 1), insts_context_embedding), 1)  # [num_insts, h]
                # ==================================================================== #
                # 
                if k == 1:
                    self.train_event_embeddings = insts_event_embedding

                if self.merge_type == 'sum':
                    insts_user_embedding_new = tf.add(insts_item_embedding, insts_event_embedding)
                    insts_item_embedding_new = tf.add(insts_user_embedding, insts_event_embedding)
                    temp = tf.add(insts_item_embedding, insts_user_embedding)
                    insts_context_embedding_new = tf.reshape(
                        attention[:, 0:4], [temp.shape[0] * 4, 1]) * tf.reshape(
                        tf.tile(temp, [1, 4]), [temp.shape[0] * 4, temp.shape[1]])
                elif self.merge_type == 'ip':
                    norm_insts_context_embedding = tf.nn.l2_normalize(insts_context_embedding, axis=1,
                                                                      name='normalize_context')
                    insts_user_embedding_new = tf.multiply(insts_item_embedding, norm_insts_context_embedding)
                    insts_item_embedding_new = tf.multiply(insts_user_embedding, norm_insts_context_embedding)
                elif self.merge_type == 'mlp':
                    insts_user_embedding_new = tf.layers.dense(
                        tf.concat([insts_item_embedding, insts_context_embedding], axis=1),
                        self.hidden_factor,
                        activation=tf.nn.leaky_relu,
                        use_bias=True,
                        reuse=tf.AUTO_REUSE,
                        name='gc_user_l%d' % k
                    )
                    insts_item_embedding_new = tf.layers.dense(
                        tf.concat([insts_user_embedding, insts_context_embedding], axis=1),
                        self.hidden_factor,
                        activation=tf.nn.leaky_relu,
                        use_bias=True,
                        reuse=tf.AUTO_REUSE,
                        name='gc_item_l%d' % k
                    )
                else:
                    raise ValueError("Invalid merge_type!")

                if self.adj_norm_type in ['ls', 'db']:
                    layer_user_embedding = tf.math.unsorted_segment_sqrt_n(
                        insts_user_embedding_new,
                        self.insts2userid,
                        self.num_users,
                        name='aggregate_user_l%d' % k)  # [num_users, h]
                    layer_item_embedding = tf.math.unsorted_segment_sqrt_n(
                        insts_item_embedding_new,
                        self.insts2itemid,
                        self.num_valid_items,
                        name='aggregate_item_l%d' % k)  # [num_valid_items, h]
                    if self.train2cg is not None:
                        insts_context_embedding_new = tf.gather(insts_context_embedding_new, self.train2cg_idx)
                        layer_context_embedding = tf.math.unsorted_segment_sqrt_n(
                            insts_context_embedding_new,
                            self.train2cg,
                            self.num_context_features - self.num_items,
                            name='aggregate_context_l%d' % k)
                    else:
                        layer_context_embedding = tf.math.unsorted_segment_sqrt_n(
                            insts_context_embedding_new,
                            tf.reshape(self.context_feature_wo_last[self.insts2contextid], [-1]),
                            self.num_context_features - self.num_items,
                            name='aggregate_context_l%d' % k)
                elif self.adj_norm_type in ['rs', 'rd']:
                    layer_user_embedding = tf.math.unsorted_segment_sum(
                        insts_user_embedding_new,
                        self.insts2userid,
                        self.num_users,
                        name='aggregate_user_l%d' % k)  # [num_users, h]
                    layer_item_embedding = tf.math.unsorted_segment_sum(
                        insts_item_embedding_new,
                        self.insts2itemid,
                        self.num_valid_items,
                        name='aggregate_item_l%d' % k)  # [num_valid_items, h]
                elif self.adj_norm_type == 'ld':
                    layer_user_embedding = tf.math.unsorted_segment_mean(
                        insts_user_embedding_new,
                        self.insts2userid,
                        self.num_users,
                        name='aggregate_user_l%d' % k)  # [num_users, h]
                    layer_item_embedding = tf.math.unsorted_segment_mean(
                        insts_item_embedding_new,
                        self.insts2itemid,
                        self.num_valid_items,
                        name='aggregate_item_l%d' % k)  # [num_valid_items, h]

                all_user_embeddings += [layer_user_embedding * self.gcn_layer_weight[k]]
                all_item_embeddings += [layer_item_embedding * self.gcn_layer_weight[k]]
                all_context_embeddings += [layer_context_embedding * self.gcn_layer_weight[k]]

            all_user_embeddings = tf.stack(all_user_embeddings, 1)
            self.u_g_embeddings = tf.reduce_sum(all_user_embeddings, axis=1, keepdims=False,
                                                name='updated_u_embedding')  # [num_users, h]
            all_item_embeddings = tf.stack(all_item_embeddings, 1)
            self.i_g_embeddings = tf.reduce_sum(all_item_embeddings, axis=1, keepdims=False,
                                                name='updated_item_embedding')  # [num_valid_items, h]
            all_context_embeddings = tf.stack(all_context_embeddings, 1)
            self.context_g_embeddings = tf.reduce_sum(all_context_embeddings, axis=1, keepdims=False,
                                                      name='updated_context_embedding')
            ######################## Decoder ########################
            self.updated_user_embeddings = tf.Variable(tf.zeros_like(self.user_embeddings),
                                                       name='updated_user_embeddings',
                                                       dtype=tf.float32)  # (users, embedding_size)
            self.updated_item_embeddings = tf.Variable(tf.zeros_like(self.valid_item_embeddings),
                                                       name='updated_item_embeddings', dtype=tf.float32)
            self.updated_context_embeddings = tf.Variable(tf.zeros_like(self.context_feature_embeddings),
                                                          name='updated_context_embeddings', dtype=tf.float32)
            self.update_user_assign = self.updated_user_embeddings.assign(self.u_g_embeddings)
            self.update_item_assign = self.updated_item_embeddings.assign(self.i_g_embeddings)
            self.update_context_assign = self.updated_context_embeddings.assign(self.context_g_embeddings)
            self.context_feature_embedding_wo_last = tf.nn.embedding_lookup(self.context_g_embeddings,
                                                                            self.context_feature_wo_last)  # [num_contexts, 4, h]
            self.context_feature_embedding_last = tf.nn.embedding_lookup(self.item_embeddings,
                                                                         self.context_feature_last)  # [num_contexts, h]
            self.context_feature_embedding = tf.concat([self.context_feature_embedding_wo_last,
                                                        tf.expand_dims(self.context_feature_embedding_last, 1)],
                                                       axis=1)  # [num_contexts, 5, h]
            if self.decoder_type == 'FM':
                # user id
                batch_user_embedding = tf.nn.embedding_lookup(self.u_g_embeddings, self.user_input)  # [batch_size, h]
                batch_user_bias = tf.nn.embedding_lookup(self.user_bias, self.user_input)  # [batch_size, 1]
                # context id
                batch_context_feature_embedding = tf.nn.embedding_lookup(self.context_feature_embedding,
                                                                         self.context_input)  # [batch_size, 5, h]
                batch_context_feature_bias = tf.nn.embedding_lookup(self.context_feature_bias,
                                                                    self.context_input)  # [batch_size, 5, 1]
                # positive item id
                batch_item_embedding = tf.nn.embedding_lookup(self.i_g_embeddings, self.item_input)  # [batch_size, h]
                batch_item_bias = tf.nn.embedding_lookup(self.item_bias, self.item_input)  # [batch_size, 1]
                # postive part
                batch_embedding = tf.concat([tf.expand_dims(batch_user_embedding, 1),
                                             tf.expand_dims(batch_item_embedding, 1),
                                             batch_context_feature_embedding], axis=1)  # [batch_size, 7, h]
                batch_bias = tf.concat([tf.expand_dims(batch_user_bias, 1),
                                        tf.expand_dims(batch_item_bias, 1),
                                        batch_context_feature_bias], axis=1)  # [batch_size, 7, 1]
                square_of_sum = tf.square(tf.reduce_sum(batch_embedding, 1))  # [batch_size, h]
                sum_of_square = tf.reduce_sum(tf.square(batch_embedding), 1)  # [batch_size, h]
                bi_linear = 0.5 * tf.subtract(square_of_sum, sum_of_square)  # [batch_size, h]
                bi_linear = tf.reduce_sum(bi_linear, 1, keepdims=True)  # [batch_size, 1]
                bias = tf.reduce_sum(batch_bias, 1, keep_dims=False)  # [batch_size, 1]
                self.output = tf.add(bi_linear, bias)
                self.output = tf.add(self.output, self.global_bias)

            elif self.decoder_type == 'FM-pooling':
                # user id
                batch_user_embedding = tf.nn.embedding_lookup(self.u_g_embeddings, self.user_input)  # [batch_size, h]
                batch_user_bias = tf.nn.embedding_lookup(self.user_bias, self.user_input)  # [batch_size, 1]
                # context id
                batch_context_feature_embedding = tf.nn.embedding_lookup(self.context_feature_embedding,
                                                                         self.context_input)  # [batch_size, 5, h]
                batch_context_feature_embedding = tf.nn.reduce_mean(batch_context_feature_embedding, axis=1,
                                                                    keepdims=True)  # [batch_size, 1, h]
                batch_context_feature_bias = tf.nn.embedding_lookup(self.context_feature_bias,
                                                                    self.context_input)  # [batch_size, 5, 1]
                batch_context_feature_bias = tf.nn.reduce_mean(batch_context_feature_bias, axis=1, keepdims=True)
                # positive item id
                batch_item_embedding = tf.nn.embedding_lookup(self.i_g_embeddings, self.item_input)  # [batch_size, h]
                batch_item_bias = tf.nn.embedding_lookup(self.item_bias, self.item_input)  # [batch_size, 1]
                # postive part
                batch_embedding = tf.concat([tf.expand_dims(batch_user_embedding, 1),
                                             tf.expand_dims(batch_item_embedding, 1),
                                             batch_context_feature_embedding], axis=1)  # [batch_size, 7, h]
                batch_bias = tf.concat([tf.expand_dims(batch_user_bias, 1),
                                        tf.expand_dims(batch_item_bias, 1),
                                        batch_context_feature_bias], axis=1)  # [batch_size, 7, 1]
                square_of_sum = tf.square(tf.reduce_sum(batch_embedding, 1))  # [batch_size, h]
                sum_of_square = tf.reduce_sum(tf.square(batch_embedding), 1)  # [batch_size, h]
                bi_linear = 0.5 * tf.subtract(square_of_sum, sum_of_square)  # [batch_size, h]
                bi_linear = tf.reduce_sum(bi_linear, 1, keepdims=True)  # [batch_size, 1]
                bias = tf.reduce_sum(batch_bias, 1, keep_dims=False)  # [batch_size, 1]
                self.output = tf.add(bi_linear, bias)
                self.output = tf.add(self.output, self.global_bias)

            elif self.decoder_type == 'MLP':
                # user id
                batch_user_embedding = tf.nn.embedding_lookup(self.u_g_embeddings, self.user_input)  # [batch_size, h]
                # context id
                batch_context_feature_embedding = tf.nn.embedding_lookup(self.context_feature_embedding,
                                                                         self.context_input)  # [batch_size, 5, h]
                # positive item id
                batch_item_embedding = tf.nn.embedding_lookup(self.i_g_embeddings, self.item_input)  # [batch_size, h]
                batch_embedding = tf.concat([tf.expand_dims(batch_user_embedding, 1),
                                             tf.expand_dims(batch_item_embedding, 1),
                                             batch_context_feature_embedding], axis=1)  # [batch_size, 7, h]
                batch_embedding = tf.reshape(batch_embedding, [-1, self.hidden_factor * (self.num_context_fields + 2)])
                for k in range(1, self.num_hidden_layers + 1):
                    batch_embedding = tf.layers.dense(batch_embedding,
                                                      self.hidden_factor,
                                                      activation=tf.nn.leaky_relu,
                                                      use_bias=True,
                                                      reuse=tf.AUTO_REUSE,
                                                      name='predict_hidden_l%d' % k)
                self.output = tf.layers.dense(batch_embedding,
                                              1,
                                              use_bias=False,
                                              reuse=tf.AUTO_REUSE,
                                              name='predictor')

            elif self.decoder_type == 'IP':
                # user id
                batch_user_embedding = tf.nn.embedding_lookup(self.u_g_embeddings, self.user_input)  # [batch_size, h]
                # positive item id
                batch_item_embedding = tf.nn.embedding_lookup(self.i_g_embeddings, self.item_input)  # [batch_size, h]
                self.output = tf.reduce_sum(tf.multiply(batch_user_embedding, batch_item_embedding), 1, keepdims=True)

    def _create_loss(self):
        with tf.name_scope("loss"):
            self.log_loss = tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=tf.squeeze(self.output)))

            u_embeddings_pre = tf.nn.embedding_lookup(self.user_embeddings, self.user_input)
            i_embeddings_pre = tf.nn.embedding_lookup(self.item_embeddings, self.item_input)
            feature_embeddings_pre = tf.concat([self.user_feature_embeddings,
                                                self.item_feature_embeddings,
                                                self.context_feature_embeddings], 0)
            self.emb_loss = self.reg * Tool.l2_loss(u_embeddings_pre, i_embeddings_pre, feature_embeddings_pre)
            # self.emb_loss = self.regs[0] * Tool.l2_loss(self.all_init_embedding, self.all_init_bias)
            self.loss = self.log_loss + self.emb_loss

    def _create_optimizer(self):
        with tf.name_scope("learner"):
            self.optimizer = Learner.optimizer(self.optimizer_type, self.loss, self.learning_rate)

    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_inference()
        self._create_loss()
        self._create_optimizer()
        self.saver = tf.train.Saver()

    # ---------- training process -------
    def train_model(self):
        # FM_user_embedding, FM_item_embedding = self.sess.run([self.user_embeddings, self.item_embeddings])
        # np.savez('./visual/FM/results.npz', user_embed=FM_user_embedding, item_embed=FM_item_embedding)
        logger.info(self.evaluator.metrics_info())
        buf, flag = self.evaluate()
        logger.info("epoch 0:\t%s" % buf)
        user_input_val, context_input_val, item_input_val, labels_val = DataGenerator._get_pointwise_all_data_context(
            self.dataset, self.num_negatives, phase='valid')
        data_iter_val = DataIterator(user_input_val, context_input_val, item_input_val, labels_val,
                                     batch_size=self.test_batch_size, shuffle=False)
        stopping_step = 0
        for epoch in range(1, self.num_epochs + 1):
            total_loss, total_emb_loss = 0.0, 0.0
            training_start_time = time()

            # Generate training instances
            if self.loss_type == 'bpr_loss':
                user_input, context_input, item_input_pos, item_input_neg = DataGenerator._get_pairwise_all_data_context(
                    self.dataset)
                data_iter = DataIterator(user_input, context_input, item_input_pos, item_input_neg,
                                         batch_size=self.batch_size, shuffle=True)

                time1 = time()
                for bat_users, bat_context, bat_items_pos, bat_items_neg in data_iter:
                    feed_dict = {self.user_input: bat_users,
                                 self.context_input: bat_context,
                                 self.item_input: bat_items_pos,
                                 self.item_input_neg: bat_items_neg}
                    loss, _ = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
                    total_loss += loss
            else:
                user_input, context_input, item_input, labels = DataGenerator._get_pointwise_all_data_context(
                    self.dataset, self.num_negatives)
                data_iter = DataIterator(user_input, context_input, item_input, labels,
                                         batch_size=self.batch_size, shuffle=True)
                time1 = time()
                for bat_users, bat_context, bat_items, bat_labels in data_iter:
                    feed_dict = {self.user_input: bat_users,
                                 self.context_input: bat_context,
                                 self.item_input: bat_items,
                                 self.labels: bat_labels}
                    loss, emb_loss, _ = self.sess.run((self.loss, self.emb_loss, self.optimizer), feed_dict=feed_dict)
                    total_loss += loss
                    total_emb_loss += emb_loss
            logger.info("[iter %d : loss: %f = %f + %f, time: %.1f = %.1f + %.1f]" % ( \
                epoch,
                total_loss / len(user_input),
                (total_loss - total_emb_loss) / len(user_input),
                total_emb_loss / len(user_input),
                time() - training_start_time,
                time1 - training_start_time,
                time() - time1))
            # total_loss_val, total_emb_loss_val = 0.0, 0.0
            # for bat_users, bat_context, bat_items, bat_labels in data_iter_val:
            #     feed_dict = {self.user_input: bat_users,
            #                  self.context_input: bat_context,
            #                  self.item_input: bat_items,
            #                  self.labels: bat_labels}
            #     loss, emb_loss = self.sess.run((self.loss, self.emb_loss), feed_dict=feed_dict)
            #     total_loss_val += loss
            #     total_emb_loss_val += emb_loss
            # logger.info("[Validation loss @ %d: %.4f = %.4f + %.4f]" % (epoch,
            #                                                             total_loss_val / len(user_input_val),
            #                                                             (total_loss_val - total_emb_loss_val) / len(
            #                                                                 user_input_val),
            #                                                             total_emb_loss_val / len(user_input_val)))
            if epoch % args.test_interval == 0:
                # GCM_user_embedding, GCM_item_embedding = self.sess.run([self.u_g_embeddings, self.i_g_embeddings])
                # np.savez('./visual/GCM/results.npz', user_embed=GCM_user_embedding, item_embed=GCM_item_embedding)
                buf, flag = self.evaluate()
                if flag:
                    self.best_epoch = epoch
                    stopping_step = 0
                    logger.info("Find a better model.")
                    if self.save_flag > 0:
                        logger.info("Save model to file as pretrain.")
                        self.saver.save(self.sess, self.save_file)
                else:
                    stopping_step += 1
                    if stopping_step >= args.stop_cnt:
                        logger.info("Early stopping is trigger at epoch: {}".format(epoch))
                        break
                logger.info("epoch %d:\t%s" % (epoch, buf))

        buf = '\t'.join([("%.4f" % x).ljust(12) for x in self.best_result])
        logger.info("best_result@epoch %d:\n" % self.best_epoch + buf)

        # params = self.sess.run([self.user_embeddings, self.item_embeddings])
        # with open("pretrained/%s_epochs=%d_embedding=%d_MF.pkl" % (self.dataset.dataset_name, self.num_epochs,self.embedding_size), "wb") as fout:
        #         pickle.dump(params, fout)

    @timer
    def evaluate(self):
        _ = self.sess.run((self.update_user_assign, self.update_item_assign, self.update_context_assign))
        flag = False
        current_result, buf = self.evaluator.evaluate4CARS(self)
        if self.best_result[0] + self.best_result[2] < current_result[0] + current_result[2]:
            self.best_result = current_result
            flag = True
        return buf, flag

    def predict(self, user_ids, context_ids):
        batch_size = len(user_ids)
        user_inputs = np.tile(np.array(user_ids).reshape(-1, 1), [1, self.num_valid_items]).reshape(-1)
        context_inputs = np.tile(np.array(context_ids).reshape(-1, 1), [1, self.num_valid_items]).reshape(-1)
        item_inputs = np.tile(np.array(list(range(self.num_valid_items))).reshape(1, -1), [batch_size, 1]).reshape(-1)
        feed_dict = {self.user_input: user_inputs,
                     self.context_input: context_inputs,
                     self.item_input: item_inputs}
        ratings = self.sess.run(self.output, feed_dict=feed_dict)
        ratings = np.reshape(ratings, [batch_size, self.num_valid_items])
        return ratings

    def gen_data_for_grouping(self):
        grouping_train = self.sess.run(self.train_event_embeddings)
        context_ids = self.context_feature_wo_last[self.insts2contextid]
        cg_insts = []
        id2insts = []
        grouping_means = []
        c_values = []
        maxid = max(set(np.reshape(context_ids, -1).tolist()))+1
        for i in range(4):
            print(set(context_ids[:, i].tolist()))
            x, y, z, c_value = [], [], [], []
            for j in set(context_ids[:, i].tolist()):
                indexes = np.argwhere(context_ids[:, i] == j).reshape(-1)
                id2inst = dict(zip(range(len(indexes)), indexes))
                cg_inst = grouping_train[indexes]
                cg_means = np.mean(grouping_train[indexes], axis=0).reshape((1, -1))
                x.append(cg_inst)
                y.append(id2inst)
                z.append(cg_means)
                c_value.append(j)
                # print(id2inst)
            cg_insts.append(x)
            id2insts.append(y)
            grouping_means.append(z)
            c_values.append(c_value)
        return cg_insts, grouping_means, id2insts,c_values, grouping_train.shape[0], maxid, context_ids


def find_nearest_neighbors_by_rate(x, rate, queries=None, k=5, gpu_id=None):
    """
    Find k nearest neighbors for each of the n examples.
    Distances are computed using Squared Euclidean distance metric.
    Arguments:
    ----------
    queries
    x (ndarray): N examples to search within. [N x d].
    gpu_id (int): use CPU if None else use GPU with the specified id.
    queries (ndarray): find nearest neigbor for each query example. [M x d] matrix
        If None than find k nearest neighbors for each row of x
        (excluding self exampels).
    k (int): number of nearest neighbors to find.
    Return
    I (ndarray): Indices of the nearest neighnpors. [M x k]
    distances (ndarray): Distances to the nearest neighbors. [M x k]
    """
    if gpu_id is not None and not isinstance(gpu_id, int):
        raise ValueError('gpu_id must be None or int')
    x = np.asarray(x.reshape(x.shape[0], -1), dtype=np.float32)
    remove_self = False  # will have queries in the search results?
    if queries is None:
        remove_self = True
        queries = x
        k += 1

    d = x.shape[1]
    if x.shape[0] < 2048:
        k = x.shape[0]
    else:
        k = int(x.shape[0] * rate)
    # if k < 2048:
    #     k = x.shape[0]
    print('[INFO]FAISS: cpu::find {} nearest neighbors ...... total num :: {}'.format(k - int(remove_self), x.shape[0]))
    index = faiss.IndexFlatL2(d)
    index.add(x)
    #
    distances, nns = index.search(queries, k)
    if remove_self:
        for i in range(len(nns)):
            indices = np.nonzero(nns[i, :] != i)[0]
            indices.sort()
            if len(indices) > k - 1:
                indices = indices[:-1]
            nns[i, :-1] = nns[i, indices]
            distances[i, :-1] = distances[i, indices]
        nns = nns[:, :-1]
        distances = distances[:, :-1]
    return nns, distances


if __name__ == '__main__':
    # configurations
    model_name = 'PEG'
    save_name = 'PEG_ue_relu'
    show_name = 'PEG_ue_relu'
    args = configs.parse_args()
    args = configs.post_process_for_config(args, model_name)
    nowTime = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    model_str = '%s-l%d-%s-%s-%s-reg%.0e-%s' % (
        show_name,
        args.num_gcn_layers,
        args.merge_type,
        args.decoder_type,
        args.adj_norm_type,
        args.reg,
        nowTime
    )
    pretrain_model_str = '%s-l%d-%s-%s-%s-reg%.0e' % (
        save_name,
        args.pretrain_layer,
        args.merge_type,
        args.decoder_type,
        args.adj_norm_type,
        args.pretrain_reg,
    )
    print(pretrain_model_str)

    # Logging
    current_time = strftime("%Y%m%d%H%M%S", localtime())
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    # wirte into file
    ensureDir('%s/logs/%s-%s/' % (args.proj_path, model_name, args.dataset))
    fh = logging.FileHandler(
        '%s/logs/%s-%s/%s_%s.log' % (args.proj_path, model_name, args.dataset, model_str, current_time))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    # show on screen
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)
    # add two Hander
    logger.addHandler(fh)
    logger.addHandler(sh)

    # load data
    data = DATA(args, logger)

    if args.grouping:
        # 
        filenames = os.listdir(args.proj_path + 'pretrain/pretrain-PEG-%s/' % args.dataset)
        print(filenames)
        filter_names = []
        for name in filenames:
            if pretrain_model_str in name:
                filter_names.append(name)
        pretrain_model_str_real = filter_names[0]
        ensureDir(args.proj_path + 'pretrain/pretrain-PEG-%s/%s/' % (args.dataset, pretrain_model_str_real))
        args.read_file = args.proj_path + 'pretrain/pretrain-PEG-%s/%s/model' % (args.dataset,
                                                                                 pretrain_model_str_real)
        # ensureDir(pretrain_root)
        # args.read_file ='%s/model' % pretrain_root
        print(args.read_file)

        insts2contextid = data.all_data_dict['train_data']['context_id'].to_list()
        insts2userid = data.all_data_dict['train_data']['user_id'].to_list()
        test2userid = data.all_data_dict['test_data']['user_id'].to_list()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            # model
            model = PEG(sess, data, args)
            model.build_graph()
            sess.run(tf.global_variables_initializer())
            # Training
            logger.info(
                '########################### begin grouping ###########################'
            )

            cg_insts, grouping_means, id2insts, c_values, num_insts, maxid, context_ids = model.gen_data_for_grouping()
            insts2newcontextids = []
            print(maxid)
            print("=====")
            rate_str = str(args.keep_rate).replace(".", "")
            for i in range(len(cg_insts)):
                insts2newcontextid = np.array([maxid for _ in range(num_insts)])
                for j in range(len(cg_insts[i])):
                    nns, centroids = find_nearest_neighbors_by_rate(cg_insts[i][j], rate=args.keep_rate,
                                                                    queries=grouping_means[i][j], gpu_id=0)
                    # print(nns)
                    # print(nns[0])
                    for value in nns[0]:
                        if value != -1:
                            insts2newcontextid[id2insts[i][j][value]] = c_values[i][j]
                # insts2newcontextids.extend(insts2newcontextid)
                insts2newcontextid = np.reshape(insts2newcontextid, [-1, 1])
                if i == 0:
                    insts2newcontextids = insts2newcontextid
                else:
                    insts2newcontextids = np.concatenate([insts2newcontextids, insts2newcontextid], axis=1)

            insts2newcontextids = np.reshape(insts2newcontextids, -1)
            train2cg_idx = np.argwhere(insts2newcontextids != maxid).reshape(-1)
            insts2newcontextids = insts2newcontextids[insts2newcontextids != maxid].reshape(-1)

            with open(args.proj_path + 'pretrain/pretrain-PEG-%s/%s/' % (args.dataset, pretrain_model_str_real)
                      + 'train2cg_'+rate_str+'.pkl', "wb") as fout:
                pickle.dump(insts2newcontextids, fout)
            with open(args.proj_path + 'pretrain/pretrain-PEG-%s/%s/' % (args.dataset, pretrain_model_str_real)
                      + 'train2cg_'+rate_str+'_idx.pkl', "wb") as fout:
                pickle.dump(train2cg_idx, fout)

            logger.info(
                '########################### end grouping ###########################')

    else:
        Hyperparameters = ['model_name', 'dataset', 'epoch', 'batch_size', 'test_batch_size', 'test_interval',
                           'stop_cnt', 'decoder_type', 'hidden_factor', 'loss_type', 'lr', 'optimizer', 'pretrain',
                           'save_flag', 'reg', 'init_method', 'stddev', 'topk', 'num_gcn_layers', 'gcn_layer_weight',
                           'merge_type', 'adj_norm_type']
        if args.loss_type == 'log_loss':
            Hyperparameters.append('num_negatives')
        if args.decoder_type == 'MLP':
            Hyperparameters.append('num_hidden_layers')

        hyper_info = '\n'.join(["{}={}".format(arg, value) for arg, value in vars(args).items() if arg in
                                Hyperparameters])
        logger.info('HyperParamters:\n' + hyper_info)
        if args.pretrain:
            filenames = os.listdir(args.proj_path + 'pretrain/pretrain-PEG-%s/' % args.dataset)
            filter_names = []
            for name in filenames:
                if pretrain_model_str in name:
                    filter_names.append(name)
            pretrain_model_str_real = filter_names[0]
            ensureDir(args.proj_path + 'pretrain/pretrain-PEG-%s/%s/' % (args.dataset, pretrain_model_str_real))
            args.read_file = args.proj_path + 'pretrain/pretrain-PEG-%s/%s/model' % (args.dataset,
                                                                                     pretrain_model_str_real)
        if args.save_flag:
            ensureDir(args.proj_path + 'pretrain/pretrain-PEG-%s/%s/' % (args.dataset, model_str))
            args.save_file = args.proj_path + 'pretrain/pretrain-PEG-%s/%s/model' % (args.dataset, model_str)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            # Training
            if args.finetune:
                filenames = os.listdir(args.proj_path + 'pretrain/pretrain-PEG-%s/' % args.dataset)
                filter_names = []
                for name in filenames:
                    if pretrain_model_str in name:
                        filter_names.append(name)
                pretrain_model_str_real = filter_names[0]
                ensureDir(args.proj_path + 'pretrain/pretrain-PEG-%s/%s/' % (args.dataset, pretrain_model_str_real))
                args.read_file = args.proj_path + 'pretrain/pretrain-PEG-%s/%s/model' % (
                args.dataset, pretrain_model_str_real)

                rate_str = str(args.keep_rate).replace(".", "")
                with open(args.proj_path + 'pretrain/pretrain-PEG-%s/%s/' % (args.dataset, pretrain_model_str_real)
                          + 'train2cg_'+rate_str+'.pkl', 'rb') as f:
                    train2cg = pickle.load(f)
                with open(args.proj_path + 'pretrain/pretrain-PEG-%s/%s/' % (args.dataset, pretrain_model_str_real)
                          + 'train2cg_'+rate_str+'_idx.pkl', 'rb') as f:
                    train2cg_idx = pickle.load(f)
                model = PEG(sess, data, args, train2cg=train2cg, train2cg_idx=train2cg_idx)

                model.build_graph()
                sess.run(tf.global_variables_initializer())
                logger.info(
                    '########################### begin finetune ###########################'
                )
                model.train_model()
                logger.info(
                    '########################### end finetune ###########################')
            else:
                # model
                model = PEG(sess, data, args)
                model.build_graph()
                sess.run(tf.global_variables_initializer())

                logger.info(
                    '########################### begin training ###########################'
                )
                model.train_model()
                logger.info(
                    '########################### end training ###########################')
