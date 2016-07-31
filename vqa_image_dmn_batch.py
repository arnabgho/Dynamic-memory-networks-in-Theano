import random
import numpy as np

import theano
import theano.tensor as T
from theano.compile.nanguardmode import NanGuardMode

import lasagne
from lasagne import layers
from lasagne import nonlinearities
import cPickle as pickle

import utils
import nn_utils
import re
import copy
import json
import h5py
floatX = theano.config.floatX

class VQA_IMAGE_DMN_batch:

    def __init__(self,word2vec, word_vector_size, dim,
                mode, answer_module, input_mask_mode, memory_hops, batch_size, l2,
                normalize_attention, batch_norm, dropout,h5file,json_dict_file ,num_answers,img_vector_size,img_seq_len,
                img_h5file_train,img_h5file_test,**kwargs):

        print "==> not used params in DMN class:", kwargs.keys()

        self.vocab = {}
        self.ivocab = {}
        self.lr=0.001
        self.word2vec = word2vec
        self.word_vector_size = word_vector_size
        self.dim = dim
        self.mode = mode
        self.answer_module = answer_module
        self.input_mask_mode = input_mask_mode
        self.memory_hops = memory_hops
        self.batch_size = batch_size
        self.l2 = l2
        self.normalize_attention = normalize_attention
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.h5file=h5py.File(h5file,"r")
        self.img_h5file_train=h5py.File(img_h5file_train,"r")
        self.img_h5file_test=h5py.File(img_h5file_test,"r")
        self.img_seq_len=img_seq_len

        self.img_vector_size=img_vector_size
        with open (json_dict_file) as f:
            self.json_dict=json.load(f)

        #self.train_input, self.train_q, self.train_answer, self.train_fact_count, self.train_input_mask  = self._process_input(babi_train_raw)
        #self.test_input, self.test_q, self.test_answer, self.test_fact_count, self.test_input_mask  = self._process_input(babi_test_raw)
        #self.vocab_size = len(self.vocab)
        self.vocab_size=num_answers

        self.input_var = T.tensor3('input_var') # (batch_size, seq_len, glove_dim)
        self.img_input_var=T.tensor3('img_input_var') # (batch_size * img_seq_len , img_vector_size)
        self.q_var = T.tensor3('question_var') # as self.input_var
        self.answer_var = T.ivector('answer_var') # answer of example in minibatch
        self.fact_count_var = T.ivector('fact_count_var') # number of facts in the example of minibatch
        self.input_mask_var = T.imatrix('input_mask_var') # (batch_size, indices)

        print "==> building input module"
        self.W_inp_res_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.word_vector_size))
        self.W_inp_res_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_inp_res = nn_utils.constant_param(value=0.0, shape=(self.dim,))

        self.W_inp_upd_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.word_vector_size))
        self.W_inp_upd_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_inp_upd = nn_utils.constant_param(value=0.0, shape=(self.dim,))

        self.W_inp_hid_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.word_vector_size))
        self.W_inp_hid_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_inp_hid = nn_utils.constant_param(value=0.0, shape=(self.dim,))

        input_var_shuffled = self.input_var.dimshuffle(1, 2, 0)
        inp_dummy = theano.shared(np.zeros((self.dim, self.batch_size), dtype=floatX))
        inp_c_history, _ = theano.scan(fn=self.input_gru_step,
                            sequences=input_var_shuffled,
                            outputs_info=T.zeros_like(inp_dummy))

        inp_c_history_shuffled = inp_c_history.dimshuffle(2, 0, 1)

        inp_c_list = []
        inp_c_mask_list = []
        for batch_index in range(self.batch_size):
            taken = inp_c_history_shuffled[batch_index].take(self.input_mask_var[batch_index, :self.fact_count_var[batch_index]], axis=0)
            inp_c_list.append(T.concatenate([taken, T.zeros((self.input_mask_var.shape[1] - taken.shape[0], self.dim), floatX)]))
            inp_c_mask_list.append(T.concatenate([T.ones((taken.shape[0],), np.int32), T.zeros((self.input_mask_var.shape[1] - taken.shape[0],), np.int32)]))

        self.inp_c = T.stack(inp_c_list).dimshuffle(1, 2, 0)
        inp_c_mask = T.stack(inp_c_mask_list).dimshuffle(1, 0)

###################### Adding the Image Input Module

        print "==> building image img_input module"
        ### Don't Really Need the GRU to reduce the sentences into vectors ###
        self.img_input_var=T.reshape(self.img_input_var , ( self.batch_size * self.img_seq_len , self.img_vector_size ))

        img_input_layer=layers.InputLayer( shape=(self.batch_size*self.img_seq_len, self.img_vector_size), input_var=self.img_input_var)

        ## Convert the img_vector_size to self.dim using a MLP ##
        img_input_layer=layers.DenseLayer( img_input_layer , num_units=self.dim )

        img_input_var_dim=layers.get_output(img_input_layer)

        img_input_var_dim=T.reshape(img_input_var_dim ,(self.batch_size , self.img_seq_len , self.dim )  )

        #self.img_inp_c = T.stack(img_input_var_dim).dimshuffle(1, 2, 0)

        self.img_inp_c = img_input_var_dim.dimshuffle(1,2,0)
###################################################
        q_var_shuffled = self.q_var.dimshuffle(1, 2, 0)
        q_dummy = theano.shared(np.zeros((self.dim, self.batch_size), dtype=floatX))
        q_q_history, _ = theano.scan(fn=self.input_gru_step,
                            sequences=q_var_shuffled,
                            outputs_info=T.zeros_like(q_dummy))
        self.q_q = q_q_history[-1]


        print "==> creating parameters for memory module"
        self.W_mem_res_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.W_mem_res_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_mem_res = nn_utils.constant_param(value=0.0, shape=(self.dim,))

        self.W_mem_upd_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.W_mem_upd_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_mem_upd = nn_utils.constant_param(value=0.0, shape=(self.dim,))

        self.W_mem_hid_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.W_mem_hid_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_mem_hid = nn_utils.constant_param(value=0.0, shape=(self.dim,))

        self.W_b = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.W_1 = nn_utils.normal_param(std=0.1, shape=(self.dim, 7 * self.dim + 0))
        self.W_2 = nn_utils.normal_param(std=0.1, shape=(1, self.dim))
        self.b_1 = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        self.b_2 = nn_utils.constant_param(value=0.0, shape=(1,))


        print "==> building episodic memory module (fixed number of steps: %d)" % self.memory_hops
        memory = [self.q_q.copy()]
        for iter in range(1, self.memory_hops + 1):
            current_episode = self.new_episode(memory[iter - 1])
            memory.append(self.GRU_update(memory[iter - 1], current_episode,
                                          self.W_mem_res_in, self.W_mem_res_hid, self.b_mem_res,
                                          self.W_mem_upd_in, self.W_mem_upd_hid, self.b_mem_upd,
                                          self.W_mem_hid_in, self.W_mem_hid_hid, self.b_mem_hid))

        last_mem_raw = memory[-1].dimshuffle((1, 0))

################################# Episodic Memory Module for Image

        print "==> creating parameters for image memory module"
        self.W_img_mem_res_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.W_img_mem_res_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_img_mem_res = nn_utils.constant_param(value=0.0, shape=(self.dim,))

        self.W_img_mem_upd_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.W_img_mem_upd_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_img_mem_upd = nn_utils.constant_param(value=0.0, shape=(self.dim,))

        self.W_img_mem_hid_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.W_img_mem_hid_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_img_mem_hid = nn_utils.constant_param(value=0.0, shape=(self.dim,))

        self.W_img_b = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.W_img_1 = nn_utils.normal_param(std=0.1, shape=(self.dim, 7 * self.dim + 0))
        self.W_img_2 = nn_utils.normal_param(std=0.1, shape=(1, self.dim))
        self.b_img_1 = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        self.b_img_2 = nn_utils.constant_param(value=0.0, shape=(1,))


        print "==> building episodic img_memory module (fixed number of steps: %d)" % self.memory_hops
        img_memory = [self.q_q.copy()]
        for iter in range(1, self.memory_hops + 1):
            current_episode = self.new_img_episode(img_memory[iter - 1])
            img_memory.append(self.GRU_update(img_memory[iter - 1], current_episode,
                                          self.W_img_mem_res_in, self.W_img_mem_res_hid, self.b_img_mem_res,
                                          self.W_img_mem_upd_in, self.W_img_mem_upd_hid, self.b_img_mem_upd,
                                          self.W_img_mem_hid_in, self.W_img_mem_hid_hid, self.b_img_mem_hid))

        last_img_mem_raw = img_memory[-1].dimshuffle((1, 0))




#######################################################################

        ### Concatenating The 2 Memory Modules Representations Assuming the representation as self.batch_size x self.dim  ###

        combined_mem_raw=T.concatenate([last_mem_raw,last_img_mem_raw],axis=1)

        #net = layers.InputLayer(shape=(self.batch_size, self.dim), input_var=last_mem_raw)

        net = layers.InputLayer(shape=(self.batch_size, self.dim+self.dim), input_var=combined_mem_raw)
        if self.batch_norm:
            net = layers.BatchNormLayer(incoming=net)
        if self.dropout > 0 and self.mode == 'train':
            net = layers.DropoutLayer(net, p=self.dropout)
        last_mem = layers.get_output(net).dimshuffle((1, 0))


        print "==> building answer module"
        #self.W_a = nn_utils.normal_param(std=0.1, shape=(self.vocab_size, self.dim))
        self.W_a = nn_utils.normal_param(std=0.1, shape=(self.vocab_size, self.dim+self.dim))
        if self.answer_module == 'feedforward':
            self.prediction = nn_utils.softmax(T.dot(self.W_a, last_mem))

        elif self.answer_module == 'recurrent':
            self.W_ans_res_in = nn_utils.normal_param(std=0.1, shape=(2*self.dim, self.dim + self.vocab_size))
            self.W_ans_res_hid = nn_utils.normal_param(std=0.1, shape=(2*self.dim, 2*self.dim))
            self.b_ans_res = nn_utils.constant_param(value=0.0, shape=(2*self.dim,))

            self.W_ans_upd_in = nn_utils.normal_param(std=0.1, shape=(2*self.dim, self.dim + self.vocab_size))
            self.W_ans_upd_hid = nn_utils.normal_param(std=0.1, shape=(2*self.dim,2*self.dim))
            self.b_ans_upd = nn_utils.constant_param(value=0.0, shape=(2*self.dim,))

            self.W_ans_hid_in = nn_utils.normal_param(std=0.1, shape=(2*self.dim, self.dim + self.vocab_size))
            self.W_ans_hid_hid = nn_utils.normal_param(std=0.1, shape=(2*self.dim, 2*self.dim))
            self.b_ans_hid = nn_utils.constant_param(value=0.0, shape=(2*self.dim,))

            def answer_step(prev_a, prev_y):
                a = self.GRU_update(prev_a, T.concatenate([prev_y, self.q_q]),
                                  self.W_ans_res_in, self.W_ans_res_hid, self.b_ans_res,
                                  self.W_ans_upd_in, self.W_ans_upd_hid, self.b_ans_upd,
                                  self.W_ans_hid_in, self.W_ans_hid_hid, self.b_ans_hid)

                y = nn_utils.softmax(T.dot(self.W_a, a))
                return [a, y]

            # TODO: add conditional ending
            dummy = theano.shared(np.zeros((self.vocab_size, self.batch_size), dtype=floatX))
            results, updates = theano.scan(fn=answer_step,
                outputs_info=[last_mem, T.zeros_like(dummy)], #(last_mem, y)
                n_steps=1)
            self.prediction = results[1][-1]

        else:
            raise Exception("invalid answer_module")

        self.prediction = self.prediction.dimshuffle(1, 0)

        self.params = [self.W_inp_res_in, self.W_inp_res_hid, self.b_inp_res,
                  self.W_inp_upd_in, self.W_inp_upd_hid, self.b_inp_upd,
                  self.W_inp_hid_in, self.W_inp_hid_hid, self.b_inp_hid,
                  self.W_mem_res_in, self.W_mem_res_hid, self.b_mem_res,
                  self.W_mem_upd_in, self.W_mem_upd_hid, self.b_mem_upd,
                  self.W_mem_hid_in, self.W_mem_hid_hid, self.b_mem_hid, #self.W_b
                  self.W_1, self.W_2, self.b_1, self.b_2, self.W_a,  ## Add the parameters of the Image Input Module
                  self.W_img_mem_res_in, self.W_img_mem_res_hid, self.b_img_mem_res,
                  self.W_img_mem_upd_in, self.W_img_mem_upd_hid, self.b_img_mem_upd,
                  self.W_img_mem_hid_in, self.W_img_mem_hid_hid, self.b_img_mem_hid, #self.W_img_b_img
                  self.W_img_1, self.W_img_2, self.b_img_1, self.b_img_2]  ## Add the parameters of the Image Input Module

        dim_transform_mlp_params=layers.get_all_params(img_input_layer )

        self.params=self.params+ dim_transform_mlp_params

        if self.answer_module == 'recurrent':
            self.params = self.params + [self.W_ans_res_in, self.W_ans_res_hid, self.b_ans_res,
                              self.W_ans_upd_in, self.W_ans_upd_hid, self.b_ans_upd,
                              self.W_ans_hid_in, self.W_ans_hid_hid, self.b_ans_hid]


        print "==> building loss layer and computing updates"
        self.loss_ce = T.nnet.categorical_crossentropy(self.prediction, self.answer_var).mean()

        if self.l2 > 0:
            self.loss_l2 = self.l2 * nn_utils.l2_reg(self.params)
        else:
            self.loss_l2 = 0

        self.loss = self.loss_ce + self.loss_l2

        self.learning_rate=T.scalar(name="learning_rate")
        updates=lasagne.updates.adam(self.loss,self.params,learning_rate=self.learning_rate)
        #updates = lasagne.updates.adadelta(self.loss, self.params)
        #updates = lasagne.updates.momentum(self.loss, self.params, learning_rate=0.001)

        if self.mode == 'train':
            print "==> compiling train_fn"
            self.train_fn = theano.function(inputs=[self.input_var, self.q_var, self.answer_var,
                                                    self.fact_count_var, self.input_mask_var,self.img_input_var,self.learning_rate],
                                            outputs=[self.prediction, self.loss],
                                            updates=updates)

        print "==> compiling test_fn"
        self.test_fn = theano.function(inputs=[self.input_var, self.q_var, self.answer_var,
                                               self.fact_count_var, self.input_mask_var,self.img_input_var,self.learning_rate],on_unused_input='ignore',
                                       outputs=[self.prediction, self.loss])



    def GRU_update(self, h, x, W_res_in, W_res_hid, b_res,
                         W_upd_in, W_upd_hid, b_upd,
                         W_hid_in, W_hid_hid, b_hid):
        """ mapping of our variables to symbols in DMN paper:
        W_res_in = W^r
        W_res_hid = U^r
        b_res = b^r
        W_upd_in = W^z
        W_upd_hid = U^z
        b_upd = b^z
        W_hid_in = W
        W_hid_hid = U
        b_hid = b^h
        """
        z = T.nnet.sigmoid(T.dot(W_upd_in, x) + T.dot(W_upd_hid, h) + b_upd.dimshuffle(0, 'x'))
        r = T.nnet.sigmoid(T.dot(W_res_in, x) + T.dot(W_res_hid, h) + b_res.dimshuffle(0, 'x'))
        _h = T.tanh(T.dot(W_hid_in, x) + r * T.dot(W_hid_hid, h) + b_hid.dimshuffle(0, 'x'))
        return z * h + (1 - z) * _h


    def _empty_word_vector(self):
        return np.zeros((self.word_vector_size,), dtype=floatX)


    def input_gru_step(self, x, prev_h):
        return self.GRU_update(prev_h, x, self.W_inp_res_in, self.W_inp_res_hid, self.b_inp_res,
                                     self.W_inp_upd_in, self.W_inp_upd_hid, self.b_inp_upd,
                                     self.W_inp_hid_in, self.W_inp_hid_hid, self.b_inp_hid)


    def new_attention_step(self, ct, prev_g, mem, q_q):
        z = T.concatenate([ct, mem, q_q, ct * q_q, ct * mem, (ct - q_q) ** 2, (ct - mem) ** 2], axis=0)

        l_1 = T.dot(self.W_1, z) + self.b_1.dimshuffle(0, 'x')
        l_1 = T.tanh(l_1)
        l_2 = T.dot(self.W_2, l_1) + self.b_2.dimshuffle(0, 'x')
        G = T.nnet.sigmoid(l_2)[0]
        return G

    def new_img_attention_step(self, ct, prev_g, mem, q_q):
        z = T.concatenate([ct, mem, q_q, ct * q_q, ct * mem, (ct - q_q) ** 2, (ct - mem) ** 2], axis=0)

        l_1 = T.dot(self.W_img_1, z) + self.b_img_1.dimshuffle(0, 'x')
        l_1 = T.tanh(l_1)
        l_2 = T.dot(self.W_img_2, l_1) + self.b_img_2.dimshuffle(0, 'x')
        G = T.nnet.sigmoid(l_2)[0]
        return G


    def new_episode_step(self, ct, g, prev_h):
        gru = self.GRU_update(prev_h, ct,
                             self.W_mem_res_in, self.W_mem_res_hid, self.b_mem_res,
                             self.W_mem_upd_in, self.W_mem_upd_hid, self.b_mem_upd,
                             self.W_mem_hid_in, self.W_mem_hid_hid, self.b_mem_hid)

        h = g * gru + (1 - g) * prev_h
        return h


    def new_episode(self, mem):
        g, g_updates = theano.scan(fn=self.new_attention_step,
            sequences=self.inp_c,
            non_sequences=[mem, self.q_q],
            outputs_info=T.zeros_like(self.inp_c[0][0]))

        if (self.normalize_attention):
            g = nn_utils.softmax(g)

        e, e_updates = theano.scan(fn=self.new_episode_step,
            sequences=[self.inp_c, g],
            outputs_info=T.zeros_like(self.inp_c[0]))

        e_list = []
        for index in range(self.batch_size):
            e_list.append(e[self.fact_count_var[index] - 1, :, index])
        return T.stack(e_list).dimshuffle((1, 0))

    def new_img_episode_step(self, ct, g, prev_h):
        gru = self.GRU_update(prev_h, ct,
                             self.W_img_mem_res_in, self.W_img_mem_res_hid, self.b_img_mem_res,
                             self.W_img_mem_upd_in, self.W_img_mem_upd_hid, self.b_img_mem_upd,
                             self.W_img_mem_hid_in, self.W_img_mem_hid_hid, self.b_img_mem_hid)

        h = g * gru + (1 - g) * prev_h
        return h


    def new_img_episode(self, mem):
        g, g_updates = theano.scan(fn=self.new_img_attention_step,
            sequences=self.img_inp_c,
            non_sequences=[mem, self.q_q],
            outputs_info=T.zeros_like(self.img_inp_c[0][0]))

        if (self.normalize_attention):
            g = nn_utils.softmax(g)

        e, e_updates = theano.scan(fn=self.new_img_episode_step,
            sequences=[self.img_inp_c, g],
            outputs_info=T.zeros_like(self.img_inp_c[0]))

        e_list = []
        for index in range(self.batch_size):
            e_list.append(e[self.img_seq_len - 1, :, index])
        return T.stack(e_list).dimshuffle((1, 0))

    def save_params(self, file_name, epoch, **kwargs):
        with open(file_name, 'w') as save_file:
            pickle.dump(
                obj = {
                    'params' : [x.get_value() for x in self.params],
                    'epoch' : epoch,
                    'gradient_value': (kwargs['gradient_value'] if 'gradient_value' in kwargs else 0)
                },
                file = save_file,
                protocol = -1
            )


    def load_state(self, file_name):
        print "==> loading state %s" % file_name
        with open(file_name, 'r') as load_file:
            dict = pickle.load(load_file)
            loaded_params = dict['params']
            for (x, y) in zip(self.params, loaded_params):
                x.set_value(y)


    def _process_batch(self, _inputs, _questions, _answers, _fact_counts, _input_masks , _img_feats):
        inputs = copy.deepcopy(_inputs)
        questions = copy.deepcopy(_questions)
        answers = copy.deepcopy(_answers)
        fact_counts = copy.deepcopy(_fact_counts)
        input_masks = copy.deepcopy(_input_masks)
        img_feats=copy.deepcopy(_img_feats)

        zipped = zip(inputs, questions, answers, fact_counts, input_masks,img_feats)

        max_inp_len = 0
        max_q_len = 0
        max_fact_count = 0
        for inp, q, ans, fact_count, input_mask,img_feat in zipped:
            max_inp_len = max(max_inp_len, len(inp))
            max_q_len = max(max_q_len, len(q))
            max_fact_count = max(max_fact_count, fact_count)

        questions = []
        inputs = []
        answers = []
        fact_counts = []
        input_masks = []
        img_feats= []

        for inp, q, ans, fact_count, input_mask , img_feat in zipped:
            while(len(inp) < max_inp_len):
                inp.append(self._empty_word_vector())

            while(len(q) < max_q_len):
                q.append(self._empty_word_vector())

            while(len(input_mask) < max_fact_count):
                input_mask.append(-1)

            inputs.append(inp)
            questions.append(q)
            answers.append(ans)
            fact_counts.append(fact_count)
            input_masks.append(input_mask)
            img_feats.append(img_feat)

        inputs = np.array(inputs).astype(floatX)
        questions = np.array(questions).astype(floatX)
        answers = np.array(answers).astype(np.int32)
        fact_counts = np.array(fact_counts).astype(np.int32)
        input_masks = np.array(input_masks).astype(np.int32)
        img_feats=np.array(img_feats).astype(floatX)

        return inputs, questions, answers, fact_counts, input_masks,img_feats


    def _process_input(self, data_raw):
        questions = []
        inputs = []
        answers = []
        fact_counts = []
        input_masks = []
        img_features=[]
        for x in data_raw:
            #inp = x["C"].lower().split(' ')
            x["C"]=x["C"].lower()
            inp=re.split("[, \-!?:'\/]+",x["C"])
            inp = [w for w in inp if len(w) > 0]
            x["Q"]=x["Q"].lower()
            q = re.split("[, \-!?:'\/]+",x["Q"])
            q = [w for w in q if len(w) > 0]

            inp_vector = [utils.process_word(word = w,
                                        word2vec = self.word2vec,
                                        vocab = self.vocab,
                                        ivocab = self.ivocab,
                                        word_vector_size = self.word_vector_size,
                                        to_return = "word2vec") for w in inp]

            q_vector = [utils.process_word(word = w,
                                        word2vec = self.word2vec,
                                        vocab = self.vocab,
                                        ivocab = self.ivocab,
                                        word_vector_size = self.word_vector_size,
                                        to_return = "word2vec") for w in q]

            if (self.input_mask_mode == 'word'):
                input_mask = range(len(inp))
            elif (self.input_mask_mode == 'sentence'):
                input_mask = [index for index, w in enumerate(inp) if w == '.']
            else:
                raise Exception("unknown input_mask_mode")
            fact_count = len(input_mask)

            inputs.append(inp_vector)
            questions.append(q_vector)
            # NOTE: here we assume the answer is one word!
            #answers.append(utils.process_word(word = x["A"],
            #                                word2vec = self.word2vec,
            #                                vocab = self.vocab,
            #                                ivocab = self.ivocab,
            #                                word_vector_size = self.word_vector_size,
            #                                to_return = "index"))
            answers.append(x["A"])
            fact_counts.append(fact_count)
            input_masks.append(input_mask)
            img_features.append(x["I"])

        return inputs, questions, answers, fact_counts, input_masks , img_features


    def get_batches_per_epoch(self, mode):
        if (mode == 'train'):
            num_train,_=self.h5file['ques_train'].shape
            return num_train / self.batch_size
        elif (mode == 'test'):
            num_test,_=self.h5file['ques_test'].shape
            return num_test / self.batch_size
        else:
            raise Exception("unknown mode")


    def shuffle_train_set(self):
        print "==> Shuffling the train set"
        #combined = zip(self.train_input, self.train_q, self.train_answer, self.train_fact_count, self.train_input_mask)
        #random.shuffle(combined)
        #self.train_input, self.train_q, self.train_answer, self.train_fact_count, self.train_input_mask = zip(*combined)

    def convert_index_to_word(self,index):
        return self.json_dict["ix_to_word"][str(index)]

    def process_vqa_data(self,mode,start,end):
        if mode=="train":
            inputs=self.h5file['cap_train'][start:end]
            qs=self.h5file['ques_train'][start:end]
            answers=self.h5file['answers'][start:end]
            img_indices=self.h5file['img_pos_train'][start:end]

        if mode=="test":
            inputs=self.h5file['cap_test'][start:end]
            qs=self.h5file['ques_test'][start:end]
            answers=self.h5file['ans_test'][start:end]
            img_indices=self.h5file['img_pos_test'][start:end]

        all_x=[]
        for i in range(self.batch_size):
            x={}
            x["A"]=answers[i]-1
            temp_Q=map( self.convert_index_to_word , filter( lambda x: x>0 , qs[i]  )  )
            x["Q"]=' '.join(temp_Q)

            temp_Cs=[]
            for j in range(inputs.shape[1]):
                temp_C=map(self.convert_index_to_word,filter(lambda x: x>0 , inputs[i][j]))
                temp_C_str=' '.join(temp_C)
                temp_Cs.append(temp_C_str)
            x["C"]=' . '.join(temp_Cs)
            if mode=="train":
                x["I"]=self.img_h5file_train['images_train'][ img_indices[i]-1].reshape((196,512))
            if mode=="test":
                x["I"]=self.img_h5file_test['images_test'][ img_indices[i]-1].reshape((196,512))

            all_x.append(x)
        return self. _process_input(all_x)

    def process_masks(self,captions):
        print("captions")
        print(captions)

    def step(self,batch_index,mode):
        if mode== "train" and self.mode== "test":
            raise Exception("Cannot train during test mode")

        start_index=batch_index * self.batch_size


        inputs, qs, answers, fact_counts, input_masks,img_feats =self.process_vqa_data(mode,start_index,start_index+self.batch_size)
        if mode=="train":
            theano_fn=self.train_fn
           # inputs = self.process_vqa_data(self.h5file['cap_train'][start_index:start_index+self.batch_size])
           # qs=self.process_vqa_data(self.h5file['ques_train' ][ start_index:start_index+self.batch_size] )
           # answers=self.process_vqa_data(self.h5file['answers'][start_index:start_index+self.batch_size] )
           # fact_counts=np.zeros(self.batch_size,dtype="int")
           # fact_counts.fill(20)
           # input_masks= process_masks( inputs )  # figure it out

        if mode=="test":
            theano_fn=self.test_fn
           # inputs=self.process_vqa_data( self.h5file['cap_test'][start_index:start_index+self.batch_size ]  )
           # qs=self.process_vqa_data( self.h5file['ques_test'][start_index:start_index+self.batch_size ]  )
           # answers=self.process_vqa_data( self.h5file['ans_test'][start_index:start_index+self.batch_size ]  )
           # fact_counts=np.zeros(self.batch_size,dtype="int")
           # fact_counts.fill(20)
           # input_masks= process_masks( inputs  ) # figure it out
        inp,q,ans,fact_count,input_mask,img_feat=self._process_batch(inputs,qs,answers,fact_counts,input_masks,img_feats )
        img_feat=img_feat.reshape((self.batch_size*self.img_seq_len,self.img_vector_size))
        ret = theano_fn( inp,q,ans,fact_count,input_mask,img_feat,self.lr)

        param_norm=np.max( [ utils.get_norm( x.get_value()) for x  in self.params])

        return { "prediction":ret[0],
                 "answers":ans,
                 "current_loss":ret[1],
                 "skipped":0,
                 "log":"pn: %.3f" % param_norm
                 }

