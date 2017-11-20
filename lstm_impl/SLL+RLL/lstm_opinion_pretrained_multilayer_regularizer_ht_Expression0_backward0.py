'''
Step 1 : Get the model working with only 1 hidden layer to the extent that we have some performance numbers from the task
Step 2 : Extend the model to include more parameters and more number of hidden layers.
Step 3 : Bi-directional (Code in the directory more certainly)
Step 4 : Multilayer bidiectional LSTM network
'''

from collections import OrderedDict
import cPickle as pkl
import random
import sys
import time
import cProfile, pstats, StringIO


import numpy
import scipy
import theano
import copy
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import mpqa_ht0



pr = cProfile.Profile()

datasets = {'mpqa_ht0': (mpqa_ht0.load_data, mpqa_ht0.prepare_data, mpqa_ht0.prepare_data_words, mpqa_ht0.find_relations_lrE)}

word_labels = dict()

lookuptable = dict()
index_dict = dict()
inv_index_dict = dict()

ndim = 300

Wemb1 = tensor.matrix('Wemb', dtype=config.floatX)
Wemb_temp = numpy.matrix(numpy.zeros((ndim), dtype=config.floatX))

with open('../data_MPQA/dict.txt') as f1 :
    for line in f1.readlines() : 
        index_dict[line.split("\t")[0]] = line.split("\t")[1].split("\n")[0]
        inv_index_dict[line.split("\t")[1].split("\n")[0]] = line.split("\t")[0]
        #print line.split("\t")[1]

'''write_file = open('dict_vect_300.txt', 'w+')

with open('../../data_MPQA/vectors_allwords.txt') as f : 
    for line in f.readlines() : 
        word_vector = []
        word = line.split("\t")[0]
        #print word
        #print inv_index_dict.keys()
        
        #print "Not Present\t"+word
        if word in inv_index_dict.keys() : 
            print "Present\t"+word
            write_file.write(word+"\t")
            for j in range(0, 300) : 
                write_file.write(line.split("\t")[j+1])
                write_file.write("\t")
            write_file.write("\n")
        #lookuptable[word] =  word_vector
        
        
exit()
'''

initialize = theano.function([Wemb1], Wemb1)

with open('dict_vect_300.txt') as f2 : 
    for line in f2 : 
        word = line.split("\t")[0]
        if len(word) > 0 : 
            word_vector = []
            for j in range(1, ndim+1) : 
                prob =(line.split("\t")[j]).strip()
                #print str(prob)+"\t"+str(j)
                prob =  float(prob)
                word_vector.append(prob)
            lookuptable[inv_index_dict.get(word)] = word_vector

print len(index_dict)

for index in range(1, len(index_dict)+1) : 
    #print str(index)+"\t"+index_dict.get(str(index))
    vector = []
    vector = lookuptable.get(str(index))
    if vector == None : #initialise it with random numbers
        #print "None"
        randn = numpy.random.rand(ndim) #relate it to the number of dimensions finally
        vector = (0.01 * randn)
    vector = numpy.matrix(vector)
    
    #print vector
    #print Wemb_temp
    Wemb_temp = numpy.concatenate((Wemb_temp, vector))
    #print Wemb_temp
#print Wemb_temp

#print numpy.matrix(Wemb_temp, dtype=config.floatX)
#    if index == 4 : 
print Wemb1.shape
Wemb1 = initialize(numpy.matrix(Wemb_temp, dtype = config.floatX))

def numpy_floatX(data) : 
    return numpy.asarray(data, dtype=config.floatX)


def get_minibatches_idx(n, minibatch_size, shuffle=True) :
    """
    Used to shuffle the dataset at each iteration
    """
    idx_list = numpy.arange(n, dtype="int32")
    if shuffle : 
        random.shuffle(idx_list)
        
    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size) : 
        minibatches.append(idx_list[minibatch_start : minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n) :
        #Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])
    return zip(range(len(minibatches)), minibatches)
        

def get_dataset(name):
    return datasets[name][0], datasets[name][1], datasets[name][2], datasets[name][3]

def load_params(path, params) : 
    pp = numpy.load(path)
    for kk, vv in params.iteritems() : 
        if kk not in pp : 
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]
    return params


def init_params(options) :
    """
    Global (not LSTM) paramete. For the embedding and the classifier.
    """
    params = OrderedDict()
    #embedding
    #randn = numpy.random.rand(options['n_words'], options['dim_proj'])
    #intialise Wemb with the lookup table. 2 -dimensional array
    #params['Wemb'] = (0.01 * randn).astype(config.floatX)
    #params['Wemb'] = Wemb1
    use_noise = theano.shared(numpy_floatX(0.))
    for i in numpy.arange(1, options['hidden_layers']+1) : 
        params = get_layer(options['encoder'])[0](options,  params, i, prefix=options['encoder']) #Here we pass arguments for param_init_lstm!!

    #params[1] = get_layer(optons['encoder'])[0](options,  params, prefix=options['encoder'])

    #classifier # Will probably need to change this part!

    '''temp = dict()
    for i in numpy.arange(options['ydim']) : 
        for j in numpy.arange(options['ydim']) : 
            if ((i == 0) and (j == 2 or j== 4 or j==6)):
                continue
            if ((i == 1 or i == 2) and (j== 4 or j==6)):
                continue
            if ((i == 3 or i == 4) and (j== 2 or j==6)): 
                continue
            if ((i == 5 or i == 6) and (j== 2 or j==4)):
                continue                
            temp[i, j] = 0.0
    
    names = ['id', 'data']
    formats = ['(int, int)', 'float32']
    dtype = dict(names = names, formats=formats)'''
    params['A'] = numpy.zeros((options['ydim'], options['ydim'])).astype(config.floatX)

    params['A_hrel'] = numpy.zeros((options['window']+1, options['window']+1, options['ydim'], options['ydim'])).astype(config.floatX)
    params['A_trel'] = numpy.zeros((options['window']+1, options['window']+1, options['ydim'], options['ydim'])).astype(config.floatX)

    #, options['window']
    #numpy.array(temp.items(), dtype=dtype)

    params['U_f'] = 0.01 * numpy.random.randn(options['hidden_layers']*options['dim_proj'], options['ydim']).astype(config.floatX)
    #params['b_f'] = numpy.zeros((options['ydim'],)).astype(config.floatX)
    
    params['U_b'] = 0.01 * numpy.random.randn(options['hidden_layers']*options['dim_proj'], options['ydim']).astype(config.floatX)
    params['b'] = numpy.zeros((options['ydim'],)).astype(config.floatX)

    params['U_flr'] = 0.01 * numpy.random.randn(options['window']+1, options['hidden_layers']*options['dim_proj'], options['ydim']).astype(config.floatX)
    params['b_flr'] = numpy.zeros(( options['window']+1, options['ydim'])).astype(config.floatX)
    
    params['U_brr'] = 0.01 * numpy.random.randn(options['window']+1, options['hidden_layers']*options['dim_proj'], options['ydim']).astype(config.floatX)
    params['b_brr'] = numpy.zeros(( options['window']+1, options['ydim'])).astype(config.floatX)
    
    params['U_flr_b'] = 0.01 * numpy.random.randn(options['window']+1, options['hidden_layers']*options['dim_proj'], options['ydim']).astype(config.floatX)
    #  params['b_flr_b'] = numpy.zeros(( options['window']+1, options['ydim'])).astype(config.floatX)

    params['U_brr_b'] = 0.01 * numpy.random.randn(options['window']+1, options['hidden_layers']*options['dim_proj'], options['ydim']).astype(config.floatX)
    #params['b_brr_b'] = numpy.zeros(( options['window']+1, options['ydim'])).astype(config.floatX)


    return params

def get_layer(name) : 
    fns = layers[name]
    return fns

def dropout_layer(state_before, use_noise, trng) :
    proj = tensor.switch(use_noise, state_before * trng.binomial(state_before.shape, p = 0.5, n=1, dtype=state_before.dtype), state_before * 0.5)
    return proj
    

def zipp(params, tparams) : 
    """
    When we reload the model. Needed for the GPU stuff
    """
    for kk, vv in params.iteritems() : 
        tparams[kk].set_value(vv)


def unzip(zipped) : 
    """
    When we pckle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems() : 
        new_params[kk] = vv.get_value()
    return new_params



#Did not understand the use of this function
def init_tparams(params) : 
    tparams = OrderedDict()
    for kk, pp in params.iteritems() :
        tparams[kk] = theano.shared(params[kk].astype(config.floatX), name=kk)
    return tparams



def _p(pp, name) : 
    return '%s_%s' % (pp, name)

def ortho_weight(ndim) : 
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)

def ortho_weight_2(ndim1, ndim2) :
    W = 0.01 * numpy.random.randn(ndim1, ndim2)
    #u, s, v = numpy.linalg.svd(W)
    #return u.astype(config.floatX)
    return W.astype(config.floatX)
    

def param_init_lstm(options, params, layer, prefix='lstm') :
    """
    Init the LSTM parameters : 
    :see: init_params
    """
    if layer == 1 : 
        print "True"
        W_f = numpy.concatenate([ortho_weight_2(ndim, options['dim_proj']),
                               ortho_weight_2(ndim, options['dim_proj']),
                               ortho_weight_2(ndim, options['dim_proj']),
                               ortho_weight_2(ndim, options['dim_proj'])], axis = 1)
        params[_p(prefix, 'W_f'+str(layer))] = W_f

        W_b = numpy.concatenate([ortho_weight_2(ndim, options['dim_proj']),
                               ortho_weight_2(ndim, options['dim_proj']),
                               ortho_weight_2(ndim, options['dim_proj']),
                               ortho_weight_2(ndim, options['dim_proj'])], axis = 1)
        params[_p(prefix, 'W_b'+str(layer))] = W_b

        
        U_f = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis = 1)

        V_f = ortho_weight(options['dim_proj'])
        params[_p(prefix, 'U_f'+str(layer))] = U_f
        params[_p(prefix, 'V_f'+str(layer))] = V_f
        b_f = numpy.zeros((4 * options['dim_proj'],))
        params[_p(prefix, 'b_f'+str(layer))] = b_f.astype(config.floatX)

        U_b = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis = 1)

        V_b = ortho_weight(options['dim_proj'])
        params[_p(prefix, 'U_b'+str(layer))] = U_b
        params[_p(prefix, 'V_b'+str(layer))] = V_b
        b_b = numpy.zeros((4 * options['dim_proj'],))
        params[_p(prefix, 'b_b'+str(layer))] = b_b.astype(config.floatX)

    else : 
        print "False"
        W_f = numpy.concatenate([ortho_weight_2(options['dim_proj'], options['dim_proj']),
                               ortho_weight_2(options['dim_proj'], options['dim_proj']),
                               ortho_weight_2(options['dim_proj'], options['dim_proj']),
                               ortho_weight_2(options['dim_proj'], options['dim_proj'])], axis = 1)
        params[_p(prefix, 'W_f'+str(layer))] = W_f
        
        W_b = numpy.concatenate([ortho_weight_2(options['dim_proj'], options['dim_proj']),
                               ortho_weight_2(options['dim_proj'], options['dim_proj']),
                               ortho_weight_2(options['dim_proj'], options['dim_proj']),
                               ortho_weight_2(options['dim_proj'], options['dim_proj'])], axis = 1)
        params[_p(prefix, 'W_b'+str(layer))] = W_b
        
        U_f = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis = 1)

        V_f = ortho_weight(options['dim_proj'])
        params[_p(prefix, 'U_f'+str(layer))] = U_f
        params[_p(prefix, 'V_f'+str(layer))] = V_f
        b_f = numpy.zeros((4 * options['dim_proj'],))
        params[_p(prefix, 'b_f'+str(layer))] = b_f.astype(config.floatX)

        U_b = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis = 1)

        V_b = ortho_weight(options['dim_proj'])
        params[_p(prefix, 'U_b'+str(layer))] = U_b
        params[_p(prefix, 'V_b'+str(layer))] = V_b
        b_b = numpy.zeros((4 * options['dim_proj'],))
        params[_p(prefix, 'b_b'+str(layer))] = b_b.astype(config.floatX)



    BDWf_f = ortho_weight(options['dim_proj'])
    BDWf_b = ortho_weight(options['dim_proj'])
    BDb_f = numpy.zeros((options['dim_proj'],))
    
    params[_p(prefix, 'BDWf_f'+str(layer))] = BDWf_f
    params[_p(prefix, 'BDWf_b'+str(layer))] = BDWf_b
    params[_p(prefix, 'BDb_f'+str(layer))] = BDb_f.astype(config.floatX)

    BDWb_f = ortho_weight(options['dim_proj'])
    BDWb_b = ortho_weight(options['dim_proj'])
    BDb_b = numpy.zeros((options['dim_proj'],))
    
    params[_p(prefix, 'BDWb_f'+str(layer))] = BDWb_f
    params[_p(prefix, 'BDWb_b'+str(layer))] = BDWb_b
    params[_p(prefix, 'BDb_b'+str(layer))] = BDb_b.astype(config.floatX)

    return params


def lstm_layer(use_noise, tparams, state_below_f, state_below_b, options, layer, prefix='lstm', mask=None) :
    nsteps = state_below_f.shape[0]
    if state_below_f.ndim == 3:
        n_samples = state_below_f.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _stepf(m_, x_, h_, c_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U_f'+str(layer))])
        preact += x_
        preact += tparams[_p(prefix, 'b_f'+str(layer))]

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
        #o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
        c = tensor.tanh(_slice(preact, 3, options['dim_proj']))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        preact1 = _slice(preact, 2, options['dim_proj']) + tensor.dot(c, tparams[_p(prefix, 'V_f'+str(layer))])
        o = tensor.nnet.sigmoid(preact1)


        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    def _stepb(m_, x_, h_, c_) :
        preact = tensor.dot(h_, tparams[_p(prefix, 'U_b'+str(layer))])
        preact += x_
        preact += tparams[_p(prefix, 'b_b'+str(layer))]

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
        #o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
        c = tensor.tanh(_slice(preact, 3, options['dim_proj']))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        preact1 = _slice(preact, 2, options['dim_proj']) + tensor.dot(c, tparams[_p(prefix, 'V_b'+str(layer))])
        o = tensor.nnet.sigmoid(preact1)


        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c
        
    trng = RandomStreams(1234)

    #Used for dropout.
    #use_noise = theano.shared(numpy_floatX(0.))


    state_below_f = (tensor.dot(state_below_f, tparams[_p(prefix, 'W_f'+str(layer))]) +
                   tparams[_p(prefix, 'b_f'+str(layer))])

    state_below_b = (tensor.dot(state_below_b, tparams[_p(prefix, 'W_b'+str(layer))]) +
                     tparams[_p(prefix, 'b_b'+str(layer))])

    if options['use_dropout'] : 
        state_below_f = dropout_layer(state_below_f, use_noise, trng)
    if options['use_dropout'] : 
        state_below_b = dropout_layer(state_below_b, use_noise, trng)


        
    dim_proj = options['dim_proj']
    rval_f, updates = theano.scan(_stepf,
                                sequences=[mask, state_below_f],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)

    #still need to find how to reverse a tensor and the output rval_b before combining

    rval_b, updates = theano.scan(_stepb,
                                sequences=[mask, state_below_b],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                  n_steps=nsteps, go_backwards=True)
    y_f = tensor.tanh(tensor.dot(rval_f[0], tparams[_p(prefix, 'BDWf_f'+str(layer))]) + tensor.dot(rval_b[0][::-1], tparams[_p(prefix, 'BDWf_b'+str(layer))]) + tparams[_p(prefix, 'BDb_f'+str(layer))])
    
    #####Made CHANGE HERE!!!!!!!!!!!!!!!!!!!!!! DID not have any tensor.tanh!!!
    y_r = tensor.tanh(tensor.dot(rval_f[0], tparams[_p(prefix, 'BDWb_f'+str(layer))]) + tensor.dot(rval_b[0][::-1], tparams[_p(prefix, 'BDWb_b'+str(layer))]) + tparams[_p(prefix, 'BDb_b'+str(layer))])

    return y_f, y_r


#ff : Feed Forward (normal neual net), only useful to put after lstm
#befor e the classifier

layers = {'lstm' : (param_init_lstm, lstm_layer)}


#X[0] : corresponds to the number of words in a sentence
#X[1] : numbe of training examples
#X[2] : dimension 128


def build_model(tparams, options, Wemb) : 
   # trng = RandomStreams(1234)

    #Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

    x = tensor.matrix('x', dtype='int32')
    mask = tensor.matrix('mask', dtype=config.floatX)
    #y = tensor.vector('y', dtype='int64')
    y = tensor.matrix('y', dtype='int32')

    all_holder_relations = tensor.tensor3('all_holder_relations', dtype='int32')
    all_target_relations = tensor.tensor3('all_target_relations', dtype='int32')

    #, holder_indices, target_indices

    holder_indices = tensor.tensor3('holder_indices', dtype='int32')
    target_indices = tensor.tensor3('target_indices', dtype='int32')

    n_holder_shape = all_holder_relations.shape[2]
    n_target_shape = all_target_relations.shape[2]

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    #print Wemb1[[2, 3, 4]].reshape([1, 3, 128])
    #print x
    
    #emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps, n_samples, options['dim_proj']])
    #emb = Wemb[x.flatten()].reshape([n_timesteps, n_samples, options['dim_proj']])
    emb = Wemb[x.flatten()].reshape([n_timesteps, n_samples, ndim])
    
    y = y.reshape([n_timesteps, n_samples])

    final_output = []

    concoutput_f = tensor.matrix('concoutput_f', dtype=config.floatX)
    concoutput_b = tensor.matrix('concoutput_b', dtype=config.floatX)

    previous_output_f = emb.astype(config.floatX)
    previous_output_b = emb.astype(config.floatX)

    for i in numpy.arange(1, options['hidden_layers']+1) : 
        proj_f, proj_b= get_layer(options['encoder'])[1](use_noise, tparams, previous_output_f, previous_output_b, options, i, prefix=options['encoder'], mask = mask) #Here 1 signifies that there is only 1 hidden layer?!
        
        '''if options['encoder'] == 'lstm' : #Is this sum kind of an average?
        proj = (proj * mask[:, :, None]).sum(axis=0)
        proj = proj/ mask.sum(axis=0)[:, None]'''
        '''if options['use_dropout'] : 
            proj_f = dropout_layer(proj_f, use_noise, trng)
        if options['use_dropout'] : 
            proj_b = dropout_layer(proj_b, use_noise, trng)'''

        if i == 1 : 
            concoutput_f = proj_f
            concoutput_b = proj_b
        else : 
            concoutput_f = tensor.concatenate([concoutput_f, proj_f], axis=2)
            concoutput_b = tensor.concatenate([concoutput_b, proj_b], axis=2)

        previous_output_f = proj_f
        previous_output_b = proj_b
        
        #final_output.append(previous_output)


    '''output, output_updates = theano.scan(lambda u, v : tensor.nnet.relu(tensor.dot(u, tparams['U_f']) + tensor.dot(v, tparams['U_b']) + tparams['b']), sequences = (proj_f, proj_b))

    output_lr, output_updates = theano.scan(lambda u, v : tensor.nnet.relu(tensor.dot(u, tparams['U_flr']) +tensor.dot(v, tparams['U_brr']) + tparams['b_flr']), sequences = (proj_f, proj_b))
    output_rr, output_updates = theano.scan(lambda v, u : tensor.nnet.relu(tensor.dot(v, tparams['U_brr_b']) + tensor.dot(u, tparams['U_flr_b'])+ tparams['b_brr']), sequences = (proj_b, proj_f))'''

    output, output_updates = theano.scan(lambda u, v : (tensor.dot(u, tparams['U_f']) + tensor.dot(v, tparams['U_b']) + tparams['b']), sequences = (concoutput_f, concoutput_b))

    output_lr, output_updates = theano.scan(lambda u, v : (tensor.dot(u, tparams['U_flr']) +tensor.dot(v, tparams['U_brr']) + tparams['b_flr']), sequences = (concoutput_f, concoutput_b))
    output_rr, output_updates = theano.scan(lambda v, u : (tensor.dot(v, tparams['U_brr_b']) + tensor.dot(u, tparams['U_flr_b'])+ tparams['b_brr']), sequences = (concoutput_b, concoutput_f))

    ####Now sentence likelihood!!

    y_f = y.flatten(ndim=1)


    def compute_score(o_, ind_, res_, prev_ind_) : 
        #theano.tensor.cast(ind_, 'int64')
        res_ += (tparams['A'][prev_ind_, ind_]*1) + o_[:,ind_]
        return res_, ind_
        
    sent_score, updates_sent = theano.scan(compute_score, sequences=[output, tensor.cast(y_f, 'int32')], outputs_info=[tensor.unbroadcast(tensor.alloc(numpy_floatX(0.), n_samples, 1), 1), tensor.alloc(numpy.cast['int32'](0))])

    def score_alltags(o_, d_, t_) : 
        cand = d_.dimshuffle(0, 'x', 1)+(tparams['A'].dimshuffle(0, 1, 'x')*1)
        #cand = cand.reshape([options['ydim'], options['ydim'], n_samples])
        cand_max = cand.max(axis=0)
        log_sum_exp_val = cand_max + tensor.log(tensor.sum(tensor.exp(cand-cand_max.dimshuffle('x', 0, 1)), axis=0))        
        
        nextd_ = o_.dimshuffle(1, 0) + log_sum_exp_val

        return nextd_, o_.dimshuffle(1, 0)
        
    #pred, updates1 = theano.scan(lambda u : tensor.nnet.softmax(u), sequences = (output))

    #plusOne = tensor.alloc(numpy_floatX(0.), output.shape[0], output.shape[1], 1)

    all_tags, updates_alltags = theano.scan(score_alltags, sequences=[output], outputs_info=[tensor.alloc(numpy_floatX(0.), options['ydim'], n_samples) , tensor.alloc(numpy.float32(0.), options['ydim'], n_samples)])

    #all_tags_score = numpy.logaddexp(all_tags[1], all_tags[2], all_tags[3], all_tags[4], all_tags[5], all_tags[6], all_tags[7])
    all_tags_max = all_tags[0][-1].max(axis=0)
    all_tags_score = all_tags_max + tensor.log(tensor.sum(tensor.exp(all_tags[0][-1]-all_tags_max.dimshuffle('x', 0)), axis=0))

    #all_tags_score = scipy.misc.logsumexp(all_tags)

    #final_score = sent_score[0] - all_tags_score
    final_score = (sent_score[0][-1] - all_tags_score).mean()

    def alltags(d_) : 
        cand = d_.dimshuffle(0, 'x', 1)+(tparams['A'].dimshuffle(0, 1, 'x') * 1)
        pred = cand.argmax(axis=0)
        return pred
        
    tags, updates_alltags = theano.scan(alltags, sequences=[all_tags[0]], outputs_info=None)

    pred_l = all_tags[0][-1].argmax(axis=0)
    
    pred, updatess = theano.scan(lambda x,i : x[i][0], sequences=tags, outputs_info=pred_l, go_backwards=True)

    #f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
    f_pred = theano.function([x, mask], pred[::-1], name='f_pred', on_unused_input='ignore')

    cost = -(final_score)

    #tensor.concatenate([pred[::-1][:-1,:], [pred_l]], axis=0)
    #pred_m = tensor.reshape(pred, (pred.shape[0] * pred.shape[1], -1))

    #f_pred = theano.function([x, mask], pred_m.argmax(axis=1).reshape([n_timesteps, n_samples]), name='f_pred')
    
    '''final_L1_norm = 0
    for k, p in tparams.iteritems() : 
        L1_norm = [(abs(g)).sum() for g in p.get_value()]
        for n in L1_norm  :
            final_L1_norm += n

    final_L2_norm = 0
    for k, p in tparams.iteritems() : 
        L2_norm = [(g ** 2).sum() for g in p.get_value()]
        for n in L2_norm : 
            final_L2_norm += n'''
    
    #cost = -((tensor.log((pred_m)[tensor.arange(pred_m.shape[0]), y_f] + 1e-8))*y_f).mean()

    #cost += 0.001 * abs(final_L1_norm) + 0.001 * abs(final_L2_norm)

    '''final_L2_norm_A = 0
    L2_norm_A = [(gg**2).sum() for gg in tparams['A'].get_value()]
    for n in L2_norm_A : 
        final_L2_norm_A += n
    cost += 0.1 * final_L2_norm_A

    final_L1_norm_A = 0
    L1_norm_A = [(abs(gg)).sum() for gg in tparams['A'].get_value()]
    for n in L1_norm_A : 
        final_L1_norm_A += n
    cost += 0.1 * final_L1_norm_A'''

    '''final_L2_norm_A = 0
    L2_norm_A = [(gg**2).sum() for gg in tparams['A'].get_value()]
    for n in L2_norm_A : 
        final_L2_norm_A += n
    cost += 0.1 * final_L2_norm_A

    final_L1_norm_A = 0
    L1_norm_A = [(abs(gg)).sum() for gg in tparams['A'].get_value()]
    for n in L1_norm_A : 
        final_L1_norm_A += n
    cost += 0.1 * final_L1_norm_A'''

    #cost += -tensor.log((pred_m)[tensor.arange(pred_m.shape[0]), y_f] + 1e-8).mean()

    
    def compute_score_hrel(o_, win_, dimInd_, currY_, res_, prevY_, prevWin_, prevdimInd_) : 
        #res_ += (tparams['A_trel'][y_[currInd_], y_[win_[:, dimInd_[:,0]-1]], win_[:, dimInd_[:,0]-1]]).sum() + o_[:,currInd_]
        #res_ += (tparams['A_hrel'][win_[:, dimInd_[:,0]-1], prevY_, currY_]).sum() + o_[:, currY_]
        res_ += ((tparams['A_hrel'][prevWin_[:, prevdimInd_[:,0]-1], win_[:, dimInd_[:,0]-1], prevY_, currY_]) + o_[:,  win_[:, dimInd_[:,0]-1], currY_]).sum()
        #res1_ += (tparams['A_hrel'][prevWin_[:, prevdimInd_[:,0]-1], win_[:, dimInd_[:,0]-1], prevY_, currY_]).sum()
        return res_, currY_, win_, dimInd_
        
    sent_hscore, updates_sent = theano.scan(compute_score_hrel, sequences=[output_lr, all_holder_relations, holder_indices, y_f], outputs_info=[tensor.unbroadcast(tensor.alloc(numpy_floatX(0.), n_samples, 1), 1), tensor.alloc(numpy.int32(0)), tensor.unbroadcast(tensor.alloc(numpy.int32(0), n_samples, n_holder_shape), 1), tensor.unbroadcast(tensor.alloc(numpy.int32(0), n_samples, 1), 1)])

    def score_alltags_hrel(o_, d_, t_) : 
        cand = d_.dimshuffle(0, 'x', 1, 'x', 2)+(tparams['A_hrel'].dimshuffle(0, 1, 2, 3, 'x'))
        cand_max = cand.max(axis=2)
        cand_max2 = cand_max.max(axis=0)

        log_sum_exp_val = cand_max2 + tensor.log(tensor.sum(tensor.sum(tensor.exp(cand-cand_max2.dimshuffle('x', 0, 'x', 1, 2)), axis=2), axis=0))
        
        #o_.dimshuffle('x', 1, 0) + tensor.log(tensor.sum(tensor.sum(tensor.exp(cand), axis=2), axis=0))
        nextd_ = o_.dimshuffle(1, 2, 0) + log_sum_exp_val
        ###o_.dimshuffle('x', 1, 0) + 
        return nextd_, cand_max2

        
    all_htags, updates_allhtags = theano.scan(score_alltags_hrel, sequences=[output_lr], outputs_info=[tensor.alloc( numpy.float32(0.), options['window']+1, options['ydim'], n_samples) , tensor.alloc(numpy.float32(0.), options['window']+1, options['ydim'], n_samples)])

    all_htags_max = all_htags[0][-1].max(axis=1)
    all_htags_max2 = all_htags_max.max(axis=0)

    all_htags_score2 = all_htags_max2 + tensor.log(tensor.sum(tensor.sum(tensor.exp(all_htags[0][-1]-all_htags_max2.dimshuffle('x', 'x', 0)), axis=1), axis=0))

    #all_htags_score2 = tensor.log(tensor.sum(tensor.sum(tensor.exp(all_htags[0][-1]), axis=1), axis=0))

    final_hscore = (sent_hscore[0][-1] - all_htags_score2).mean()

    def alltags_hrel(d_) : 
        cand = d_.dimshuffle(0, 'x', 1, 'x', 2)+(tparams['A_hrel'].dimshuffle(0, 1, 2, 3, 'x'))
        max1, pred1 = theano.tensor.max_and_argmax(cand, axis=2)
        pred2 = max1.argmax(axis=0)
        
        return pred1, pred2
        
    htags, updates_allhtags = theano.scan(alltags_hrel, sequences=[all_htags[0]], outputs_info=None)

    max_h, pred_hl = theano.tensor.max_and_argmax(all_htags[0][-1], axis=1)
    pred_hw = max_h.argmax(axis=0)

    #pred, updatess = theano.scan(lambda x,l,w : x[l, w][0], sequences=tags, outputs_info=[pred_l[:, pred_w], pred_w], go_backwards=True)

    def temp_fun(x_, y_, l_, w_) : 
        return x_[y_[w_,l_,0], w_,l_,0], y_[w_,l_,0]
        
    hpred, updatess = theano.scan(temp_fun, sequences=[htags[0], htags[1]], outputs_info=[pred_hl[pred_hw, 0], pred_hw], go_backwards=True)


    #f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
    #temp_func = theano.function([x, mask, y, all_holder_relations, all_target_relations], [all_tags[1], sent_score[0][-1]], name='temp_func')
    f_pred_hrel = theano.function([x, mask], outputs=[hpred[0][::-1], hpred[1][::-1]], name='f_pred_hrel', on_unused_input='ignore')
    cost += -final_hscore


    '''final_L2_norm_A = 0
    L2_norm_A = [(gg**2).sum() for gg in tparams['A_hrel'].get_value()]
    for n in L2_norm_A : 
        final_L2_norm_A += n
    cost += 0.1 * final_L2_norm_A

    final_L1_norm_A = 0
    L1_norm_A = [(abs(gg)).sum() for gg in tparams['A_hrel'].get_value()]
    for n in L1_norm_A : 
        final_L1_norm_A += n
    cost += 0.1 * final_L1_norm_A'''
    

    def compute_score_trel(o_, win_, dimInd_, currY_, res_, prevY_, prevWin_, prevdimInd_) : 
        #res_ += (tparams['A_trel'][y_[currInd_], y_[win_[:, dimInd_[:,0]-1]], win_[:, dimInd_[:,0]-1]]).sum() + o_[:,currInd_]
        #res_ += (tparams['A_hrel'][win_[:, dimInd_[:,0]-1], prevY_, currY_]).sum() + o_[:, currY_]
        res_ += ((tparams['A_trel'][prevWin_[:, prevdimInd_[:,0]-1], win_[:, dimInd_[:,0]-1], prevY_, currY_]) + o_[:,  win_[:, dimInd_[:,0]-1], currY_]).sum()
        return res_, currY_, win_, dimInd_
        
    sent_tscore, updates_sent = theano.scan(compute_score_trel, sequences=[output_rr, all_target_relations, target_indices, y_f], outputs_info=[tensor.unbroadcast(tensor.alloc(numpy_floatX(0.), n_samples, 1), 1), tensor.alloc(numpy.int32(0)), tensor.unbroadcast(tensor.alloc(numpy.int32(0), n_samples, n_target_shape), 1), tensor.unbroadcast(tensor.alloc(numpy.int32(0), n_samples, 1), 1)], go_backwards=True)


    def score_alltags_trel(o_, d_, t_) : 
        cand = d_.dimshuffle(0, 'x', 1, 'x', 2)+(tparams['A_trel'].dimshuffle(0, 1, 2, 3, 'x'))
        #cand = cand.reshape([options['ydim'], options['ydim'], n_samples])
        cand_max = cand.max(axis=2)
        cand_max2 = cand_max.max(axis=0)
        log_sum_exp_val2 = cand_max2 + tensor.log(tensor.sum(tensor.sum(tensor.exp(cand-cand_max2.dimshuffle('x', 0, 'x', 1, 2)), axis=2), axis=0))

        nextd_ = o_.dimshuffle(1, 2, 0) + log_sum_exp_val2
        return nextd_, cand_max2

        
    all_ttags, updates_allttags = theano.scan(score_alltags_trel, sequences=[output_rr], outputs_info=[tensor.alloc(numpy.float32(0.), options['window']+1, options['ydim'], n_samples) , tensor.alloc(numpy.float32(0.), options['window']+1, options['ydim'], n_samples)], go_backwards=True)

    all_ttags_max = all_ttags[0][-1].max(axis=1)
    all_ttags_max2 = all_ttags_max.max(axis=0)
    all_ttags_score2 = all_ttags_max2 + tensor.log(tensor.sum(tensor.sum(tensor.exp(all_ttags[0][-1]-all_ttags_max2.dimshuffle('x', 'x', 0)), axis=1), axis=0))


    final_tscore = (sent_tscore[0][-1] - all_ttags_score2).mean()

    def alltags_trel(d_) : 
        cand = d_.dimshuffle(0, 'x', 1, 'x', 2)+(tparams['A_trel'].dimshuffle(0, 1, 2, 3, 'x') * 1)
        #pred1 = cand.argmax(axis=2)
        #log_sum_exp_val = cand_max + tensor.log(tensor.sum(tensor.exp(cand-cand_max.dimshuffle(0, 1, 'x', 2, 3)), axis=2))
        #pred2 = cand.argmax(axis=0)
        max1, pred1 = theano.tensor.max_and_argmax(cand, axis=2)
        pred2 = max1.argmax(axis=0)
        return pred1, pred2
        
    ttags, updates_alltttags = theano.scan(alltags_trel, sequences=[all_ttags[0]], outputs_info=None, go_backwards=True)
    max_t, pred_tl = tensor.max_and_argmax(all_ttags[0][-1], axis=1)
    pred_tw = max_t.argmax(axis=0)

    #tpred, updatess = theano.scan(lambda x,y,l,w : [x[l], y[w]], sequences=[ttags[0], ttags[1]], outputs_info=[pred_tl, pred_tw], go_backwards=True)
    tpred, updatess = theano.scan(temp_fun, sequences=[ttags[0], ttags[1]], outputs_info=[pred_tl[pred_tw, 0], pred_tw])

    #f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
    #temp_func = theano.function([x, mask, y, all_holder_relations, all_target_relations], [all_tags[1], sent_score[0][-1]], name='temp_func', on_unused_input='ignore')
    f_pred_trel = theano.function([x, mask],  outputs=[tpred[0], tpred[1]], name='f_pred_trel', on_unused_input='ignore')
    cost += -final_tscore

    '''final_L2_norm_A = 0
    L2_norm_A = [(gg**2).sum() for gg in tparams['A_trel'].get_value()]
    for n in L2_norm_A : 
        final_L2_norm_A += n
    cost += 0.1 * final_L2_norm_A

    final_L1_norm_A = 0
    L1_norm_A = [(abs(gg)).sum() for gg in tparams['A_trel'].get_value()]
    for n in L1_norm_A : 
        final_L1_norm_A += n
    cost += 0.1 * final_L1_norm_A'''


    #temp_func = theano.function([x, mask, y, all_holder_relations, all_target_relations, holder_indices, target_indices], [all_tags[1], sent_score[0][-1], final_score, final_hscore, final_tscore], name='temp_func', on_unused_input='ignore')

    temp_func = theano.function([x, mask, y, all_holder_relations, all_target_relations, holder_indices, target_indices], [all_htags[1], sent_score[0][-1], final_score, final_hscore, sent_hscore[0], all_htags[0]], name='temp_func', on_unused_input='ignore')

    return use_noise, x, mask, y, all_holder_relations, all_target_relations, holder_indices, target_indices, f_pred, f_pred_hrel, f_pred_trel, final_output, cost, tparams, temp_func


def sgd(lr, tparams, grads, x, mask, y, cost) :
    """Stochastic Gradient Descent
    :note : A more complicated version of sgd then needed. This is done like that for adadelta and rmsprop.
    """
    #New set of shared variable that will contain the gradient for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    #Function that computes gradients for a mini-batch, but do not updates the weights
    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup, name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]
    
    #Function that updates the weights from the previously computed gradient.
    f_update = theano.function([lr], [], updates=pup, name='sgd_f_update')
    
    return f_grad_shared, f_update


def sgd_w_momentum(lr, tparams, grads, x, mask, y, cost, prev_updates) : 
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    #also check for the norm of the gradients here 

    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup, name='sgd_f_grad_shared')

    pup = [(p, p - (lr * 0.1 * g + 0.7 * pg)) for p, g, pg in zip(tparams.values(), gshared, prev_updates)]

    f_prev_updates = theano.function([], [(lr * g + 0.7 * pg) for p, g, pg in zip(tparams.values(), gshared, prev_updates)]  )


    f_update = theano.function([lr], [], updates = pup, name='sgd_f_update')
    
    return f_grad_shared, f_update, f_prev_updates
    


def adadelta(lr, tparams, grads, x, mask, y, all_holder_relations, all_target_relations, holder_indices, target_indices, cost) : 
    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_grad' % k) 
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.), name = '%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    grad_norm = tensor.sqrt(sum(map(lambda x: tensor.sqr(x).sum(), grads)))
    not_finite = tensor.or_(tensor.isnan(grad_norm), tensor.isinf(grad_norm))
    #grad_norm = tensor.sqrt(grad_norm)
    rescale = 5.
    scaling_num = rescale
    scaling_den = tensor.maximum(rescale, grad_norm)

    #for n, (param, gparam) in enumerate(zip(params, gparams)):
    
    zgup = [(zg, (g*(scaling_num/scaling_den))) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * ((g*(scaling_num/scaling_den)) ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y, all_holder_relations, all_target_relations, holder_indices, target_indices], cost, updates=zgup + rg2up, name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * (zg)
             for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]

    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p+ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up, on_unused_input='ignore', name='adadelta_f_update')

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, x, mask, y, cost) :
    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.), 
                                   name='%s_rgrad' % k)
                     for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name = '%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost, updates = zgup + rgup + rg2up, 
                                    name = 'rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.), 
                           name = '%s_updir' % k)
             for k, p in tparams.iteritems()]

    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads, 
                                            running_grads2)]

    param_up = [(p, p + udn[1]) 
                for p, udn in zip(tparams.values(), updir_new)]

    f_update = theano.function([lr], [], updates = updir_new + param_up, 
                               on_unused_input = 'ignore',
                               name = 'rmsprop_f_update')

    return f_grad_shared, f_update
    
'''def pred_probs(f_pred_prob, prepare_data, data, iterator, verbose=False) : 
    """ If you want to use a trained model, this is useful to compute the probabilities of new example
    """
    n_samples = len(data[0])
    probs = numpy.zeros((n_samples, 2)).astype(config.floatX)
    
    n_done = 0

    for _, valid in iterator : 
        x, mask, y = prepare_data([data[0][t] for t in valid_index], numpy.matrix(data[1])[valid_index], maxlen=None)
        pred_probs = f_pred_prob(x, mask)
        probs[valid_index, :] = pred_probs

        n_done += len(valid_index)
        if verbose : 
            print '%d/%d samples classified '%(n_done, n_samples)

    return probs

'''

#f_pred, prepare_data, train, kf


#f_pred, f_pred_hrel, f_pred_trel, prepare_data, find_relations_lr, data, word_labels, iterator, tparams, window, verbose = False) :


def correct_relations(y, preds) : 
    for j in range(len(y[0])) :
        for i in range(len(y)) :

            if i == 0 :
                if preds[i][j] > 0 and preds[i][j] < 3:
                    preds[i][j] = 1
                elif preds[i][j] > 2 and preds[i][j] < 5:
                    preds[i][j] = 3
                elif preds[i][j] > 4 :
                    preds[i][j] = 5
            
            if preds[i][j] == 0 :
                if (i+1) < len(y)-1 :
                    if preds[i+1][j] > 0 and preds[i+1][j] < 3:
                        preds[i+1][j] = 1
                    elif preds[i+1][j] > 2 and preds[i+1][j] < 5:
                            preds[i+1][j] = 3
                    elif preds[i+1][j] > 4 :
                            preds[i+1][j] = 5

            if preds[i][j] > 0 and preds[i][j] < 3:
                if (i+1) < len(y)-1 :
                    if preds[i+1][j] > 0 and preds[i+1][j] < 3:
                        preds[i+1][j] = 2
            elif preds[i][j] > 2 and preds[i][j] < 5:
                if (i+1) < len(y)-1 :
                    if preds[i+1][j] > 2 :
                        preds[i+1][j] = 4
            elif preds[i][j] > 4:
                if (i+1) < len(y)-1 :
                    if preds[i+1][j] > 4 :
                        preds[i+1][j] = 6

    return preds

def pred_error_relation2(f_pred, f_pred_hrel, f_pred_trel, prepare_data, prepare_data_words, find_relations_lrE, data, word_labels, iterator, tparams, window, relation,  verbose = False) :

    All_pos_exp = 0
    TP_prec_exp = 0
    All_ans_exp = 0
    TP_rec_exp = 0
    
    TP_prec_exp_prop = 0.0
    TP_rec_exp_prop = 0.0
    TP_prec_exp_exact = 0.0
    TP_rec_exp_exact = 0.0


    All_pos_holder = 0
    TP_prec_holder = 0
    All_ans_holder = 0
    TP_rec_holder = 0
    
    TP_prec_holder_prop = 0.0
    TP_rec_holder_prop = 0.0
    TP_prec_holder_exact = 0.0
    TP_rec_holder_exact = 0.0


    All_pos_target = 0
    TP_prec_target = 0
    All_ans_target = 0
    TP_rec_target = 0
    
    TP_prec_target_prop = 0.0
    TP_rec_target_prop = 0.0
    TP_prec_target_exact = 0.0
    TP_rec_target_exact = 0.0


    All_gold_holder = 0
    All_gold_target = 0
    All_pred_holder = 0
    All_pred_target = 0

    TP_holder_prec = 0
    TP_holder_rec = 0
    TP_target_prec = 0
    TP_target_rec = 0

    pred_all_relations_full = []
    pred_all_relations_y_full = []

    gold_holder_relations_full = []
    gold_target_relations_full = []

    predictions = open('./lstm_results_2_'+str(relation), 'w+')                                             

    for _, valid_index in iterator : 
        #print valid_index
        #print data
        #print len(data[2])

        x = [data[0][t] for t in valid_index]
        y = numpy.array(data[1])[valid_index]

        word_y = [word_labels[t] for t in valid_index]
        
        all_holder_relations = []
        all_target_relations = []
        
        holder_dim  = 1
        target_dim = 1

        x, mask, y = prepare_data([data[0][t] for t in valid_index], 
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)

        #print y
        
        #print "==========================="
                
        #preds = []
        
        old_preds = numpy.zeros((len(y), len(y[0])))
        preds = numpy.zeros((len(y), len(y[0])))
        if relation == 1 : 
            #preds = f_pred(x, mask)
            preds, preds_hw = f_pred_hrel(x, mask)
            preds_t, preds_tw = f_pred_trel(x, mask)
            preds_h = copy.deepcopy(preds)
        if relation == 0 : 
            preds = f_pred(x, mask)
            old_preds = copy.deepcopy(preds)
            preds_h, preds_hw = f_pred_hrel(x, mask)
            preds_t, preds_tw = f_pred_trel(x, mask)
            
        if relation == 2 : 
            tpreds1 = f_pred(x, mask)
            preds_h, preds_hw = f_pred_hrel(x, mask)
            preds_t1, preds_tw = f_pred_trel(x, mask)

            tpreds = correct_relations(y, tpreds1)
            preds_t = correct_relations(y, preds_t1)
            old_preds = copy.deepcopy(tpreds1)

            preds = numpy.zeros((len(y), len(y[0])))
            
            for j in range(len(y[0])) :
                for i in range(len(y)) :
                    if preds_h[i][j] > 0 :
                        preds[i][j] = preds_h[i][j]
                    elif tpreds[i][j] == preds_t[i][j] : 
                        preds[i][j] = tpreds[i][j]
                    else : 
                        preds[i][j] = 0

        if relation == 3 : 
            tpreds1 = f_pred(x, mask)
            preds_h, preds_hw = f_pred_hrel(x, mask)
            preds_t1, preds_tw = f_pred_trel(x, mask)

            tpreds = correct_relations(y, tpreds1)
            preds_t = correct_relations(y, preds_t1)
            
            preds = numpy.zeros((len(y), len(y[0])))
            
            for j in range(len(y[0])) :
                for i in range(len(y)) :
                    if tpreds[i][j] == preds_t[i][j] and tpreds[i][j] == preds_h[i][j] : 
                        preds[i][j] == tpreds[i][j]
                    else : 
                        preds[i][j] == 0

        #for k in numpy.arange(len(preds1)) : 
        #    preds.append(viterbi_segment(preds1[k], tparams))
        #print preds
        #x, mask, preds = prepare_data([data[0][t] for t in valid_index], 
        #                         preds,
        #                                                      maxlen=None)
        #print preds
        #print y
        targets = numpy.array(data[1])[valid_index]  #this is most probably is the array of the arrays!
        #is complicated. Either hard_target or the soft_target. Only hard_target implemented for now
        #print y        


        '''x, mask, y = prepare_data([data[0][t] for t in valid_index], 
                                  numpy.array(data[1])[valid_index],
                                                               maxlen=None)
        preds = f_pred(x, mask)
        #print preds
        targets = numpy.array(data[1])[valid_index]  #this is most probably is the array of the arrays!
        #is complicated. Either hard_target or the soft_target. Only hard_target implemented for now
        #print y      

        if "train" in file_data : 
            word_targets = numpy.array(word_labels[0])[valid_index]
        if "valid" in file_data : 
            word_targets = numpy.array(word_labels[1])[valid_index]
        if "test" in file_data : 
            word_targets = numpy.array(word_labels[2])[valid_index]

        temp, mask, gold = prepare_data_words([data[0][t] for t in valid_index],
                                                      word_targets, 
                                                          maxlen=None)'''


        word_targets = numpy.array(word_labels)[valid_index]
        temp, mask, gold = prepare_data_words([data[0][t] for t in valid_index],
                                                      word_targets, 
                                                          maxlen=None)

        for j in range(len(y[0])) :
            for i in range(len(y)) :
                if preds[i][j] > 6 : 
                    preds[i][j] = 0

        for j in range(len(y[0])) :
            for i in range(len(y)) :
                if y[i][j] > 6 : 
                    y[i][j] = 0



        for j in range(len(y[0])) :
            for i in range(len(y)) :
                if preds[i][j] == 0 : 
                    if (i+1) < len(y)-1 : 
                        if preds[i+1][j] > 0 and preds[i+1][j] < 3: 
                            preds[i+1][j] = 1
                        elif preds[i+1][j] > 2 and preds[i+1][j] < 5:
                            preds[i+1][j] = 3
                        elif preds[i+1][j] > 4 : 
                            preds[i+1][j] = 5

                if preds[i][j] > 0 and preds[i][j] < 3:
                    if (i+1) < len(y)-1 : 
                        if preds[i+1][j] > 0 and preds[i+1][j] < 3:
                            preds[i+1][j] = 2
                elif preds[i][j] > 2 and preds[i][j] < 5: 
                    if (i+1) < len(y)-1 : 
                        if preds[i+1][j] > 2 :
                            preds[i+1][j] = 4
                elif preds[i][j] > 4: 
                    if (i+1) < len(y)-1 : 
                        if preds[i+1][j] > 4 :
                            preds[i+1][j] = 6


                            
        new_preds = preds

        for j in range(len(x[0])) : 
            pred_holder = []
            pred_target = []
            pred_expr = []
            
            gold_holder_start = dict()
            gold_holder_end = dict()
            gold_target_start = dict()
            gold_target_end = dict()
            gold_expr_start = dict()
            gold_expr_end = dict()
            
            start_index_h = -1
            end_index_h = -1

            start_index_t = -1
            end_index_t = -1

            start_index_e = -1
            end_index_e = -1
            
            
            for i in range(len(x)) : 
                #if x[i][j] > 0 : 
                #    print str(index_dict[str(x[i][j])])+"\t"+str(gold[i][j])+"\t"+str(preds[i][j])
                if preds[i][j] == 3 and x[i][j] > 0: #beginning of a holder
                    start_index_h = i
                    if i+1 >= len(x) : 
                        pred_holder.append([start_index_h, start_index_h])
                        start_index_h = -1
                        end_index_h = -1
                        break
                    while (i+1) < len(x) and preds[i+1][j] == 4 : 
                        i+=1
                        end_index_h = i

                    if end_index_h == -1 : 
                        end_index_h = start_index_h
                    pred_holder.append([start_index_h, end_index_h])
                    start_index_h = -1
                    end_index_h = -1


                if preds[i][j] == 5 and x[i][j] > 0: 
                    start_index_t = i
                    if i+1 >= len(x) : 
                        pred_target.append([start_index_t, start_index_t])
                        start_index_t = -1
                        end_index_t = -1
                        break

                    while (i+1) < len(x) and preds[i+1][j] == 6 : 
                        i+=1
                        end_index_t = i

                    if end_index_t == -1 : 
                        end_index_t = start_index_t
                    pred_target.append([start_index_t, end_index_t])
                    start_index_t = -1
                    end_index_t = -1


                if preds[i][j] == 1 and x[i][j] > 0: 
                    start_index_e = i
                    if i+1 >= len(x) : 
                        pred_expr.append([start_index_e, start_index_e])
                        start_index_e = -1
                        end_index_e = -1
                        break

                    while (i+1) < len(x) and preds[i+1][j] == 2 : 
                        i+=1
                        end_index_e = i

                    if end_index_e == -1 : 
                        end_index_e = start_index_e
                    pred_expr.append([start_index_e, end_index_e])
                    start_index_e = -1
                    end_index_e = -1

                    
            for i in range(len(x)) : 
                if gold[i][j].startswith("B_AGENT") and len(gold[i][j].split("_")) > 2 : 
                    #print gold[i][j].split("_")
                    rel = []
                    for k in numpy.arange(1, len(gold[i][j].split("_REL"))) :
                        rel.append(int(gold[i][j].split("_REL")[k]))
                        gold_holder_start[rel[-1]] = i
                    
                    if i+1 >= len(x) : 
                        for r in rel : 
                            gold_holder_end[r] = i
                        break
                    
                    while i+1 < len(x) and gold[i+1][j].startswith("AGENT") :
                        i+=1
                        for r in rel : 
                            gold_holder_end[r] = i
                            
                    for r in rel  :
                        if r not in gold_holder_end.keys() : 
                            gold_holder_end[r] = gold_holder_start[r]
                        
                if gold[i][j].startswith("B_TARGET") and len(gold[i][j].split("_")) > 2 : 
                    #print gold[i][j].split("_")
                    #rel = int(gold[i][j].split("_")[-1][3:])
                    #gold_target_start[rel] = i
                    rel = []
                    for k in numpy.arange(1, len(gold[i][j].split("_REL"))) : 
                        rel.append(int(gold[i][j].split("_REL")[k]))
                        gold_target_start[rel[-1]] = i

                    if i+1 >= len(x) :
                        for  r in rel : 
                            gold_target_end[r] = i
                        break

                    while i+1 < len(x) and gold[i+1][j].startswith("TARGET") :
                        i+=1
                        for  r in rel : 
                            gold_target_end[r] = i

                    for r in rel : 
                        if r not in gold_target_end.keys() : 
                            gold_target_end[r] = gold_target_start[r]

                if gold[i][j].startswith("B_DSE") and len(gold[i][j].split("_")) > 2: 
                    #print gold[i][j].split("_")
                    #rel = int(gold[i][j].split("_")[-1][3:])
                    #gold_expr_start[rel] = i

                    rel = []
                    for k in numpy.arange(1, len(gold[i][j].split("_REL"))) : 
                        rel.append(int(gold[i][j].split("_REL")[k]))
                        gold_expr_start[rel[-1]] = i

                    
                    if i+1 >= len(x) : 
                        for r in rel : 
                            gold_expr_end[r] = i
                        break

                    while i+1 < len(x) and gold[i+1][j].startswith("DSE") :
                        i+=1
                        for r in rel : 
                            gold_expr_end[r] = i

                    for r in rel : 
                        if r not in gold_expr_end.keys() : 
                            gold_expr_end[r] = gold_expr_start[r]

            #######Now all the respective annotations are done!!!!

            gold_holder_relations = []
            gold_target_relations = []

            #pred_holder_relations = []
            #pred_target_relations = []

            pred_all_relations = []
            pred_all_relations_y = []

            for key in gold_expr_start.keys() : 
                #print key
                start_e = gold_expr_start[key]
                #print gold_expr_start.keys()
                #print gold_expr_end.keys()
                end_e = gold_expr_end[key]
                if key in gold_holder_start.keys() : 
                    start_h = gold_holder_start[key]
                    end_h = gold_holder_end[key]
                    gold_holder_relations.append([start_h, end_h, start_e, end_e])

                if key in gold_target_start.keys() : 
                    start_t = gold_target_start[key]
                    end_t = gold_target_end[key]
                    gold_target_relations.append([start_t, end_t, start_e, end_e])

            #gold_holder_relations_full.append(gold_holder_relations)
            #gold_target_relations_full.append(gold_target_relations)
            
            #For the predicted all pairs are the ones

            
            holder_index = dict()
            target_index = dict()
            expr_index = dict()
            
            for i in range(len(x)) : 
                holder_index[i] = []
                target_index[i] = []
                expr_index[i] = []


            for i in range(len(x)) : 
                if preds[i][j] > 0  and x[i][j] > 0 : 
                    if preds[i][j] == 1 or preds[i][j] == 2 : 
                        for p in range(len(pred_expr)) :
                            [start_e, end_e] = pred_expr[p]
                            if start_e <= i and i<= end_e : 
                                expr_index[i].append(p)
                    if preds[i][j] == 3 or preds[i][j] == 4 : 
                        for p in range(len(pred_holder)) :
                            [start_h, end_h] = pred_holder[p]
                            if start_h <= i and i<= end_h : 
                                holder_index[i].append(p)
                    if preds[i][j] == 5 or preds[i][j] == 6 : 
                        for p in range(len(pred_target)) :
                            [start_t, end_t] = pred_target[p]
                            if start_t <= i and i<= end_t : 
                                target_index[i].append(p)
            

            #preds_h, preds_hw = f_pred_hrel(x, mask)
            #preds_t, preds_tw = f_pred_trel(x, mask)

            
            def Concat_append(pair, entity) : 
                [start_index, end_index] = pair
                
                for i in range(len(entity)) :
                    [start_e, end_e] = entity[i]
                    if end_e == start_index-1 : 
                        entity[i] = [start_e, end_index]
                        return [start_e, end_index], entity

                    if start_e == end_index+1 : 
                        entity[i] = [start_index, end_e]
                        return [start_index, end_e], entity
                        
                entity.append([start_index, end_index])
                return [start_index, end_index], entity


            pred_holder_relations = []
            pred_target_relations = []

            
            #these many windows to the left
            for i in range(len(x)) : 
                if (i-preds_hw[i][j]) < 0 : 
                    continue
                if preds_hw[i][j] > 0 and x[i][j] > 0 : 
                    if len(expr_index[i]) > 0 : 
                        if (i-preds_hw[i][j]) >= 0 and len(holder_index[i-preds_hw[i][j]]) > 0 :
                            for ei in expr_index[i] : 
                                [start_e, end_e] = pred_expr[ei]
                                for hi in holder_index[i-preds_hw[i][j]] : 
                                    [start_h, end_h] = pred_holder[hi]
                                    if [start_h, end_h, start_e, end_e] not in pred_holder_relations : 
                                        if [start_h, end_h-1, start_e, end_e] in pred_holder_relations : 
                                            pred_holder_relations.remove([start_h, end_h-1, start_e, end_e])
                                        if [start_h, end_h, start_e+1, end_e] in pred_holder_relations : 
                                            pred_holder_relations.remove([start_h, end_h, start_e+1, end_e])

                                        pred_holder_relations.append([start_h, end_h, start_e, end_e])
                        else : 
                            if (i-preds_hw[i][j]) >= 0 and (preds_h[i-preds_hw[i][j]][j] == 3 or preds_h[i-preds_hw[i][j]][j] == 4) : 
                                [start_h, end_h], pred_holder = Concat_append([i-preds_hw[i][j], i-preds_hw[i][j]], pred_holder)
                                new_preds[i-preds_hw[i][j]][j] = preds_h[i-preds_hw[i][j]][j]
                                for ei in expr_index[i] : 
                                    [start_e, end_e] = pred_expr[ei]
                                    if [start_h, end_h, start_e, end_e] not in pred_holder_relations : 
                                        if [start_h, end_h-1, start_e, end_e] in pred_holder_relations : 
                                            pred_holder_relations.remove([start_h, end_h-1, start_e, end_e])
                                        if [start_h, end_h, start_e+1, end_e] in pred_holder_relations : 
                                            pred_holder_relations.remove([start_h, end_h, start_e+1, end_e])

                                        pred_holder_relations.append([start_h, end_h, start_e, end_e])
                                
                    elif preds_h[i][j] == 1 or preds_h[i][j] == 2 :                     
                            if (i-preds_hw[i][j]) >= 0 and len(holder_index[i-preds_hw[i][j]]) > 0 :
                                [start_e, end_e], pred_expr = Concat_append([i, i], pred_expr)
                                new_preds[i][j] = preds_h[i][j]

                                for hi in holder_index[i-preds_hw[i][j]] : 
                                    [start_h, end_h] = pred_holder[hi]
                                    if [start_h, end_h, start_e, end_e] not in pred_holder_relations : 
                                        if [start_h, end_h-1, start_e, end_e] in pred_holder_relations : 
                                            pred_holder_relations.remove([start_h, end_h-1, start_e, end_e])
                                        if [start_h, end_h, start_e+1, end_e] in pred_holder_relations : 
                                            pred_holder_relations.remove([start_h, end_h, start_e+1, end_e])

                                        pred_holder_relations.append([start_h, end_h, start_e, end_e])
                            else : 
                                if (i-preds_hw[i][j]) >= 0 and (preds_h[i-preds_hw[i][j]][j] == 3 or preds_h[i-preds_hw[i][j]][j] == 4) : 
                                    [start_e, end_e], pred_expr = Concat_append([i, i], pred_expr)
                                    new_preds[i][j] = preds_h[i][j]

                                    new_preds[i-preds_hw[i][j]][j] = preds_h[i-preds_hw[i][j]][j]
                                    [start_h, end_h], pred_holder = Concat_append([i-preds_hw[i][j], i-preds_hw[i][j]], pred_holder)
                                    if [start_h, end_h, start_e, end_e] not in pred_holder_relations : 
                                        if [start_h, end_h-1, start_e, end_e] in pred_holder_relations : 
                                            pred_holder_relations.remove([start_h, end_h-1, start_e, end_e])
                                        if [start_h, end_h, start_e+1, end_e] in pred_holder_relations : 
                                            pred_holder_relations.remove([start_h, end_h, start_e+1, end_e])

                                        pred_holder_relations.append([start_h, end_h, start_e, end_e])
                                    
                    elif len(target_index[i]) > 0 : 
                        if (i-preds_hw[i][j]) >= 0 and len(expr_index[i-preds_hw[i][j]]) > 0 :
                            for ti in target_index[i] : 
                                [start_t, end_t] = pred_target[ti]
                                for ei in expr_index[i-preds_hw[i][j]] : 
                                    [start_e, end_e] = pred_expr[ei]
                                    if [start_t, end_t, start_e, end_e] not in pred_target_relations : 
                                        if [start_t, end_t-1, start_e, end_e] in pred_target_relations : 
                                            pred_target_relations.remove([start_t, end_t-1, start_e, end_e])
                                        if [start_t, end_t, start_e+1, end_e] in pred_target_relations : 
                                            pred_target_relations.remove([start_t, end_t, start_e+1, end_e])

                                        pred_target_relations.append([start_t, end_t, start_e, end_e])
                        else : 
                            if (i-preds_hw[i][j]) >= 0 and (preds_h[i-preds_hw[i][j]][j] == 1 or preds_h[i-preds_hw[i][j]][j] == 2) : 
                                [start_e, end_e], pred_expr = Concat_append([i-preds_hw[i][j], i-preds_hw[i][j]], pred_expr)
                                new_preds[i-preds_hw[i][j]][j] = preds_h[i-preds_hw[i][j]][j]
                                for ti in target_index[i] : 
                                    [start_t, end_t] = pred_target[ti]
                                    if [start_t, end_t, start_e, end_e] not in pred_target_relations : 
                                        if [start_t, end_t-1, start_e, end_e] in pred_target_relations : 
                                            pred_target_relations.remove([start_t, end_t-1, start_e, end_e])
                                        if [start_t, end_t, start_e+1, end_e] in pred_target_relations : 
                                            pred_target_relations.remove([start_t, end_t, start_e+1, end_e])

                                        pred_target_relations.append([start_t, end_t, start_e, end_e])
                                
                    elif preds_h[i][j] == 5 or preds_h[i][j] == 6 : 
                            if (i-preds_hw[i][j]) >= 0 and len(expr_index[i-preds_hw[i][j]]) > 0 :
                                [start_t, end_t], pred_target = Concat_append([i, i], pred_target)
                                new_preds[i][j] = preds_h[i][j]

                                for ei in expr_index[i-preds_hw[i][j]] : 
                                    [start_e, end_e] = pred_expr[ei]
                                    if [start_t, end_t, start_e, end_e] not in pred_target_relations : 
                                        if [start_t, end_t-1, start_e, end_e] in pred_target_relations : 
                                            pred_target_relations.remove([start_t, end_t-1, start_e, end_e])
                                        if [start_t, end_t, start_e+1, end_e] in pred_target_relations : 
                                            pred_target_relations.remove([start_t, end_t, start_e+1, end_e])

                                        pred_target_relations.append([start_t, end_t, start_e, end_e])
                            else : 
                                if (i-preds_hw[i][j]) >= 0 and (preds_h[i-preds_hw[i][j]][j] == 1 or preds_h[i-preds_hw[i][j]][j] == 2) : 
                                    [start_t, end_t], pred_target = Concat_append([i, i], pred_target)
                                    new_preds[i][j] = preds_h[i][j]

                                    new_preds[i-preds_hw[i][j]][j] = preds_h[i-preds_hw[i][j]][j]
                                    [start_e, end_e], pred_expr = Concat_append([i-preds_hw[i][j], i-preds_hw[i][j]], pred_expr)
                                    if [start_t, end_t, start_e, end_e] not in pred_target_relations : 
                                        if [start_t, end_t-1, start_e, end_e] in pred_target_relations : 
                                            pred_target_relations.remove([start_t, end_t-1, start_e, end_e])
                                        if [start_t, end_t, start_e+1, end_e] in pred_target_relations : 
                                            pred_target_relations.remove([start_t, end_t, start_e+1, end_e])

                                        pred_target_relations.append([start_t, end_t, start_e, end_e])



            for i in range(len(x)) : 
                if (i+preds_tw[i][j]) >= len(x) : 
                    continue
                if preds_tw[i][j] > 0 and x[i][j] > 0 : 
                    if len(expr_index[i]) > 0 : 
                        if (i+preds_tw[i][j]) < len(x)  and len(holder_index[i+preds_tw[i][j]]) > 0 :
                            for ei in expr_index[i] : 
                                [start_e, end_e] = pred_expr[ei]
                                for hi in holder_index[i+preds_tw[i][j]] : 
                                    [start_h, end_h] = pred_holder[hi]
                                    if [start_h, end_h, start_e, end_e] not in pred_holder_relations : 
                                        if [start_h, end_h-1, start_e, end_e] in pred_holder_relations : 
                                            pred_holder_relations.remove([start_h, end_h-1, start_e, end_e])
                                        if [start_h, end_h, start_e+1, end_e] in pred_holder_relations : 
                                            pred_holder_relations.remove([start_h, end_h, start_e+1, end_e])

                                        pred_holder_relations.append([start_h, end_h, start_e, end_e])
                        else : 
                            if (i+preds_tw[i][j]) < len(x) and (preds_t[i+preds_tw[i][j]][j] == 3 or preds_t[i+preds_tw[i][j]][j] == 4) : 
                                [start_h, end_h], pred_holder = Concat_append([i+preds_tw[i][j], i+preds_tw[i][j]], pred_holder)
                                new_preds[i+preds_tw[i][j]][j] = preds_t[i+preds_tw[i][j]][j]
                                for ei in expr_index[i] : 
                                    [start_e, end_e] = pred_expr[ei]
                                    if [start_h, end_h, start_e, end_e] not in pred_holder_relations : 
                                        if [start_h, end_h-1, start_e, end_e] in pred_holder_relations : 
                                            pred_holder_relations.remove([start_h, end_h-1, start_e, end_e])
                                        if [start_h, end_h, start_e+1, end_e] in pred_holder_relations : 
                                            pred_holder_relations.remove([start_h, end_h, start_e+1, end_e])

                                        pred_holder_relations.append([start_h, end_h, start_e, end_e])
                                
                    elif preds_t[i][j] == 1 or preds_t[i][j] == 2 :                     
                            if (i+preds_tw[i][j]) < len(x) and len(expr_index[i+preds_tw[i][j]]) > 0 :
                                [start_e, end_e], pred_expr = Concat_append([i, i], pred_expr)
                                new_preds[i][j] = preds_h[i][j]

                                for hi in holder_index[i+preds_tw[i][j]] : 
                                    [start_h, end_h] = pred_holder[hi]
                                    if [start_h, end_h, start_e, end_e] not in pred_holder_relations :
                                        if [start_h, end_h-1, start_e, end_e] in pred_holder_relations : 
                                            pred_holder_relations.remove([start_h, end_h-1, start_e, end_e])
                                        if [start_h, end_h, start_e+1, end_e] in pred_holder_relations : 
                                            pred_holder_relations.remove([start_h, end_h, start_e+1, end_e])
 
                                        pred_holder_relations.append([start_h, end_h, start_e, end_e])
                            else : 
                                if (i+preds_tw[i][j]) < len(x) and (preds_t[i+preds_tw[i][j]][j] == 3 or preds_t[i+preds_tw[i][j]][j] == 4) : 
                                    [start_e, end_e], pred_expr = Concat_append([i, i], pred_expr)
                                    new_preds[i][j] = preds_h[i][j]

                                    new_preds[i+preds_tw[i][j]][j] = preds_t[i+preds_tw[i][j]][j]
                                    [start_h, end_h], pred_holder = Concat_append([i+preds_tw[i][j], i+preds_tw[i][j]], pred_holder)
                                    if [start_h, end_h, start_e, end_e] not in pred_holder_relations : 
                                        if [start_h, end_h-1, start_e, end_e] in pred_holder_relations : 
                                            pred_holder_relations.remove([start_h, end_h-1, start_e, end_e])
                                        if [start_h, end_h, start_e+1, end_e] in pred_holder_relations : 
                                            pred_holder_relations.remove([start_h, end_h, start_e+1, end_e])

                                        pred_holder_relations.append([start_h, end_h, start_e, end_e])
                                    
                    elif len(target_index[i]) > 0 : 
                        if (i+preds_tw[i][j]) < len(x) and len(expr_index[i+preds_tw[i][j]]) > 0 :
                            for ti in target_index[i] : 
                                [start_t, end_t] = pred_target[ti]
                                for ei in expr_index[i+preds_tw[i][j]] : 
                                    [start_e, end_e] = pred_expr[ei]
                                    if [start_t, end_t, start_e, end_e] not in pred_target_relations : 
                                        if [start_t, end_t-1, start_e, end_e] in pred_target_relations : 
                                            pred_target_relations.remove([start_t, end_t-1, start_e, end_e])
                                        if [start_t, end_t, start_e+1, end_e] in pred_target_relations : 
                                            pred_target_relations.remove([start_t, end_t, start_e+1, end_e])

                                        pred_target_relations.append([start_t, end_t, start_e, end_e])
                        else : 
                            if (i+preds_tw[i][j]) < len(x) and (preds_t[i+preds_tw[i][j]][j] == 1 or preds_t[i+preds_tw[i][j]][j] == 2) : 
                                [start_e, end_e], pred_expr = Concat_append([i+preds_tw[i][j], i+preds_tw[i][j]], pred_expr)
                                new_preds[i+preds_tw[i][j]][j] = preds_t[i+preds_tw[i][j]][j]
                                for ti in target_index[i] : 
                                    [start_t, end_t] = pred_target[ti]
                                    if [start_t, end_t, start_e, end_e] not in pred_target_relations : 
                                        if [start_t, end_t-1, start_e, end_e] in pred_target_relations : 
                                            pred_target_relations.remove([start_t, end_t-1, start_e, end_e])
                                        if [start_t, end_t, start_e+1, end_e] in pred_target_relations : 
                                            pred_target_relations.remove([start_t, end_t, start_e+1, end_e])
                                        pred_target_relations.append([start_t, end_t, start_e, end_e])
                                
                    elif preds_t[i][j] == 5 or preds_t[i][j] == 6 : 

                            if (i+preds_tw[i][j]) < len(x) and len(expr_index[i+preds_tw[i][j]]) > 0 :
                                [start_t, end_t], pred_target = Concat_append([i, i], pred_target)
                                new_preds[i][j] = preds_t[i][j]

                                for ei in expr_index[i+preds_tw[i][j]] : 
                                    [start_e, end_e] = pred_expr[ei]
                                    if [start_t, end_t, start_e, end_e] not in pred_target_relations : 
                                        if [start_t, end_t-1, start_e, end_e] in pred_target_relations : 
                                            pred_target_relations.remove([start_t, end_t-1, start_e, end_e])
                                        if [start_t, end_t, start_e+1, end_e] in pred_target_relations : 
                                            pred_target_relations.remove([start_t, end_t, start_e+1, end_e])

                                        pred_target_relations.append([start_t, end_t, start_e, end_e])
                            else : 
                                if (i+preds_tw[i][j]) < len(x) and (preds_t[i+preds_tw[i][j]][j] == 1 or preds_t[i+preds_tw[i][j]][j] == 2) : 
                                    [start_t, end_t], pred_target = Concat_append([i, i], pred_target)
                                    new_preds[i][j] = preds_t[i][j]

                                    new_preds[i+preds_tw[i][j]][j] = preds_t[i+preds_tw[i][j]][j]
                                    [start_e, end_e], pred_expr = Concat_append([i+preds_tw[i][j], i+preds_tw[i][j]], pred_expr)
                                    if [start_t, end_t, start_e, end_e] not in pred_target_relations : 
                                        if [start_t, end_t-1, start_e, end_e] in pred_target_relations : 
                                            pred_target_relations.remove([start_t, end_t-1, start_e, end_e])
                                        if [start_t, end_t, start_e+1, end_e] in pred_target_relations : 
                                            pred_target_relations.remove([start_t, end_t, start_e+1, end_e])

                                        pred_target_relations.append([start_t, end_t, start_e, end_e])
                    


            '''for [start_e, end_e] in pred_expr : 
                for [start_h, end_h] in pred_holder : 
                    pred_all_relations.append([start_h, end_h, start_e, end_e])
                    pred_all_relations_y.append(1)

                for [start_t, end_t] in pred_target : 
                    pred_all_relations.append([start_t, end_t, start_e, end_e])
                    pred_all_relations_y.append(2)'''

            
            #pred_all_relations_full.append(pred_all_relations)
            #pred_all_relations_y_full.append(pred_all_relations_y)


            #print "Gold=========================================="
            #print pred_holder_relations
            #print gold_holder_relations
            #print gold_target_relations

            
            #print "Pred=========================================="
            #print pred_holder_relations
            #print pred_all_relations
            #print pred_all_relations_y

            
            #print pred_all_relations
            #print pred_all_relations_y

            #if len(pred_all_relations) == 0 : 
            #    temp = []
            #    pred_all_relations.append(temp)

            #pred_all_relations_full = []
            #pred_all_relations_full_y = []

            #if not len(pred_all_relations[0]) == 0 : 
            #pred_all_relations_full.append(pred_all_relations)
            #pred_all_relations_full_y.append(pred_all_relations_y)

            #print "Before===="
            #print pred_all_relations_full
            #print pred_all_relations_full_y
            
            #relations_x, mask_t, relations_y = prepare_data_relations(pred_all_relations_full, pred_all_relations_full_y)

            #print "After relations prep==="
            #print relations_x
            #print relations_y

            #pred_relations = f_pred_relations(x, mask, relations_x)

            #print pred_relations

            #pred_holder_relations = []
            #pred_target_relations = []

            '''for l in range(len(pred_relations[0])) :
                for k in range(len(pred_relations)) :
                    if pred_relations[k][l] == 1 : 
                        pred_holder_relations.append(relations_x[k][l])
                    if pred_relations[k][l] == 2 : 
                        pred_target_relations.append(relations_x[k][l])
                        #pred_holder_relations_all.append(pred_holder_relations)
                        #pred_target_relations_all.append(pred_target_relations)'''
            
            #print "-----------------------------------------------------------------"
            #print gold_holder_relations
            #print gold_target_relations
            #print "Predicted relation=========================================="
            #print pred_holder_relations
            #print pred_target_relations


            #preds = new_preds

            pred_holder_later = []
            for [sh1, eh1, se1, ee1] in pred_holder_relations : 
                for [sh2, eh2, se2, ee2] in pred_holder_later : 
                    if sh1==sh2 and eh1 == eh2 and se1 == se2 and ee1 == ee2 : 
                        break
                pred_holder_later.append([sh1, eh1, se1, ee1])

            pred_holder_relations = pred_holder_later

            pred_target_later = []
            for [sh1, eh1, se1, ee1] in pred_target_relations : 
                for [sh2, eh2, se2, ee2] in pred_target_later : 
                    if sh1==sh2 and eh1 == eh2 and se1 == se2 and ee1 == ee2 : 
                        break
                pred_target_later.append([sh1, eh1, se1, ee1])

            pred_target_relations = pred_target_later
            
            
            
            final_tag = []
            for i in range(len(x)) :
                if preds[i][j] == 1 :
                    final_tag.append("B_DSE")
                if preds[i][j] == 2 :
                    final_tag.append("DSE")
                if preds[i][j] == 3 :
                    final_tag.append("B_AGENT")
                if preds[i][j] == 4 :
                    final_tag.append("AGENT")
                if preds[i][j] == 5 :
                    final_tag.append("B_TARGET")
                if preds[i][j] == 6 :
                    final_tag.append("TARGET")
                if preds[i][j] == 0 :
                    final_tag.append("O")
                if preds[i][j] == 7 : 
                    final_tag.append("B_ESE")
                if preds[i][j] == 8 : 
                    final_tag.append("ESE")
                if preds[i][j] == 9 :
                    final_tag.append("B_OBJ")
                if preds[i][j] == 10 : 
                    final_tag.append("OBJ")
            
            count_h = 1
            count_t = 1
            
            for [sh, eh, se, ee] in pred_holder_relations : 
                for ind_h in numpy.arange(sh, eh+1) : 
                    final_tag[ind_h] = final_tag[ind_h]+"_-"+str(count_h)
                for ind_e in numpy.arange(se, ee+1) : 
                    final_tag[ind_e] = final_tag[ind_e]+"_-"+str(count_h)
                count_h+=1
            
            #count_t = 1
            for [st, et, se, ee] in pred_target_relations :
                for ind_t in numpy.arange(st, et+1) :
                    final_tag[ind_t] = final_tag[ind_t]+"_"+str(count_t)
                for ind_e in numpy.arange(se, ee+1) :
                    final_tag[ind_e] = final_tag[ind_e]+"_"+str(count_t)
                count_t+=1

            #remove all the ones with no relations at place of holders and targets!!

            '''for i in range(len(x)) : 
                if final_tag[i] == 'B_AGENT' or final_tag[i] == 'AGENT' or final_tag[i] == 'B_TARGET' or final_tag[i] == 'TARGET': 
                    final_tag[i] = 'O'
                    preds[i][j] = 0'''


            for i in range(len(x)) :
                predictions.write(str(index_dict[str(x[i][j])]))
                predictions.write("\t")
                '''predictions.write(str(y[i][j]))
                predictions.write("\t")
                predictions.write(str(preds[i][j]))
                predictions.write("\t")
                predictions.write(str(old_preds[i][j]))
                predictions.write("\t")
                predictions.write(str(preds_h[i][j]))
                predictions.write("\t")
                predictions.write(str(preds_t[i][j]))
                predictions.write("\t")
                predictions.write(str(preds_hw[i][j]))
                predictions.write("\t")
                predictions.write(str(preds_tw[i][j]))
                predictions.write("\t")'''
                predictions.write(str(word_y[0][i]))
                predictions.write("\t")
                predictions.write(str(final_tag[i]))

                predictions.write("\n")
            #predictions.write(str(word_y))
            #predictions.write("\n")
            predictions.write("\n")


            for [pstart_h, pend_h, pstart_e, pend_e] in pred_holder_relations : 
                for [gstart_h, gend_h, gstart_e, gend_e] in gold_holder_relations : 
                    list1 = numpy.arange(pstart_h, pend_h+1)
                    list2 = numpy.arange(gstart_h, gend_h+1)

                    list3 = numpy.arange(pstart_e, pend_e+1)
                    list4 = numpy.arange(gstart_e, gend_e+1)


                    if len(list(set(list1) & set(list2))) > 0 and len(list(set(list3) & set(list4))) > 0 :
                        TP_holder_prec+=1


            for [gstart_h, gend_h, gstart_e, gend_e] in gold_holder_relations : 
                for [pstart_h, pend_h, pstart_e, pend_e] in pred_holder_relations : 
                    list1 = numpy.arange(pstart_h, pend_h+1)
                    list2 = numpy.arange(gstart_h, gend_h+1)

                    list3 = numpy.arange(pstart_e, pend_e+1)
                    list4 = numpy.arange(gstart_e, gend_e+1)

                    if len(list(set(list1) & set(list2))) > 0 and len(list(set(list3) & set(list4))) > 0 :
                        TP_holder_rec+=1

            for [pstart_t, pend_t, pstart_e, pend_e] in pred_target_relations : 
                for [gstart_t, gend_t, gstart_e, gend_e] in gold_target_relations : 
                    list1 = numpy.arange(pstart_t, pend_t+1)
                    list2 = numpy.arange(gstart_t, gend_t+1)

                    list3 = numpy.arange(pstart_e, pend_e+1)
                    list4 = numpy.arange(gstart_e, gend_e+1)

                    if len(list(set(list1) & set(list2))) > 0 and len(list(set(list3) & set(list4))) > 0 : 
                        TP_target_prec+=1

            for [gstart_t, gend_t, gstart_e, gend_e] in gold_target_relations : 
                for [pstart_t, pend_t, pstart_e, pend_e] in pred_target_relations : 
                    list1 = numpy.arange(pstart_t, pend_t+1)
                    list2 = numpy.arange(gstart_t, gend_t+1)

                    list3 = numpy.arange(pstart_e, pend_e+1)
                    list4 = numpy.arange(gstart_e, gend_e+1)

                    if len(list(set(list1) & set(list2))) > 0 and len(list(set(list3) & set(list4))) > 0 :
                        TP_target_rec+=1

            All_gold_holder += len(gold_holder_relations)
            All_gold_target += len(gold_target_relations)
            All_pred_holder += len(pred_holder_relations)
            All_pred_target += len(pred_target_relations)
            
            #print "HOlder\t"+str(temp_holder_pred)
            #print "Gold\t"+str(temp_holder_gold)
        
        #preds = new_preds
        

        #########################################################Relation part!!!###########################################

        for j in range(len(y[0])) :
            for i in range(len(y)) :
                if preds[i][j] == 0 : 
                    if (i+1) < len(y)-1 : 
                        if preds[i+1][j] > 0 and preds[i+1][j] < 3: 
                            preds[i+1][j] = 1
                        elif preds[i+1][j] > 2 and preds[i+1][j] < 5:
                            preds[i+1][j] = 3
                        elif preds[i+1][j] > 4 : 
                            preds[i+1][j] = 5

                if preds[i][j] > 0 and preds[i][j] < 3:
                    if (i+1) < len(y)-1 : 
                        if preds[i+1][j] > 0 and preds[i+1][j] < 3:
                            preds[i+1][j] = 2
                elif preds[i][j] > 2 and preds[i][j] < 5: 
                    if (i+1) < len(y)-1 : 
                        if preds[i+1][j] > 2 :
                            preds[i+1][j] = 4
                elif preds[i][j] > 4: 
                    if (i+1) < len(y)-1 : 
                        if preds[i+1][j] > 4 :
                            preds[i+1][j] = 6


        for j in range(len(x[0])) : 

            pred_holder = []
            pred_target = []
            pred_expr = []
                        
            start_index_h = -1
            end_index_h = -1

            start_index_t = -1
            end_index_t = -1

            start_index_e = -1
            end_index_e = -1
            
            
            for i in range(len(x)) : 
                #if x[i][j] > 0 : 
                #    print str(index_dict[str(x[i][j])])+"\t"+str(gold[i][j])+"\t"+str(preds[i][j])
                if preds[i][j] == 3 and x[i][j] > 0: #beginning of a holder
                    start_index_h = i
                    if i+1 >= len(x) : 
                        pred_holder.append([start_index_h, start_index_h])
                        start_index_h = -1
                        end_index_h = -1
                        break
                    while (i+1) < len(x) and preds[i+1][j] == 4 : 
                        i+=1
                        end_index_h = i

                    if end_index_h == -1 : 
                        end_index_h = start_index_h
                    pred_holder.append([start_index_h, end_index_h])
                    start_index_h = -1
                    end_index_h = -1


                if preds[i][j] == 5 and x[i][j] > 0: 
                    start_index_t = i
                    if i+1 >= len(x) : 
                        pred_target.append([start_index_t, start_index_t])
                        start_index_t = -1
                        end_index_t = -1
                        break

                    while (i+1) < len(x) and preds[i+1][j] == 6 : 
                        i+=1
                        end_index_t = i

                    if end_index_t == -1 : 
                        end_index_t = start_index_t
                    pred_target.append([start_index_t, end_index_t])
                    start_index_t = -1
                    end_index_t = -1


                if preds[i][j] == 1 and x[i][j] > 0: 
                    start_index_e = i
                    if i+1 >= len(x) : 
                        pred_expr.append([start_index_e, start_index_e])
                        start_index_e = -1
                        end_index_e = -1
                        break

                    while (i+1) < len(x) and preds[i+1][j] == 2 : 
                        i+=1
                        end_index_e = i

                    if end_index_e == -1 : 
                        end_index_e = start_index_e
                    pred_expr.append([start_index_e, end_index_e])
                    start_index_e = -1
                    end_index_e = -1


            gold_holder = []
            gold_target = []
            gold_expr = []
                        
            start_index_h = -1
            end_index_h = -1

            start_index_t = -1
            end_index_t = -1

            start_index_e = -1
            end_index_e = -1
            
            
            for i in range(len(x)) : 
                #if x[i][j] > 0 : 
                #    print str(index_dict[str(x[i][j])])+"\t"+str(gold[i][j])+"\t"+str(preds[i][j])
                if y[i][j] == 3 and x[i][j] > 0: #beginning of a holder
                    start_index_h = i
                    if i+1 >= len(x) : 
                        gold_holder.append([start_index_h, start_index_h])
                        start_index_h = -1
                        end_index_h = -1
                        break
                    while (i+1) < len(x) and y[i+1][j] == 4 : 
                        i+=1
                        end_index_h = i

                    if end_index_h == -1 : 
                        end_index_h = start_index_h
                    gold_holder.append([start_index_h, end_index_h])
                    start_index_h = -1
                    end_index_h = -1


                if y[i][j] == 5 and x[i][j] > 0: 
                    start_index_t = i
                    if i+1 >= len(x) : 
                        gold_target.append([start_index_t, start_index_t])
                        start_index_t = -1
                        end_index_t = -1
                        break

                    while (i+1) < len(x) and y[i+1][j] == 6 : 
                        i+=1
                        end_index_t = i

                    if end_index_t == -1 : 
                        end_index_t = start_index_t
                    gold_target.append([start_index_t, end_index_t])
                    start_index_t = -1
                    end_index_t = -1


                if y[i][j] == 1 and x[i][j] > 0: 
                    start_index_e = i
                    if i+1 >= len(x) : 
                        gold_expr.append([start_index_e, start_index_e])
                        start_index_e = -1
                        end_index_e = -1
                        break

                    while (i+1) < len(x) and y[i+1][j] == 2 : 
                        i+=1
                        end_index_e = i

                    if end_index_e == -1 : 
                        end_index_e = start_index_e
                    gold_expr.append([start_index_e, end_index_e])
                    start_index_e = -1
                    end_index_e = -1


            #do analysis and add to the global counts
            #TP_prec_target_exact = 0.0
            #TP_rec_target_exact = 0.0

            
            for [pstarte, pende] in pred_holder : 
                for [gstarte, gende] in gold_holder : 
                    if pstarte == gstarte and pende == gende : 
                        TP_prec_holder_exact += 1
                        TP_rec_holder_exact += 1
                        break

            for [pstarte, pende] in pred_target : 
                for [gstarte, gende] in gold_target : 
                    if pstarte == gstarte and pende == gende : 
                        TP_prec_target_exact += 1
                        TP_rec_target_exact += 1
                        break

            for [pstarte, pende] in pred_expr : 
                for [gstarte, gende] in gold_expr : 
                    if pstarte == gstarte and pende == gende : 
                        TP_prec_exp_exact += 1
                        TP_rec_exp_exact += 1
                        break
            
            
            
        for j in range(len(y[0])) :
            list_expr = []
            num_overlap = 0
            total_expr = 0

            gold_holders = []
            gold_targets = []
            gold_exp = []

            gold_present_pred_holder = []
            gold_present_pred_target = []
            gold_present_pred_exp = []

            gold_present_pred_holder_prop = []
            gold_present_pred_target_prop = []
            gold_present_pred_exp_prop = []



            gold_present_holder = []
            gold_present_target = []
            gold_present_exp = []

            current_holder_gold = []
            current_target_gold = []
            current_exp_gold = []

            pred_holders = []
            pred_targets = []
            pred_exp = []

            pred_present_gold_holder = []
            pred_present_gold_target = []
            pred_present_gold_exp = []

            pred_present_gold_holder_prop = []
            pred_present_gold_target_prop = []
            pred_present_gold_exp_prop = []

            pred_present_holder = []
            pred_present_target = []
            pred_present_exp = []

            current_holder_pred = []
            current_target_pred = []
            current_exp_pred = []


            for i in range(len(y)) :

                #So preds[i][j] contains the true labels and y[i][j] contains the predicted labels
                if y[i][j] == 1 and x[i][j] > 0 : 
                    #start another and keep this in the gold_holders
                    if len(current_exp_gold) > 0 : 
                        gold_exp.append(current_exp_gold)
                        if sum(gold_present_exp) > 0 :
                            gold_present_pred_exp.append(1)
                            gold_present_pred_exp_prop.append(float(sum(gold_present_exp))/len(gold_present_exp))
                        else : 
                            gold_present_pred_exp.append(0)
                            gold_present_pred_exp_prop.append(0.0)

                    gold_present_exp = []
                    current_exp_gold = []
                    if preds[i][j] == 1 or preds[i][j] == 2 : 
                        gold_present_exp.append(1)
                    else : 
                        gold_present_exp.append(0)
                    current_exp_gold.append(i)
                if y[i][j] == 2 and x[i][j] > 0 : 
                    if len(current_exp_gold) > 0 and current_exp_gold[-1] == i-1 : 
                        #Should never be the case!! but still need to troubleshoot it!
                        if preds[i][j] == 1 or preds[i][j] == 2 : 
                            gold_present_exp.append(1)
                        else : 
                            gold_present_exp.append(0)

                        current_exp_gold.append(i)
                    else : 
                        if len(current_exp_gold) > 0 : 
                            gold_exp.append(current_exp_gold)
                            if sum(gold_present_exp) > 0 :
                                gold_present_pred_exp.append(1)
                                gold_present_pred_exp_prop.append(float(sum(gold_present_exp))/len(gold_present_exp))

                            else : 
                                gold_present_pred_exp.append(0)
                                gold_present_pred_exp_prop.append(0.0)

                        gold_present_exp = []

                        current_exp_gold = []

                        if preds[i][j] == 1 or preds[i][j] == 2 : 
                            gold_present_exp.append(1)
                        else : 
                            gold_present_exp.append(0)

                        current_exp_gold.append(i)

                        
                if y[i][j] == 3 and x[i][j] > 0 : 
                    #start another and keep this in the gold_holders
                    if len(current_holder_gold) > 0 : 
                        gold_holders.append(current_holder_gold)
                        if sum(gold_present_holder) > 0 :
                            gold_present_pred_holder.append(1)
                            gold_present_pred_holder_prop.append(float(sum(gold_present_holder))/len(gold_present_holder))
                            
                        else : 
                            gold_present_pred_holder.append(0)
                            gold_present_pred_exp_prop.append(0.0)


                    gold_present_holder = []

                    current_holder_gold = []
                    
                    if preds[i][j] == 3 or preds[i][j] == 4 : 
                        gold_present_holder.append(1)
                    else : 
                        gold_present_holder.append(0)


                    current_holder_gold.append(i)
                if y[i][j] == 4 and x[i][j] > 0 : 
                    if len(current_holder_gold) > 0 and current_holder_gold[-1] == i-1 : 
                        #Should never be the case!! but still need to troubleshoot it!
                        if preds[i][j] == 3 or preds[i][j] == 4 : 
                            gold_present_holder.append(1)
                        else : 
                            gold_present_holder.append(0)

                        current_holder_gold.append(i)
                    else : 
                        if len(current_holder_gold) > 0 : 
                            gold_holders.append(current_holder_gold)
                            if sum(gold_present_holder) > 0 :
                                gold_present_pred_holder.append(1)
                                gold_present_pred_holder_prop.append(float(sum(gold_present_holder))/len(gold_present_holder))
                                
                            else : 
                                gold_present_pred_holder.append(0)
                                gold_present_pred_holder_prop.append(0.0)


                        gold_present_holder = []

                        current_holder_gold = []

                        if preds[i][j] == 3 or preds[i][j] == 4 : 
                            gold_present_holder.append(1)
                        else : 
                            gold_present_holder.append(0)

                        current_holder_gold.append(i)
                
                        
                if y[i][j] == 5 and x[i][j] > 0 : 
                    #start another and keep this in the gold_holders
                    if len(current_target_gold) > 0 : 
                        gold_targets.append(current_target_gold)
                        if sum(gold_present_target) > 0 :
                            gold_present_pred_target.append(1)
                            gold_present_pred_target_prop.append(float(sum(gold_present_target))/len(gold_present_target))
                            
                        else : 
                            gold_present_pred_target.append(0)
                            gold_present_pred_target_prop.append(0.0)


                    gold_present_target = []

                    current_target_gold = []
                    
                    if preds[i][j] == 5 or preds[i][j] == 6 : 
                        gold_present_target.append(1)
                    else : 
                        gold_present_target.append(0)

                    current_target_gold.append(i)

                if y[i][j] == 6 and x[i][j] > 0 : 
                    if len(current_target_gold) > 0 and current_target_gold[-1] == i-1 : 
                        #Should never be the case!! but till need to troubleshoot it!
                        if preds[i][j] == 5 or preds[i][j] == 6 : 
                            gold_present_target.append(1)
                        else : 
                            gold_present_target.append(0)

                        current_target_gold.append(i)
                    else : 

                        if len(current_target_gold) > 0 : 
                            gold_targets.append(current_target_gold)

                            if sum(gold_present_target) > 0 :
                                gold_present_pred_target.append(1)
                                gold_present_pred_target_prop.append(float(sum(gold_present_target))/len(gold_present_target))
                                
                            else : 
                                gold_present_pred_target.append(0)
                                gold_present_pred_target_prop.append(0.0)
                                                    
                        gold_present_target = []

                        current_holder_gold = []

                        if preds[i][j] == 5 or preds[i][j] == 6 : 
                            gold_present_target.append(1)
                        else : 
                            gold_present_target.append(0)
                        
                        current_target_gold.append(i)


                if preds[i][j] == 1 and x[i][j] > 0 : 
                    #start another and keep this in the gold_holders
                    if len(current_exp_pred) > 0 : 
                        pred_exp.append(current_exp_pred)
                        if sum(pred_present_exp) > 0 :
                            pred_present_gold_exp.append(1)
                            pred_present_gold_exp_prop.append(float(sum(pred_present_exp))/len(pred_present_exp))
                            
                        else : 
                            pred_present_gold_exp.append(0)
                            pred_present_gold_exp_prop.append(0.0)


                    pred_present_exp = []
                    current_exp_pred = []
                    
                    if y[i][j] == 1 or y[i][j] == 2 : 
                        pred_present_exp.append(1)
                    else : 
                        pred_present_exp.append(0)

                    current_exp_pred.append(i)
                if preds[i][j] == 2 and x[i][j] > 0 : 
                    if len(current_exp_pred) > 0 and current_exp_pred[-1] == i-1 : 
                        #Should never be the case!! but still need to troubleshoot it!
                        if y[i][j] == 1 or y[i][j] == 2 : 
                            pred_present_exp.append(1)
                        else : 
                            pred_present_exp.append(0)

                        current_exp_pred.append(i)
                    else : 
                        if len(current_exp_pred) > 0 : 
                            pred_exp.append(current_exp_pred)
                            if sum(pred_present_exp) > 0 :
                                pred_present_gold_exp.append(1)
                                pred_present_gold_exp_prop.append(float(sum(pred_present_exp))/len(pred_present_exp))
                            else : 
                                pred_present_gold_exp.append(0)
                                pred_present_gold_exp_prop.append(0.0)

                        pred_present_exp = []

                        current_exp_pred = []

                        if y[i][j] == 1 or y[i][j] == 2 : 
                            pred_present_exp.append(1)
                        else : 
                            pred_present_exp.append(0)

                        current_exp_pred.append(i)

                        
                if preds[i][j] == 3 and x[i][j] > 0 : 
                    #start another and keep this in the gold_holders
                    if len(current_holder_pred) > 0 : 
                        pred_holders.append(current_holder_pred)
                        if sum(pred_present_holder) > 0 :
                            pred_present_gold_holder.append(1)
                            pred_present_gold_holder_prop.append(float(sum(pred_present_holder))/len(pred_present_holder))
                        else : 
                            pred_present_gold_holder.append(0)
                            pred_present_gold_holder_prop.append(0.0)

                    pred_present_holder = []

                    current_holder_pred = []
                    
                    if y[i][j] == 3 or y[i][j] == 4 : 
                        pred_present_holder.append(1)
                    else : 
                        pred_present_holder.append(0)

                    current_holder_pred.append(i)
                if preds[i][j] == 4 and x[i][j] > 0 : 
                    if len(current_holder_pred) > 0 and current_holder_pred[-1] == i-1 : 
                        #Should never be the case!! but still need to troubleshoot it!
                        current_holder_pred.append(i)
                        if y[i][j] == 3 or y[i][j] == 4 : 
                            pred_present_holder.append(1)
                        else : 
                            pred_present_holder.append(0)

                    else : 
                        if len(current_holder_pred) > 0 : 
                            pred_holders.append(current_holder_pred)
                            if sum(pred_present_holder) > 0 :
                                pred_present_gold_holder.append(1)
                                pred_present_gold_holder_prop.append(float(sum(pred_present_holder))/len(pred_present_holder))
                                
                            else : 
                                pred_present_gold_holder.append(0)
                                pred_present_gold_exp_prop.append(0.0)


                        pred_present_holder = []

                        current_holder_pred = []
                        if y[i][j] == 3 or y[i][j] == 4 : 
                            pred_present_holder.append(1)
                        else : 
                            pred_present_holder.append(0)

                        current_holder_pred.append(i)
                
                        
                if preds[i][j] == 5 and x[i][j] > 0 : 
                    #start another and keep this in the gold_holders
                    if len(current_target_pred) > 0 : 
                        pred_targets.append(current_target_pred)
                        if sum(pred_present_target) > 0 :
                            pred_present_gold_target.append(1)
                            pred_present_gold_target_prop.append(float(sum(pred_present_target))/len(pred_present_target))
                            
                        else : 
                            pred_present_gold_target.append(0)
                            pred_present_gold_target_prop.append(0.0)


                    pred_present_target = []

                    current_target_pred = []
                    if y[i][j] == 5 or y[i][j] == 6 : 
                        pred_present_target.append(1)
                    else : 
                        pred_present_target.append(0)

                    current_target_pred.append(i)
                if preds[i][j] == 6 and x[i][j] > 0 : 
                    if len(current_target_pred) > 0 and current_target_pred[-1] == i-1 : 
                        #Should never be the case!! but still need to troubleshoot it!
                        if y[i][j] == 5 or y[i][j] == 6 : 
                            pred_present_target.append(1)
                        else : 
                            pred_present_target.append(0)

                        current_target_pred.append(i)
                    else : 
                        if len(current_target_pred) > 0 : 
                            pred_targets.append(current_target_pred)
                            
                            if sum(pred_present_target) > 0 :
                                pred_present_gold_target.append(1)
                                pred_present_gold_target_prop.append(float(sum(pred_present_target))/len(pred_present_target))
                            else : 
                                pred_present_gold_target.append(0)
                                pred_present_gold_target_prop.append(0.0)

                        pred_present_target = []

                        current_holder_pred = []
                        if y[i][j] == 5 or y[i][j] == 6 : 
                            pred_present_target.append(1)
                        else : 
                            pred_present_target.append(0)

                        current_holder_pred.append(i)
                
            #end here
            if len(current_exp_gold) > 0 : 
                gold_exp.append(current_exp_gold)
                if sum(gold_present_exp) > 0 :
                    gold_present_pred_exp_prop.append(float(sum(gold_present_exp))/len(gold_present_exp))
                    gold_present_pred_exp.append(1)

                else : 
                    gold_present_pred_exp_prop.append(0)
                    gold_present_pred_exp.append(0)

                    
            if len(current_holder_gold) > 0 : 
                gold_holders.append(current_holder_gold)
                if sum(gold_present_holder) > 0 :
                    gold_present_pred_holder_prop.append(float(sum(gold_present_holder))/len(gold_present_holder))
                    gold_present_pred_holder.append(1)
                else : 
                    gold_present_pred_holder_prop.append(0.0)
                    gold_present_pred_holder.append(0)

            if len(current_target_gold) > 0 : 
                gold_targets.append(current_target_gold)
                if sum(gold_present_target) > 0 :
                    gold_present_pred_target_prop.append(float(sum(gold_present_target))/len(gold_present_target))
                    gold_present_pred_target.append(1)
                else : 
                    gold_present_pred_target_prop.append(0.0)
                    gold_present_pred_target.append(0)


            if len(current_exp_pred) > 0 : 
                pred_exp.append(current_exp_pred)
                if sum(pred_present_exp) > 0 :
                    pred_present_gold_exp_prop.append(float(sum(pred_present_exp))/len(pred_present_exp))
                    pred_present_gold_exp.append(1)

                else : 
                    pred_present_gold_exp_prop.append(0.0)
                    pred_present_gold_exp.append(0)

                    
            if len(current_holder_pred) > 0 : 
                pred_holders.append(current_holder_pred)
                if sum(pred_present_holder) > 0 :
                    pred_present_gold_holder.append(1)
                    pred_present_gold_holder_prop.append(float(sum(pred_present_holder))/len(pred_present_holder))
                else : 
                    pred_present_gold_holder.append(0)
                    pred_present_gold_holder_prop.append(0)

            if len(current_target_pred) > 0 : 
                pred_targets.append(current_target_pred)
                if sum(pred_present_target) > 0 :
                    pred_present_gold_target.append(1)
                    pred_present_gold_target_prop.append(float(sum(pred_present_target))/len(pred_present_target))
                else : 
                    pred_present_gold_target.append(0)
                    pred_present_gold_target_prop.append(0.0)

            ####Now here do the analysiss!!!################
            All_pos_exp += len(gold_exp)
            TP_prec_exp += sum(pred_present_gold_exp)
            All_ans_exp += len(pred_exp)
            TP_rec_exp += sum(gold_present_pred_exp)
            
            TP_prec_exp_prop += sum(pred_present_gold_exp_prop)
            TP_rec_exp_prop += sum(gold_present_pred_exp_prop)

            All_pos_holder += len(gold_holders)
            TP_prec_holder += sum(pred_present_gold_holder)
            All_ans_holder += len(pred_holders)
            TP_rec_holder += sum(gold_present_pred_holder)
            
            TP_prec_holder_prop += sum(pred_present_gold_holder_prop)
            TP_rec_holder_prop += sum(gold_present_pred_holder_prop)

            All_pos_target += len(gold_targets)
            TP_prec_target += sum(pred_present_gold_target)
            All_ans_target += len(pred_targets)
            TP_rec_target += sum(gold_present_pred_target)
            
            TP_prec_target_prop += sum(pred_present_gold_target_prop)
            TP_rec_target_prop += sum(gold_present_pred_target_prop)

            
            #print str(TP_prec_exp)+"\t"+str(TP_prec_exp_prop)+"\t"+str(All_ans_exp)
            #print str(TP_rec_exp)+"\t"+ str(TP_rec_exp_prop)+"\t"+str(All_pos_exp)
            
            #if can_break == 1 : 
            #    break
            #if can_break == 1 : 
            #   break


    
    f_measure_all = 0.0
    f_measure_prop_all = 0.0
    f_measure_exact_all = 0.0
    
    final_pred = TP_rec_exp
    final_pred_prop = TP_rec_exp_prop
    #TP_prec_expr_exact += 1
    #TP_rec_expr_exact += 1
    final_pred_exact = TP_rec_exp_exact
    final_correct_expr = All_pos_exp

    final_pred_prec = TP_prec_exp
    final_pred_prec_prop = TP_prec_exp_prop
    final_pred_prec_exact = TP_prec_exp_exact
    final_correct_expr_prec = All_ans_exp

    print "Recall\t"+str(final_pred)+"\t"+str(final_pred_prop)+"\t"+str(final_pred_exact)+"\t"+str(final_correct_expr)
    print "Precision\t"+str(final_pred_prec)+"\t"+str(final_pred_prec_prop)+"\t"+str(final_pred_prec_exact)+"\t"+str(final_correct_expr_prec)
    
    recall = (numpy_floatX(final_pred)/final_correct_expr)
    precision = (numpy_floatX(final_pred_prec)/final_correct_expr_prec)
    f_measure = 2*recall*precision/(recall+precision)

    recall_prop = (numpy_floatX(final_pred_prop)/final_correct_expr)
    precision_prop = (numpy_floatX(final_pred_prec_prop)/final_correct_expr_prec)
    f_measure_prop = 2*recall_prop*precision_prop/(recall_prop+precision_prop)

    recall_exact = (numpy_floatX(final_pred_exact)/final_correct_expr)
    precision_exact = (numpy_floatX(final_pred_prec_exact)/final_correct_expr_prec)
    f_measure_exact = 2*recall_exact*precision_exact/(recall_exact+precision_exact)

    print "Recall\t"+str(recall)+"\t"+str(recall_prop)+"\t"+str(recall_exact)
    print "Precision\t"+str(precision)+"\t"+str(precision_prop)+"\t"+str(precision_exact)

    print "F-score\t Exp \t"+str(f_measure)+"\t"+str(f_measure_prop)+"\t"+str(f_measure_exact)

    f_measure_all += f_measure
    f_measure_prop_all += f_measure_prop
    f_measure_exact_all += f_measure_exact

    final_pred = TP_rec_holder
    final_pred_prop = TP_rec_holder_prop
    final_pred_exact = TP_rec_holder_exact
    final_correct_expr = All_pos_holder

    final_pred_prec = TP_prec_holder
    final_pred_prec_prop = TP_prec_holder_prop
    final_pred_prec_exact = TP_prec_holder_exact
    final_correct_expr_prec = All_ans_holder
    print "-------------"

    print "Recall\t"+str(final_pred)+"\t"+str(final_pred_prop)+"\t"+str(final_pred_exact)+"\t"+str(final_correct_expr)
    print "Precision\t"+str(final_pred_prec)+"\t"+str(final_pred_prec_prop)+"\t"+str(final_pred_prec_exact)+"\t"+str(final_correct_expr_prec)
    
    recall = (numpy_floatX(final_pred)/final_correct_expr)
    precision = (numpy_floatX(final_pred_prec)/final_correct_expr_prec)
    f_measure = 2*recall*precision/(recall+precision)

    recall_prop = (numpy_floatX(final_pred_prop)/final_correct_expr)
    precision_prop = (numpy_floatX(final_pred_prec_prop)/final_correct_expr_prec)
    f_measure_prop = 2*recall_prop*precision_prop/(recall_prop+precision_prop)

    recall_exact = (numpy_floatX(final_pred_exact)/final_correct_expr)
    precision_exact = (numpy_floatX(final_pred_prec_exact)/final_correct_expr_prec)
    f_measure_exact = 2*recall_exact*precision_exact/(recall_exact+precision_exact)

    print "Recall\t"+str(recall)+"\t"+str(recall_prop)+"\t"+str(recall_exact)
    print "Precision\t"+str(precision)+"\t"+str(precision_prop)+"\t"+str(precision_exact)
    
    print "F-score\t Holder \t"+str(f_measure)+"\t"+str(f_measure_prop)+"\t"+str(f_measure_exact)

    f_measure_all += f_measure
    f_measure_prop_all += f_measure_prop
    f_measure_exact_all += f_measure_exact

    final_pred = TP_rec_target
    final_pred_prop = TP_rec_target_prop
    final_pred_exact = TP_rec_target_exact
    final_correct_expr = All_pos_target

    final_pred_prec = TP_prec_target
    final_pred_prec_prop = TP_prec_target_prop
    final_pred_prec_exact = TP_prec_target_exact
    final_correct_expr_prec = All_ans_target

    print "-------------"
    print "Recall\t"+str(final_pred)+"\t"+str(final_pred_prop)+"\t"+str(final_pred_exact)+"\t"+str(final_correct_expr)
    print "Precision\t"+str(final_pred_prec)+"\t"+str(final_pred_prec_prop)+"\t"+str(final_pred_prec_exact)+"\t"+str(final_correct_expr_prec)

    recall = (numpy_floatX(final_pred)/final_correct_expr)
    precision = (numpy_floatX(final_pred_prec)/final_correct_expr_prec)
    f_measure = 2*recall*precision/(recall+precision)

    recall_prop = (numpy_floatX(final_pred_prop)/final_correct_expr)
    precision_prop = (numpy_floatX(final_pred_prec_prop)/final_correct_expr_prec)
    f_measure_prop = 2*recall_prop*precision_prop/(recall_prop+precision_prop)
    
    recall_exact = (numpy_floatX(final_pred_exact)/final_correct_expr)
    precision_exact = (numpy_floatX(final_pred_prec_exact)/final_correct_expr_prec)
    f_measure_exact = 2*recall_exact*precision_exact/(recall_exact+precision_exact)

    print "Recall\t"+str(recall)+"\t"+str(recall_prop)+"\t"+str(recall_exact)
    print "Precision\t"+str(precision)+"\t"+str(precision_prop)+"\t"+str(precision_exact)
    
    print "F-score\t Target\t"+str(f_measure)+"\t"+str(f_measure_prop)+"\t"+str(f_measure_exact)

    f_measure_all += f_measure
    f_measure_prop_all += f_measure_prop
    f_measure_exact_all += f_measure_exact

    print "Relations!!====================================================="
    
    f_measure1 = 0.0
    
    recall = (numpy_floatX(TP_holder_rec)/All_gold_holder)
    precision = (numpy_floatX(TP_holder_prec)/All_pred_holder)
    f_measure_holder = 2*recall*precision/(recall+precision)
    
    print "Recall\t"+str(TP_holder_rec)+"\t"+str(All_gold_holder)+"\t"+str(recall)
    print "Precision\t"+str(TP_holder_prec)+"\t"+str(All_pred_holder)+"\t"+str(precision)

    print "F-score\t Holder\t"+str(f_measure_holder)
    
    recall = (numpy_floatX(TP_target_rec)/All_gold_target)
    precision = (numpy_floatX(TP_target_prec)/All_pred_target)
    f_measure_target = 2*recall*precision/(recall+precision)

    print "Recall\t"+str(TP_target_rec)+"\t"+str(All_gold_target)+"\t"+str(recall)
    print "Precision\t"+str(TP_target_prec)+"\t"+str(All_pred_target)+"\t"+str(precision)
    print "F-score\t Target\t"+str(f_measure_target)
   
    f_measure1 = 2*f_measure_target*f_measure_holder/(f_measure_target+f_measure_holder)
    
    if f_measure1 >= 0 : 
        valid_err = 1. - f_measure1
    else : 
        valid_err = 1.

    predictions.close()
    
    return valid_err


def same_class(x, y) :
    if (x == 1 or x == 2) and (y == 1 or y == 2):
        return 1
    if (x == 3 or x == 4) and (y == 3 or y == 4):
        return 1
    if (x == 5 or x == 6) and (y == 5 or y == 6):
        return 1
    return 0


def pred_error_relation3(f_pred, f_pred_hrel, f_pred_trel, prepare_data, prepare_data_words, find_relations_lrE, data, word_labels, iterator, tparams, window, relation,  verbose = False) :

    All_pos_exp = 0
    TP_prec_exp = 0
    All_ans_exp = 0
    TP_rec_exp = 0
    
    TP_prec_exp_prop = 0.0
    TP_rec_exp_prop = 0.0
    TP_prec_exp_exact = 0.0
    TP_rec_exp_exact = 0.0


    All_pos_holder = 0
    TP_prec_holder = 0
    All_ans_holder = 0
    TP_rec_holder = 0
    
    TP_prec_holder_prop = 0.0
    TP_rec_holder_prop = 0.0
    TP_prec_holder_exact = 0.0
    TP_rec_holder_exact = 0.0


    All_pos_target = 0
    TP_prec_target = 0
    All_ans_target = 0
    TP_rec_target = 0
    
    TP_prec_target_prop = 0.0
    TP_rec_target_prop = 0.0
    TP_prec_target_exact = 0.0
    TP_rec_target_exact = 0.0


    All_gold_holder = 0
    All_gold_target = 0
    All_pred_holder = 0
    All_pred_target = 0

    TP_holder_prec = 0
    TP_holder_rec = 0
    TP_target_prec = 0
    TP_target_rec = 0

    pred_all_relations_full = []
    pred_all_relations_y_full = []

    gold_holder_relations_full = []
    gold_target_relations_full = []

    predictions = open('./lstm_results_3_'+str(relation), 'w+')                                             

    for _, valid_index in iterator : 
        #print valid_index
        #print data
        #print len(data[2])

        x = [data[0][t] for t in valid_index]
        y = numpy.array(data[1])[valid_index]

        word_y = [word_labels[t] for t in valid_index]
        
        all_holder_relations = []
        all_target_relations = []
        
        holder_dim  = 1
        target_dim = 1

        x, mask, y = prepare_data([data[0][t] for t in valid_index], 
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)

        #print y
        
        #print "==========================="
                
        #preds = []


        preds1 = f_pred(x, mask)
        preds_h, preds_hw = f_pred_hrel(x, mask)
        preds_t, preds_tw = f_pred_trel(x, mask)

        old_preds = copy.deepcopy(preds1)
        preds = numpy.zeros((len(y), len(y[0])))


        #for every preds_h and preds_tw, mention if it is related or not related
        preds1 = correct_relations(y, preds1)
        preds_h = correct_relations(y, preds_h)
        preds_t = correct_relations(y, preds_t)

        for j in range(len(y[0])) :
            entities_h = dict()
            entity_spans_h = dict()
            index = 0
            related_h = numpy.zeros((len(y)))
            for i in range(len(y)) :
                if preds_h[i][j] == 1 :
                    start_e = i
                    end_e = i
                    if i+1 < len(y) : 
                        while preds_h[i+1][j] == 2 : 
                            i += 1
                            end_e = i
                            if i+1 == len(y) :
                                break
                    index+=1
                    for p in range(start_e, end_e+1) : 
                        entities_h[p] = index
                        entity_spans_h[index] = (start_e, end_e)

                if preds_h[i][j] == 3 :                            
                    start_e = i
                    end_e = i
                    if i+1 < len(y) :
                        while preds_h[i+1][j] == 4 : 
                            i += 1
                            end_e = i
                            if i+1 == len(y) :
                                break
                    index+=1
                    for p in range(start_e, end_e+1) : 
                        entities_h[p] = index
                        entity_spans_h[index] = (start_e, end_e)

                if preds_h[i][j] == 5 :                            
                    start_e = i
                    end_e = i
                    if i+1 < len(y) :
                        while preds_h[i+1][j] == 6 : 
                            i += 1
                            end_e = i
                            if i+1 == len(y) :
                                break
                    index+=1
                    for p in range(start_e, end_e+1) : 
                        entities_h[p] = index
                        entity_spans_h[index] = (start_e, end_e)

            related_entities_h = dict()
            for i in range(len(y)) :
                if preds_hw[i][j] > 0 :
                    if i-preds_hw[i][j] < 0 :
                        continue
                    if i in entities_h.keys() : 
                        related_entities_h[entities_h[i]] = 1
                        if i-preds_hw[i][j] in entities_h.keys() : 
                            related_entities_h[entities_h[i-preds_hw[i][j]]] = 1

            for i in range(len(y)) :
                if i in entities_h.keys() : 
                    if entities_h[i] in related_entities_h.keys() :
                        related_h[i] = 1

            entities_t = dict()
            entity_spans_t = dict()
            index = 0
            related_t = numpy.zeros((len(y)))
            for i in range(len(y)) :
                if preds_t[i][j] == 1 :
                    start_e = i
                    end_e = i
                    if i+1 < len(y) :
                        while preds_t[i+1][j] == 2 : 
                            i += 1
                            end_e = i
                            if i+1 == len(y) :
                                break
                    index+=1
                    for p in range(start_e, end_e+1) : 
                        entities_t[p] = index
                        entity_spans_t[index] = (start_e, end_e)

                if preds_t[i][j] == 3 :                            
                    start_e = i
                    end_e = i
                    if i+1 < len(y) :
                        while preds_t[i+1][j] == 4 : 
                            i += 1
                            end_e = i
                            if i+1 == len(y) :
                                break
                    index+=1
                    for p in range(start_e, end_e+1) : 
                        entities_t[p] = index
                        entity_spans_t[index] = (start_e, end_e)

                if preds_t[i][j] == 5 :                            
                    start_e = i
                    end_e = i
                    if i+1 < len(y) :
                        while preds_t[i+1][j] == 6 : 
                            i += 1
                            end_e = i
                            if i+1 == len(y) :
                                break
                    index+=1
                    for p in range(start_e, end_e+1) : 
                        entities_t[p] = index
                        entity_spans_t[index] = (start_e, end_e)

            related_entities_t = dict()
            for i in range(len(y)) :
                if preds_tw[i][j] > 0 :
                    if i+preds_tw[i][j] > len(y)-1 :
                        continue
                    if i in entities_t.keys() : 
                        related_entities_t[entities_t[i]] = 1
                    if i+preds_tw[i][j] in entities_t.keys() : 
                        related_entities_t[entities_t[i+preds_tw[i][j]]] = 1

            for i in range(len(y)) :
                if i in entities_t.keys() : 
                    if entities_t[i] in related_entities_t.keys() :
                        related_t[i] = 1


            #All the checks corresponding to the three labels

            for i in range(len(y)) :
                if related_h[i] == 1 :
                    #atleast two of these should match
                    if same_class(preds_h[i][j], preds1[i][j]) == 1 or same_class(preds_h[i][j], preds_t[i][j]) == 1 :
                        preds[i][j] = preds_h[i][j]
                if related_t[i] == 1 :
                    if same_class(preds_t[i][j], preds1[i][j]) == 1 or same_class(preds_t[i][j], preds_h[i][j]) == 1 :
                        if preds[i][j] == 0 : 
                            preds[i][j] = preds_t[i][j]
                if related_h[i] == 0 and related_t[i] == 0 : 
                    if same_class(preds1[i][j], preds_h[i][j]) == 1 and same_class(preds_h[i][j], preds_t[i][j]) == 1 :
                        preds[i][j] = preds1[i][j]
                    else :
                        preds[i][j] = 0
                        
        #for k in numpy.arange(len(preds1)) : 
        #    preds.append(viterbi_segment(preds1[k], tparams))
        #print preds
        #x, mask, preds = prepare_data([data[0][t] for t in valid_index], 
        #                         preds,
        #                                                      maxlen=None)
        #print preds
        #print y
        targets = numpy.array(data[1])[valid_index]  #this is most probably is the array of the arrays!
        #is complicated. Either hard_target or the soft_target. Only hard_target implemented for now
        #print y        


        '''x, mask, y = prepare_data([data[0][t] for t in valid_index], 
                                  numpy.array(data[1])[valid_index],
                                                               maxlen=None)
        preds = f_pred(x, mask)
        #print preds
        targets = numpy.array(data[1])[valid_index]  #this is most probably is the array of the arrays!
        #is complicated. Either hard_target or the soft_target. Only hard_target implemented for now
        #print y      

        if "train" in file_data : 
            word_targets = numpy.array(word_labels[0])[valid_index]
        if "valid" in file_data : 
            word_targets = numpy.array(word_labels[1])[valid_index]
        if "test" in file_data : 
            word_targets = numpy.array(word_labels[2])[valid_index]

        temp, mask, gold = prepare_data_words([data[0][t] for t in valid_index],
                                                      word_targets, 
                                                          maxlen=None)'''


        word_targets = numpy.array(word_labels)[valid_index]
        temp, mask, gold = prepare_data_words([data[0][t] for t in valid_index],
                                                      word_targets, 
                                                          maxlen=None)

        for j in range(len(y[0])) :
            for i in range(len(y)) :
                if preds[i][j] > 6 : 
                    preds[i][j] = 0

        for j in range(len(y[0])) :
            for i in range(len(y)) :
                if y[i][j] > 6 : 
                    y[i][j] = 0



        for j in range(len(y[0])) :
            for i in range(len(y)) :
                if preds[i][j] == 0 : 
                    if (i+1) < len(y)-1 : 
                        if preds[i+1][j] > 0 and preds[i+1][j] < 3: 
                            preds[i+1][j] = 1
                        elif preds[i+1][j] > 2 and preds[i+1][j] < 5:
                            preds[i+1][j] = 3
                        elif preds[i+1][j] > 4 : 
                            preds[i+1][j] = 5

                if preds[i][j] > 0 and preds[i][j] < 3:
                    if (i+1) < len(y)-1 : 
                        if preds[i+1][j] > 0 and preds[i+1][j] < 3:
                            preds[i+1][j] = 2
                elif preds[i][j] > 2 and preds[i][j] < 5: 
                    if (i+1) < len(y)-1 : 
                        if preds[i+1][j] > 2 :
                            preds[i+1][j] = 4
                elif preds[i][j] > 4: 
                    if (i+1) < len(y)-1 : 
                        if preds[i+1][j] > 4 :
                            preds[i+1][j] = 6


                            
        new_preds = preds

        for j in range(len(x[0])) : 
            pred_holder = []
            pred_target = []
            pred_expr = []
            
            gold_holder_start = dict()
            gold_holder_end = dict()
            gold_target_start = dict()
            gold_target_end = dict()
            gold_expr_start = dict()
            gold_expr_end = dict()
            
            start_index_h = -1
            end_index_h = -1

            start_index_t = -1
            end_index_t = -1

            start_index_e = -1
            end_index_e = -1
            
            
            for i in range(len(x)) : 
                #if x[i][j] > 0 : 
                #    print str(index_dict[str(x[i][j])])+"\t"+str(gold[i][j])+"\t"+str(preds[i][j])
                if preds[i][j] == 3 and x[i][j] > 0: #beginning of a holder
                    start_index_h = i
                    if i+1 >= len(x) : 
                        pred_holder.append([start_index_h, start_index_h])
                        start_index_h = -1
                        end_index_h = -1
                        break
                    while (i+1) < len(x) and preds[i+1][j] == 4 : 
                        i+=1
                        end_index_h = i

                    if end_index_h == -1 : 
                        end_index_h = start_index_h
                    pred_holder.append([start_index_h, end_index_h])
                    start_index_h = -1
                    end_index_h = -1


                if preds[i][j] == 5 and x[i][j] > 0: 
                    start_index_t = i
                    if i+1 >= len(x) : 
                        pred_target.append([start_index_t, start_index_t])
                        start_index_t = -1
                        end_index_t = -1
                        break

                    while (i+1) < len(x) and preds[i+1][j] == 6 : 
                        i+=1
                        end_index_t = i

                    if end_index_t == -1 : 
                        end_index_t = start_index_t
                    pred_target.append([start_index_t, end_index_t])
                    start_index_t = -1
                    end_index_t = -1


                if preds[i][j] == 1 and x[i][j] > 0: 
                    start_index_e = i
                    if i+1 >= len(x) : 
                        pred_expr.append([start_index_e, start_index_e])
                        start_index_e = -1
                        end_index_e = -1
                        break

                    while (i+1) < len(x) and preds[i+1][j] == 2 : 
                        i+=1
                        end_index_e = i

                    if end_index_e == -1 : 
                        end_index_e = start_index_e
                    pred_expr.append([start_index_e, end_index_e])
                    start_index_e = -1
                    end_index_e = -1

                    
            for i in range(len(x)) : 
                if gold[i][j].startswith("B_AGENT") and len(gold[i][j].split("_")) > 2 : 
                    #print gold[i][j].split("_")
                    rel = []
                    for k in numpy.arange(1, len(gold[i][j].split("_REL"))) :
                        rel.append(int(gold[i][j].split("_REL")[k]))
                        gold_holder_start[rel[-1]] = i
                    
                    if i+1 >= len(x) : 
                        for r in rel : 
                            gold_holder_end[r] = i
                        break
                    
                    while i+1 < len(x) and gold[i+1][j].startswith("AGENT") :
                        i+=1
                        for r in rel : 
                            gold_holder_end[r] = i
                            
                    for r in rel  :
                        if r not in gold_holder_end.keys() : 
                            gold_holder_end[r] = gold_holder_start[r]
                        
                if gold[i][j].startswith("B_TARGET") and len(gold[i][j].split("_")) > 2 : 
                    #print gold[i][j].split("_")
                    #rel = int(gold[i][j].split("_")[-1][3:])
                    #gold_target_start[rel] = i
                    rel = []
                    for k in numpy.arange(1, len(gold[i][j].split("_REL"))) : 
                        rel.append(int(gold[i][j].split("_REL")[k]))
                        gold_target_start[rel[-1]] = i

                    if i+1 >= len(x) :
                        for  r in rel : 
                            gold_target_end[r] = i
                        break

                    while i+1 < len(x) and gold[i+1][j].startswith("TARGET") :
                        i+=1
                        for  r in rel : 
                            gold_target_end[r] = i

                    for r in rel : 
                        if r not in gold_target_end.keys() : 
                            gold_target_end[r] = gold_target_start[r]

                if gold[i][j].startswith("B_DSE") and len(gold[i][j].split("_")) > 2: 
                    #print gold[i][j].split("_")
                    #rel = int(gold[i][j].split("_")[-1][3:])
                    #gold_expr_start[rel] = i

                    rel = []
                    for k in numpy.arange(1, len(gold[i][j].split("_REL"))) : 
                        rel.append(int(gold[i][j].split("_REL")[k]))
                        gold_expr_start[rel[-1]] = i

                    
                    if i+1 >= len(x) : 
                        for r in rel : 
                            gold_expr_end[r] = i
                        break

                    while i+1 < len(x) and gold[i+1][j].startswith("DSE") :
                        i+=1
                        for r in rel : 
                            gold_expr_end[r] = i

                    for r in rel : 
                        if r not in gold_expr_end.keys() : 
                            gold_expr_end[r] = gold_expr_start[r]

            #######Now all the respective annotations are done!!!!

            gold_holder_relations = []
            gold_target_relations = []

            #pred_holder_relations = []
            #pred_target_relations = []

            pred_all_relations = []
            pred_all_relations_y = []

            for key in gold_expr_start.keys() : 
                #print key
                start_e = gold_expr_start[key]
                #print gold_expr_start.keys()
                #print gold_expr_end.keys()
                end_e = gold_expr_end[key]
                if key in gold_holder_start.keys() : 
                    start_h = gold_holder_start[key]
                    end_h = gold_holder_end[key]
                    gold_holder_relations.append([start_h, end_h, start_e, end_e])

                if key in gold_target_start.keys() : 
                    start_t = gold_target_start[key]
                    end_t = gold_target_end[key]
                    gold_target_relations.append([start_t, end_t, start_e, end_e])

            #gold_holder_relations_full.append(gold_holder_relations)
            #gold_target_relations_full.append(gold_target_relations)
            
            #For the predicted all pairs are the ones

            
            holder_index = dict()
            target_index = dict()
            expr_index = dict()
            
            for i in range(len(x)) : 
                holder_index[i] = []
                target_index[i] = []
                expr_index[i] = []


            for i in range(len(x)) : 
                if preds[i][j] > 0  and x[i][j] > 0 : 
                    if preds[i][j] == 1 or preds[i][j] == 2 : 
                        for p in range(len(pred_expr)) :
                            [start_e, end_e] = pred_expr[p]
                            if start_e <= i and i<= end_e : 
                                expr_index[i].append(p)
                    if preds[i][j] == 3 or preds[i][j] == 4 : 
                        for p in range(len(pred_holder)) :
                            [start_h, end_h] = pred_holder[p]
                            if start_h <= i and i<= end_h : 
                                holder_index[i].append(p)
                    if preds[i][j] == 5 or preds[i][j] == 6 : 
                        for p in range(len(pred_target)) :
                            [start_t, end_t] = pred_target[p]
                            if start_t <= i and i<= end_t : 
                                target_index[i].append(p)
            

            #preds_h, preds_hw = f_pred_hrel(x, mask)
            #preds_t, preds_tw = f_pred_trel(x, mask)

            
            def Concat_append(pair, entity) : 
                [start_index, end_index] = pair
                
                for i in range(len(entity)) :
                    [start_e, end_e] = entity[i]
                    if end_e == start_index-1 : 
                        entity[i] = [start_e, end_index]
                        return [start_e, end_index], entity

                    if start_e == end_index+1 : 
                        entity[i] = [start_index, end_e]
                        return [start_index, end_e], entity
                        
                entity.append([start_index, end_index])
                return [start_index, end_index], entity


            pred_holder_relations = []
            pred_target_relations = []

            
            #these many windows to the left
            for i in range(len(x)) : 
                if (i-preds_hw[i][j]) < 0 : 
                    continue
                if preds_hw[i][j] > 0 and x[i][j] > 0 : 
                    if len(expr_index[i]) > 0 : 
                        if (i-preds_hw[i][j]) >= 0 and len(holder_index[i-preds_hw[i][j]]) > 0 :
                            for ei in expr_index[i] : 
                                [start_e, end_e] = pred_expr[ei]
                                for hi in holder_index[i-preds_hw[i][j]] : 
                                    [start_h, end_h] = pred_holder[hi]
                                    if [start_h, end_h, start_e, end_e] not in pred_holder_relations : 
                                        if [start_h, end_h-1, start_e, end_e] in pred_holder_relations : 
                                            pred_holder_relations.remove([start_h, end_h-1, start_e, end_e])
                                        if [start_h, end_h, start_e+1, end_e] in pred_holder_relations : 
                                            pred_holder_relations.remove([start_h, end_h, start_e+1, end_e])

                                        pred_holder_relations.append([start_h, end_h, start_e, end_e])
                        
                                    
                    elif len(target_index[i]) > 0 : 
                        if (i-preds_hw[i][j]) >= 0 and len(expr_index[i-preds_hw[i][j]]) > 0 :
                            for ti in target_index[i] : 
                                [start_t, end_t] = pred_target[ti]
                                for ei in expr_index[i-preds_hw[i][j]] : 
                                    [start_e, end_e] = pred_expr[ei]
                                    if [start_t, end_t, start_e, end_e] not in pred_target_relations : 
                                        if [start_t, end_t-1, start_e, end_e] in pred_target_relations : 
                                            pred_target_relations.remove([start_t, end_t-1, start_e, end_e])
                                        if [start_t, end_t, start_e+1, end_e] in pred_target_relations : 
                                            pred_target_relations.remove([start_t, end_t, start_e+1, end_e])

                                        pred_target_relations.append([start_t, end_t, start_e, end_e])
                        

            for i in range(len(x)) : 
                if (i+preds_tw[i][j]) >= len(x) : 
                    continue
                if preds_tw[i][j] > 0 and x[i][j] > 0 : 
                    if len(expr_index[i]) > 0 : 
                        if (i+preds_tw[i][j]) < len(x)  and len(holder_index[i+preds_tw[i][j]]) > 0 :
                            for ei in expr_index[i] : 
                                [start_e, end_e] = pred_expr[ei]
                                for hi in holder_index[i+preds_tw[i][j]] : 
                                    [start_h, end_h] = pred_holder[hi]
                                    if [start_h, end_h, start_e, end_e] not in pred_holder_relations : 
                                        if [start_h, end_h-1, start_e, end_e] in pred_holder_relations : 
                                            pred_holder_relations.remove([start_h, end_h-1, start_e, end_e])
                                        if [start_h, end_h, start_e+1, end_e] in pred_holder_relations : 
                                            pred_holder_relations.remove([start_h, end_h, start_e+1, end_e])

                                        pred_holder_relations.append([start_h, end_h, start_e, end_e])
                                
                                    
                    elif len(target_index[i]) > 0 : 
                        if (i+preds_tw[i][j]) < len(x) and len(expr_index[i+preds_tw[i][j]]) > 0 :
                            for ti in target_index[i] : 
                                [start_t, end_t] = pred_target[ti]
                                for ei in expr_index[i+preds_tw[i][j]] : 
                                    [start_e, end_e] = pred_expr[ei]
                                    if [start_t, end_t, start_e, end_e] not in pred_target_relations : 
                                        if [start_t, end_t-1, start_e, end_e] in pred_target_relations : 
                                            pred_target_relations.remove([start_t, end_t-1, start_e, end_e])
                                        if [start_t, end_t, start_e+1, end_e] in pred_target_relations : 
                                            pred_target_relations.remove([start_t, end_t, start_e+1, end_e])

                                        pred_target_relations.append([start_t, end_t, start_e, end_e])
                                
                

            '''for [start_e, end_e] in pred_expr : 
                for [start_h, end_h] in pred_holder : 
                    pred_all_relations.append([start_h, end_h, start_e, end_e])
                    pred_all_relations_y.append(1)

                for [start_t, end_t] in pred_target : 
                    pred_all_relations.append([start_t, end_t, start_e, end_e])
                    pred_all_relations_y.append(2)'''

            
            #pred_all_relations_full.append(pred_all_relations)
            #pred_all_relations_y_full.append(pred_all_relations_y)


            #print "Gold=========================================="
            #print pred_holder_relations
            #print gold_holder_relations
            #print gold_target_relations

            
            #print "Pred=========================================="
            #print pred_holder_relations
            #print pred_all_relations
            #print pred_all_relations_y

            
            #print pred_all_relations
            #print pred_all_relations_y

            #if len(pred_all_relations) == 0 : 
            #    temp = []
            #    pred_all_relations.append(temp)

            #pred_all_relations_full = []
            #pred_all_relations_full_y = []

            #if not len(pred_all_relations[0]) == 0 : 
            #pred_all_relations_full.append(pred_all_relations)
            #pred_all_relations_full_y.append(pred_all_relations_y)

            #print "Before===="
            #print pred_all_relations_full
            #print pred_all_relations_full_y
            
            #relations_x, mask_t, relations_y = prepare_data_relations(pred_all_relations_full, pred_all_relations_full_y)

            #print "After relations prep==="
            #print relations_x
            #print relations_y

            #pred_relations = f_pred_relations(x, mask, relations_x)

            #print pred_relations

            #pred_holder_relations = []
            #pred_target_relations = []

            '''for l in range(len(pred_relations[0])) :
                for k in range(len(pred_relations)) :
                    if pred_relations[k][l] == 1 : 
                        pred_holder_relations.append(relations_x[k][l])
                    if pred_relations[k][l] == 2 : 
                        pred_target_relations.append(relations_x[k][l])
                        #pred_holder_relations_all.append(pred_holder_relations)
                        #pred_target_relations_all.append(pred_target_relations)'''
            
            #print "-----------------------------------------------------------------"
            #print gold_holder_relations
            #print gold_target_relations
            #print "Predicted relation=========================================="
            #print pred_holder_relations
            #print pred_target_relations


            #preds = new_preds

            pred_holder_later = []
            for [sh1, eh1, se1, ee1] in pred_holder_relations : 
                for [sh2, eh2, se2, ee2] in pred_holder_later : 
                    if sh1==sh2 and eh1 == eh2 and se1 == se2 and ee1 == ee2 : 
                        break
                pred_holder_later.append([sh1, eh1, se1, ee1])

            pred_holder_relations = pred_holder_later

            pred_target_later = []
            for [sh1, eh1, se1, ee1] in pred_target_relations : 
                for [sh2, eh2, se2, ee2] in pred_target_later : 
                    if sh1==sh2 and eh1 == eh2 and se1 == se2 and ee1 == ee2 : 
                        break
                pred_target_later.append([sh1, eh1, se1, ee1])

            pred_target_relations = pred_target_later
            
            
            
            final_tag = []
            for i in range(len(x)) :
                if preds[i][j] == 1 :
                    final_tag.append("B_DSE")
                if preds[i][j] == 2 :
                    final_tag.append("DSE")
                if preds[i][j] == 3 :
                    final_tag.append("B_AGENT")
                if preds[i][j] == 4 :
                    final_tag.append("AGENT")
                if preds[i][j] == 5 :
                    final_tag.append("B_TARGET")
                if preds[i][j] == 6 :
                    final_tag.append("TARGET")
                if preds[i][j] == 0 :
                    final_tag.append("O")
                if preds[i][j] == 7 : 
                    final_tag.append("B_ESE")
                if preds[i][j] == 8 : 
                    final_tag.append("ESE")
                if preds[i][j] == 9 :
                    final_tag.append("B_OBJ")
                if preds[i][j] == 10 : 
                    final_tag.append("OBJ")
            
            count_h = 1
            count_t = 1
            
            for [sh, eh, se, ee] in pred_holder_relations : 
                for ind_h in numpy.arange(sh, eh+1) : 
                    final_tag[ind_h] = final_tag[ind_h]+"_-"+str(count_h)
                for ind_e in numpy.arange(se, ee+1) : 
                    final_tag[ind_e] = final_tag[ind_e]+"_-"+str(count_h)
                count_h+=1
            
            #count_t = 1
            for [st, et, se, ee] in pred_target_relations :
                for ind_t in numpy.arange(st, et+1) :
                    final_tag[ind_t] = final_tag[ind_t]+"_"+str(count_t)
                for ind_e in numpy.arange(se, ee+1) :
                    final_tag[ind_e] = final_tag[ind_e]+"_"+str(count_t)
                count_t+=1

            #remove all the ones with no relations at place of holders and targets!!

            '''for i in range(len(x)) : 
                if final_tag[i] == 'B_AGENT' or final_tag[i] == 'AGENT' or final_tag[i] == 'B_TARGET' or final_tag[i] == 'TARGET': 
                    final_tag[i] = 'O'
                    preds[i][j] = 0'''


            for i in range(len(x)) :
                predictions.write(str(index_dict[str(x[i][j])]))
                predictions.write("\t")
                '''predictions.write(str(y[i][j]))
                predictions.write("\t")
                predictions.write(str(int(preds[i][j])))
                predictions.write("\t")
                predictions.write(str(old_preds[i][j]))
                predictions.write("\t")
                predictions.write(str(preds_h[i][j]))
                predictions.write("\t")
                predictions.write(str(preds_t[i][j]))
                predictions.write("\t")
                predictions.write(str(preds_hw[i][j]))
                predictions.write("\t")
                predictions.write(str(preds_tw[i][j]))
                predictions.write("\t")'''
                predictions.write(str(word_y[0][i]))
                predictions.write("\t")
                predictions.write(str(final_tag[i]))

                predictions.write("\n")
            #predictions.write(str(word_y))
            #predictions.write("\n")
            predictions.write("\n")


            for [pstart_h, pend_h, pstart_e, pend_e] in pred_holder_relations : 
                for [gstart_h, gend_h, gstart_e, gend_e] in gold_holder_relations : 
                    list1 = numpy.arange(pstart_h, pend_h+1)
                    list2 = numpy.arange(gstart_h, gend_h+1)

                    list3 = numpy.arange(pstart_e, pend_e+1)
                    list4 = numpy.arange(gstart_e, gend_e+1)


                    if len(list(set(list1) & set(list2))) > 0 and len(list(set(list3) & set(list4))) > 0 :
                        TP_holder_prec+=1


            for [gstart_h, gend_h, gstart_e, gend_e] in gold_holder_relations : 
                for [pstart_h, pend_h, pstart_e, pend_e] in pred_holder_relations : 
                    list1 = numpy.arange(pstart_h, pend_h+1)
                    list2 = numpy.arange(gstart_h, gend_h+1)

                    list3 = numpy.arange(pstart_e, pend_e+1)
                    list4 = numpy.arange(gstart_e, gend_e+1)

                    if len(list(set(list1) & set(list2))) > 0 and len(list(set(list3) & set(list4))) > 0 :
                        TP_holder_rec+=1

            for [pstart_t, pend_t, pstart_e, pend_e] in pred_target_relations : 
                for [gstart_t, gend_t, gstart_e, gend_e] in gold_target_relations : 
                    list1 = numpy.arange(pstart_t, pend_t+1)
                    list2 = numpy.arange(gstart_t, gend_t+1)

                    list3 = numpy.arange(pstart_e, pend_e+1)
                    list4 = numpy.arange(gstart_e, gend_e+1)

                    if len(list(set(list1) & set(list2))) > 0 and len(list(set(list3) & set(list4))) > 0 : 
                        TP_target_prec+=1

            for [gstart_t, gend_t, gstart_e, gend_e] in gold_target_relations : 
                for [pstart_t, pend_t, pstart_e, pend_e] in pred_target_relations : 
                    list1 = numpy.arange(pstart_t, pend_t+1)
                    list2 = numpy.arange(gstart_t, gend_t+1)

                    list3 = numpy.arange(pstart_e, pend_e+1)
                    list4 = numpy.arange(gstart_e, gend_e+1)

                    if len(list(set(list1) & set(list2))) > 0 and len(list(set(list3) & set(list4))) > 0 :
                        TP_target_rec+=1

            All_gold_holder += len(gold_holder_relations)
            All_gold_target += len(gold_target_relations)
            All_pred_holder += len(pred_holder_relations)
            All_pred_target += len(pred_target_relations)
            
            #print "HOlder\t"+str(temp_holder_pred)
            #print "Gold\t"+str(temp_holder_gold)
        
        #preds = new_preds
        

        #########################################################Relation part!!!###########################################

        for j in range(len(y[0])) :
            for i in range(len(y)) :
                if preds[i][j] == 0 : 
                    if (i+1) < len(y)-1 : 
                        if preds[i+1][j] > 0 and preds[i+1][j] < 3: 
                            preds[i+1][j] = 1
                        elif preds[i+1][j] > 2 and preds[i+1][j] < 5:
                            preds[i+1][j] = 3
                        elif preds[i+1][j] > 4 : 
                            preds[i+1][j] = 5

                if preds[i][j] > 0 and preds[i][j] < 3:
                    if (i+1) < len(y)-1 : 
                        if preds[i+1][j] > 0 and preds[i+1][j] < 3:
                            preds[i+1][j] = 2
                elif preds[i][j] > 2 and preds[i][j] < 5: 
                    if (i+1) < len(y)-1 : 
                        if preds[i+1][j] > 2 :
                            preds[i+1][j] = 4
                elif preds[i][j] > 4: 
                    if (i+1) < len(y)-1 : 
                        if preds[i+1][j] > 4 :
                            preds[i+1][j] = 6


        for j in range(len(x[0])) : 

            pred_holder = []
            pred_target = []
            pred_expr = []
                        
            start_index_h = -1
            end_index_h = -1

            start_index_t = -1
            end_index_t = -1

            start_index_e = -1
            end_index_e = -1
            
            
            for i in range(len(x)) : 
                #if x[i][j] > 0 : 
                #    print str(index_dict[str(x[i][j])])+"\t"+str(gold[i][j])+"\t"+str(preds[i][j])
                if preds[i][j] == 3 and x[i][j] > 0: #beginning of a holder
                    start_index_h = i
                    if i+1 >= len(x) : 
                        pred_holder.append([start_index_h, start_index_h])
                        start_index_h = -1
                        end_index_h = -1
                        break
                    while (i+1) < len(x) and preds[i+1][j] == 4 : 
                        i+=1
                        end_index_h = i

                    if end_index_h == -1 : 
                        end_index_h = start_index_h
                    pred_holder.append([start_index_h, end_index_h])
                    start_index_h = -1
                    end_index_h = -1


                if preds[i][j] == 5 and x[i][j] > 0: 
                    start_index_t = i
                    if i+1 >= len(x) : 
                        pred_target.append([start_index_t, start_index_t])
                        start_index_t = -1
                        end_index_t = -1
                        break

                    while (i+1) < len(x) and preds[i+1][j] == 6 : 
                        i+=1
                        end_index_t = i

                    if end_index_t == -1 : 
                        end_index_t = start_index_t
                    pred_target.append([start_index_t, end_index_t])
                    start_index_t = -1
                    end_index_t = -1


                if preds[i][j] == 1 and x[i][j] > 0: 
                    start_index_e = i
                    if i+1 >= len(x) : 
                        pred_expr.append([start_index_e, start_index_e])
                        start_index_e = -1
                        end_index_e = -1
                        break

                    while (i+1) < len(x) and preds[i+1][j] == 2 : 
                        i+=1
                        end_index_e = i

                    if end_index_e == -1 : 
                        end_index_e = start_index_e
                    pred_expr.append([start_index_e, end_index_e])
                    start_index_e = -1
                    end_index_e = -1


            gold_holder = []
            gold_target = []
            gold_expr = []
                        
            start_index_h = -1
            end_index_h = -1

            start_index_t = -1
            end_index_t = -1

            start_index_e = -1
            end_index_e = -1
            
            
            for i in range(len(x)) : 
                #if x[i][j] > 0 : 
                #    print str(index_dict[str(x[i][j])])+"\t"+str(gold[i][j])+"\t"+str(preds[i][j])
                if y[i][j] == 3 and x[i][j] > 0: #beginning of a holder
                    start_index_h = i
                    if i+1 >= len(x) : 
                        gold_holder.append([start_index_h, start_index_h])
                        start_index_h = -1
                        end_index_h = -1
                        break
                    while (i+1) < len(x) and y[i+1][j] == 4 : 
                        i+=1
                        end_index_h = i

                    if end_index_h == -1 : 
                        end_index_h = start_index_h
                    gold_holder.append([start_index_h, end_index_h])
                    start_index_h = -1
                    end_index_h = -1


                if y[i][j] == 5 and x[i][j] > 0: 
                    start_index_t = i
                    if i+1 >= len(x) : 
                        gold_target.append([start_index_t, start_index_t])
                        start_index_t = -1
                        end_index_t = -1
                        break

                    while (i+1) < len(x) and y[i+1][j] == 6 : 
                        i+=1
                        end_index_t = i

                    if end_index_t == -1 : 
                        end_index_t = start_index_t
                    gold_target.append([start_index_t, end_index_t])
                    start_index_t = -1
                    end_index_t = -1


                if y[i][j] == 1 and x[i][j] > 0: 
                    start_index_e = i
                    if i+1 >= len(x) : 
                        gold_expr.append([start_index_e, start_index_e])
                        start_index_e = -1
                        end_index_e = -1
                        break

                    while (i+1) < len(x) and y[i+1][j] == 2 : 
                        i+=1
                        end_index_e = i

                    if end_index_e == -1 : 
                        end_index_e = start_index_e
                    gold_expr.append([start_index_e, end_index_e])
                    start_index_e = -1
                    end_index_e = -1


            #do analysis and add to the global counts
            #TP_prec_target_exact = 0.0
            #TP_rec_target_exact = 0.0

            
            for [pstarte, pende] in pred_holder : 
                for [gstarte, gende] in gold_holder : 
                    if pstarte == gstarte and pende == gende : 
                        TP_prec_holder_exact += 1
                        TP_rec_holder_exact += 1
                        break

            for [pstarte, pende] in pred_target : 
                for [gstarte, gende] in gold_target : 
                    if pstarte == gstarte and pende == gende : 
                        TP_prec_target_exact += 1
                        TP_rec_target_exact += 1
                        break

            for [pstarte, pende] in pred_expr : 
                for [gstarte, gende] in gold_expr : 
                    if pstarte == gstarte and pende == gende : 
                        TP_prec_exp_exact += 1
                        TP_rec_exp_exact += 1
                        break
            
            
            
        for j in range(len(y[0])) :
            list_expr = []
            num_overlap = 0
            total_expr = 0

            gold_holders = []
            gold_targets = []
            gold_exp = []

            gold_present_pred_holder = []
            gold_present_pred_target = []
            gold_present_pred_exp = []

            gold_present_pred_holder_prop = []
            gold_present_pred_target_prop = []
            gold_present_pred_exp_prop = []



            gold_present_holder = []
            gold_present_target = []
            gold_present_exp = []

            current_holder_gold = []
            current_target_gold = []
            current_exp_gold = []

            pred_holders = []
            pred_targets = []
            pred_exp = []

            pred_present_gold_holder = []
            pred_present_gold_target = []
            pred_present_gold_exp = []

            pred_present_gold_holder_prop = []
            pred_present_gold_target_prop = []
            pred_present_gold_exp_prop = []

            pred_present_holder = []
            pred_present_target = []
            pred_present_exp = []

            current_holder_pred = []
            current_target_pred = []
            current_exp_pred = []


            for i in range(len(y)) :

                #So preds[i][j] contains the true labels and y[i][j] contains the predicted labels
                if y[i][j] == 1 and x[i][j] > 0 : 
                    #start another and keep this in the gold_holders
                    if len(current_exp_gold) > 0 : 
                        gold_exp.append(current_exp_gold)
                        if sum(gold_present_exp) > 0 :
                            gold_present_pred_exp.append(1)
                            gold_present_pred_exp_prop.append(float(sum(gold_present_exp))/len(gold_present_exp))
                        else : 
                            gold_present_pred_exp.append(0)
                            gold_present_pred_exp_prop.append(0.0)

                    gold_present_exp = []
                    current_exp_gold = []
                    if preds[i][j] == 1 or preds[i][j] == 2 : 
                        gold_present_exp.append(1)
                    else : 
                        gold_present_exp.append(0)
                    current_exp_gold.append(i)
                if y[i][j] == 2 and x[i][j] > 0 : 
                    if len(current_exp_gold) > 0 and current_exp_gold[-1] == i-1 : 
                        #Should never be the case!! but still need to troubleshoot it!
                        if preds[i][j] == 1 or preds[i][j] == 2 : 
                            gold_present_exp.append(1)
                        else : 
                            gold_present_exp.append(0)

                        current_exp_gold.append(i)
                    else : 
                        if len(current_exp_gold) > 0 : 
                            gold_exp.append(current_exp_gold)
                            if sum(gold_present_exp) > 0 :
                                gold_present_pred_exp.append(1)
                                gold_present_pred_exp_prop.append(float(sum(gold_present_exp))/len(gold_present_exp))

                            else : 
                                gold_present_pred_exp.append(0)
                                gold_present_pred_exp_prop.append(0.0)

                        gold_present_exp = []

                        current_exp_gold = []

                        if preds[i][j] == 1 or preds[i][j] == 2 : 
                            gold_present_exp.append(1)
                        else : 
                            gold_present_exp.append(0)

                        current_exp_gold.append(i)

                        
                if y[i][j] == 3 and x[i][j] > 0 : 
                    #start another and keep this in the gold_holders
                    if len(current_holder_gold) > 0 : 
                        gold_holders.append(current_holder_gold)
                        if sum(gold_present_holder) > 0 :
                            gold_present_pred_holder.append(1)
                            gold_present_pred_holder_prop.append(float(sum(gold_present_holder))/len(gold_present_holder))
                            
                        else : 
                            gold_present_pred_holder.append(0)
                            gold_present_pred_exp_prop.append(0.0)


                    gold_present_holder = []

                    current_holder_gold = []
                    
                    if preds[i][j] == 3 or preds[i][j] == 4 : 
                        gold_present_holder.append(1)
                    else : 
                        gold_present_holder.append(0)


                    current_holder_gold.append(i)
                if y[i][j] == 4 and x[i][j] > 0 : 
                    if len(current_holder_gold) > 0 and current_holder_gold[-1] == i-1 : 
                        #Should never be the case!! but still need to troubleshoot it!
                        if preds[i][j] == 3 or preds[i][j] == 4 : 
                            gold_present_holder.append(1)
                        else : 
                            gold_present_holder.append(0)

                        current_holder_gold.append(i)
                    else : 
                        if len(current_holder_gold) > 0 : 
                            gold_holders.append(current_holder_gold)
                            if sum(gold_present_holder) > 0 :
                                gold_present_pred_holder.append(1)
                                gold_present_pred_holder_prop.append(float(sum(gold_present_holder))/len(gold_present_holder))
                                
                            else : 
                                gold_present_pred_holder.append(0)
                                gold_present_pred_holder_prop.append(0.0)


                        gold_present_holder = []

                        current_holder_gold = []

                        if preds[i][j] == 3 or preds[i][j] == 4 : 
                            gold_present_holder.append(1)
                        else : 
                            gold_present_holder.append(0)

                        current_holder_gold.append(i)
                
                        
                if y[i][j] == 5 and x[i][j] > 0 : 
                    #start another and keep this in the gold_holders
                    if len(current_target_gold) > 0 : 
                        gold_targets.append(current_target_gold)
                        if sum(gold_present_target) > 0 :
                            gold_present_pred_target.append(1)
                            gold_present_pred_target_prop.append(float(sum(gold_present_target))/len(gold_present_target))
                            
                        else : 
                            gold_present_pred_target.append(0)
                            gold_present_pred_target_prop.append(0.0)


                    gold_present_target = []

                    current_target_gold = []
                    
                    if preds[i][j] == 5 or preds[i][j] == 6 : 
                        gold_present_target.append(1)
                    else : 
                        gold_present_target.append(0)

                    current_target_gold.append(i)

                if y[i][j] == 6 and x[i][j] > 0 : 
                    if len(current_target_gold) > 0 and current_target_gold[-1] == i-1 : 
                        #Should never be the case!! but till need to troubleshoot it!
                        if preds[i][j] == 5 or preds[i][j] == 6 : 
                            gold_present_target.append(1)
                        else : 
                            gold_present_target.append(0)

                        current_target_gold.append(i)
                    else : 

                        if len(current_target_gold) > 0 : 
                            gold_targets.append(current_target_gold)

                            if sum(gold_present_target) > 0 :
                                gold_present_pred_target.append(1)
                                gold_present_pred_target_prop.append(float(sum(gold_present_target))/len(gold_present_target))
                                
                            else : 
                                gold_present_pred_target.append(0)
                                gold_present_pred_target_prop.append(0.0)
                                                    
                        gold_present_target = []

                        current_holder_gold = []

                        if preds[i][j] == 5 or preds[i][j] == 6 : 
                            gold_present_target.append(1)
                        else : 
                            gold_present_target.append(0)
                        
                        current_target_gold.append(i)


                if preds[i][j] == 1 and x[i][j] > 0 : 
                    #start another and keep this in the gold_holders
                    if len(current_exp_pred) > 0 : 
                        pred_exp.append(current_exp_pred)
                        if sum(pred_present_exp) > 0 :
                            pred_present_gold_exp.append(1)
                            pred_present_gold_exp_prop.append(float(sum(pred_present_exp))/len(pred_present_exp))
                            
                        else : 
                            pred_present_gold_exp.append(0)
                            pred_present_gold_exp_prop.append(0.0)


                    pred_present_exp = []
                    current_exp_pred = []
                    
                    if y[i][j] == 1 or y[i][j] == 2 : 
                        pred_present_exp.append(1)
                    else : 
                        pred_present_exp.append(0)

                    current_exp_pred.append(i)
                if preds[i][j] == 2 and x[i][j] > 0 : 
                    if len(current_exp_pred) > 0 and current_exp_pred[-1] == i-1 : 
                        #Should never be the case!! but still need to troubleshoot it!
                        if y[i][j] == 1 or y[i][j] == 2 : 
                            pred_present_exp.append(1)
                        else : 
                            pred_present_exp.append(0)

                        current_exp_pred.append(i)
                    else : 
                        if len(current_exp_pred) > 0 : 
                            pred_exp.append(current_exp_pred)
                            if sum(pred_present_exp) > 0 :
                                pred_present_gold_exp.append(1)
                                pred_present_gold_exp_prop.append(float(sum(pred_present_exp))/len(pred_present_exp))
                            else : 
                                pred_present_gold_exp.append(0)
                                pred_present_gold_exp_prop.append(0.0)

                        pred_present_exp = []

                        current_exp_pred = []

                        if y[i][j] == 1 or y[i][j] == 2 : 
                            pred_present_exp.append(1)
                        else : 
                            pred_present_exp.append(0)

                        current_exp_pred.append(i)

                        
                if preds[i][j] == 3 and x[i][j] > 0 : 
                    #start another and keep this in the gold_holders
                    if len(current_holder_pred) > 0 : 
                        pred_holders.append(current_holder_pred)
                        if sum(pred_present_holder) > 0 :
                            pred_present_gold_holder.append(1)
                            pred_present_gold_holder_prop.append(float(sum(pred_present_holder))/len(pred_present_holder))
                        else : 
                            pred_present_gold_holder.append(0)
                            pred_present_gold_holder_prop.append(0.0)

                    pred_present_holder = []

                    current_holder_pred = []
                    
                    if y[i][j] == 3 or y[i][j] == 4 : 
                        pred_present_holder.append(1)
                    else : 
                        pred_present_holder.append(0)

                    current_holder_pred.append(i)
                if preds[i][j] == 4 and x[i][j] > 0 : 
                    if len(current_holder_pred) > 0 and current_holder_pred[-1] == i-1 : 
                        #Should never be the case!! but still need to troubleshoot it!
                        current_holder_pred.append(i)
                        if y[i][j] == 3 or y[i][j] == 4 : 
                            pred_present_holder.append(1)
                        else : 
                            pred_present_holder.append(0)

                    else : 
                        if len(current_holder_pred) > 0 : 
                            pred_holders.append(current_holder_pred)
                            if sum(pred_present_holder) > 0 :
                                pred_present_gold_holder.append(1)
                                pred_present_gold_holder_prop.append(float(sum(pred_present_holder))/len(pred_present_holder))
                                
                            else : 
                                pred_present_gold_holder.append(0)
                                pred_present_gold_exp_prop.append(0.0)


                        pred_present_holder = []

                        current_holder_pred = []
                        if y[i][j] == 3 or y[i][j] == 4 : 
                            pred_present_holder.append(1)
                        else : 
                            pred_present_holder.append(0)

                        current_holder_pred.append(i)
                
                        
                if preds[i][j] == 5 and x[i][j] > 0 : 
                    #start another and keep this in the gold_holders
                    if len(current_target_pred) > 0 : 
                        pred_targets.append(current_target_pred)
                        if sum(pred_present_target) > 0 :
                            pred_present_gold_target.append(1)
                            pred_present_gold_target_prop.append(float(sum(pred_present_target))/len(pred_present_target))
                            
                        else : 
                            pred_present_gold_target.append(0)
                            pred_present_gold_target_prop.append(0.0)


                    pred_present_target = []

                    current_target_pred = []
                    if y[i][j] == 5 or y[i][j] == 6 : 
                        pred_present_target.append(1)
                    else : 
                        pred_present_target.append(0)

                    current_target_pred.append(i)
                if preds[i][j] == 6 and x[i][j] > 0 : 
                    if len(current_target_pred) > 0 and current_target_pred[-1] == i-1 : 
                        #Should never be the case!! but still need to troubleshoot it!
                        if y[i][j] == 5 or y[i][j] == 6 : 
                            pred_present_target.append(1)
                        else : 
                            pred_present_target.append(0)

                        current_target_pred.append(i)
                    else : 
                        if len(current_target_pred) > 0 : 
                            pred_targets.append(current_target_pred)
                            
                            if sum(pred_present_target) > 0 :
                                pred_present_gold_target.append(1)
                                pred_present_gold_target_prop.append(float(sum(pred_present_target))/len(pred_present_target))
                            else : 
                                pred_present_gold_target.append(0)
                                pred_present_gold_target_prop.append(0.0)

                        pred_present_target = []

                        current_holder_pred = []
                        if y[i][j] == 5 or y[i][j] == 6 : 
                            pred_present_target.append(1)
                        else : 
                            pred_present_target.append(0)

                        current_holder_pred.append(i)
                
            #end here
            if len(current_exp_gold) > 0 : 
                gold_exp.append(current_exp_gold)
                if sum(gold_present_exp) > 0 :
                    gold_present_pred_exp_prop.append(float(sum(gold_present_exp))/len(gold_present_exp))
                    gold_present_pred_exp.append(1)

                else : 
                    gold_present_pred_exp_prop.append(0)
                    gold_present_pred_exp.append(0)

                    
            if len(current_holder_gold) > 0 : 
                gold_holders.append(current_holder_gold)
                if sum(gold_present_holder) > 0 :
                    gold_present_pred_holder_prop.append(float(sum(gold_present_holder))/len(gold_present_holder))
                    gold_present_pred_holder.append(1)
                else : 
                    gold_present_pred_holder_prop.append(0.0)
                    gold_present_pred_holder.append(0)

            if len(current_target_gold) > 0 : 
                gold_targets.append(current_target_gold)
                if sum(gold_present_target) > 0 :
                    gold_present_pred_target_prop.append(float(sum(gold_present_target))/len(gold_present_target))
                    gold_present_pred_target.append(1)
                else : 
                    gold_present_pred_target_prop.append(0.0)
                    gold_present_pred_target.append(0)


            if len(current_exp_pred) > 0 : 
                pred_exp.append(current_exp_pred)
                if sum(pred_present_exp) > 0 :
                    pred_present_gold_exp_prop.append(float(sum(pred_present_exp))/len(pred_present_exp))
                    pred_present_gold_exp.append(1)

                else : 
                    pred_present_gold_exp_prop.append(0.0)
                    pred_present_gold_exp.append(0)

                    
            if len(current_holder_pred) > 0 : 
                pred_holders.append(current_holder_pred)
                if sum(pred_present_holder) > 0 :
                    pred_present_gold_holder.append(1)
                    pred_present_gold_holder_prop.append(float(sum(pred_present_holder))/len(pred_present_holder))
                else : 
                    pred_present_gold_holder.append(0)
                    pred_present_gold_holder_prop.append(0)

            if len(current_target_pred) > 0 : 
                pred_targets.append(current_target_pred)
                if sum(pred_present_target) > 0 :
                    pred_present_gold_target.append(1)
                    pred_present_gold_target_prop.append(float(sum(pred_present_target))/len(pred_present_target))
                else : 
                    pred_present_gold_target.append(0)
                    pred_present_gold_target_prop.append(0.0)

            ####Now here do the analysiss!!!################
            All_pos_exp += len(gold_exp)
            TP_prec_exp += sum(pred_present_gold_exp)
            All_ans_exp += len(pred_exp)
            TP_rec_exp += sum(gold_present_pred_exp)
            
            TP_prec_exp_prop += sum(pred_present_gold_exp_prop)
            TP_rec_exp_prop += sum(gold_present_pred_exp_prop)

            All_pos_holder += len(gold_holders)
            TP_prec_holder += sum(pred_present_gold_holder)
            All_ans_holder += len(pred_holders)
            TP_rec_holder += sum(gold_present_pred_holder)
            
            TP_prec_holder_prop += sum(pred_present_gold_holder_prop)
            TP_rec_holder_prop += sum(gold_present_pred_holder_prop)

            All_pos_target += len(gold_targets)
            TP_prec_target += sum(pred_present_gold_target)
            All_ans_target += len(pred_targets)
            TP_rec_target += sum(gold_present_pred_target)
            
            TP_prec_target_prop += sum(pred_present_gold_target_prop)
            TP_rec_target_prop += sum(gold_present_pred_target_prop)

            
            #print str(TP_prec_exp)+"\t"+str(TP_prec_exp_prop)+"\t"+str(All_ans_exp)
            #print str(TP_rec_exp)+"\t"+ str(TP_rec_exp_prop)+"\t"+str(All_pos_exp)
            
            #if can_break == 1 : 
            #    break
            #if can_break == 1 : 
            #   break


    
    f_measure_all = 0.0
    f_measure_prop_all = 0.0
    f_measure_exact_all = 0.0
    
    final_pred = TP_rec_exp
    final_pred_prop = TP_rec_exp_prop
    #TP_prec_expr_exact += 1
    #TP_rec_expr_exact += 1
    final_pred_exact = TP_rec_exp_exact
    final_correct_expr = All_pos_exp

    final_pred_prec = TP_prec_exp
    final_pred_prec_prop = TP_prec_exp_prop
    final_pred_prec_exact = TP_prec_exp_exact
    final_correct_expr_prec = All_ans_exp

    print "Recall\t"+str(final_pred)+"\t"+str(final_pred_prop)+"\t"+str(final_pred_exact)+"\t"+str(final_correct_expr)
    print "Precision\t"+str(final_pred_prec)+"\t"+str(final_pred_prec_prop)+"\t"+str(final_pred_prec_exact)+"\t"+str(final_correct_expr_prec)
    
    recall = (numpy_floatX(final_pred)/final_correct_expr)
    precision = (numpy_floatX(final_pred_prec)/final_correct_expr_prec)
    f_measure = 2*recall*precision/(recall+precision)

    recall_prop = (numpy_floatX(final_pred_prop)/final_correct_expr)
    precision_prop = (numpy_floatX(final_pred_prec_prop)/final_correct_expr_prec)
    f_measure_prop = 2*recall_prop*precision_prop/(recall_prop+precision_prop)

    recall_exact = (numpy_floatX(final_pred_exact)/final_correct_expr)
    precision_exact = (numpy_floatX(final_pred_prec_exact)/final_correct_expr_prec)
    f_measure_exact = 2*recall_exact*precision_exact/(recall_exact+precision_exact)

    print "Recall\t"+str(recall)+"\t"+str(recall_prop)+"\t"+str(recall_exact)
    print "Precision\t"+str(precision)+"\t"+str(precision_prop)+"\t"+str(precision_exact)

    print "F-score\t Exp \t"+str(f_measure)+"\t"+str(f_measure_prop)+"\t"+str(f_measure_exact)

    f_measure_all += f_measure
    f_measure_prop_all += f_measure_prop
    f_measure_exact_all += f_measure_exact

    final_pred = TP_rec_holder
    final_pred_prop = TP_rec_holder_prop
    final_pred_exact = TP_rec_holder_exact
    final_correct_expr = All_pos_holder

    final_pred_prec = TP_prec_holder
    final_pred_prec_prop = TP_prec_holder_prop
    final_pred_prec_exact = TP_prec_holder_exact
    final_correct_expr_prec = All_ans_holder
    print "-------------"

    print "Recall\t"+str(final_pred)+"\t"+str(final_pred_prop)+"\t"+str(final_pred_exact)+"\t"+str(final_correct_expr)
    print "Precision\t"+str(final_pred_prec)+"\t"+str(final_pred_prec_prop)+"\t"+str(final_pred_prec_exact)+"\t"+str(final_correct_expr_prec)
    
    recall = (numpy_floatX(final_pred)/final_correct_expr)
    precision = (numpy_floatX(final_pred_prec)/final_correct_expr_prec)
    f_measure = 2*recall*precision/(recall+precision)

    recall_prop = (numpy_floatX(final_pred_prop)/final_correct_expr)
    precision_prop = (numpy_floatX(final_pred_prec_prop)/final_correct_expr_prec)
    f_measure_prop = 2*recall_prop*precision_prop/(recall_prop+precision_prop)

    recall_exact = (numpy_floatX(final_pred_exact)/final_correct_expr)
    precision_exact = (numpy_floatX(final_pred_prec_exact)/final_correct_expr_prec)
    f_measure_exact = 2*recall_exact*precision_exact/(recall_exact+precision_exact)

    print "Recall\t"+str(recall)+"\t"+str(recall_prop)+"\t"+str(recall_exact)
    print "Precision\t"+str(precision)+"\t"+str(precision_prop)+"\t"+str(precision_exact)
    
    print "F-score\t Holder \t"+str(f_measure)+"\t"+str(f_measure_prop)+"\t"+str(f_measure_exact)

    f_measure_all += f_measure
    f_measure_prop_all += f_measure_prop
    f_measure_exact_all += f_measure_exact

    final_pred = TP_rec_target
    final_pred_prop = TP_rec_target_prop
    final_pred_exact = TP_rec_target_exact
    final_correct_expr = All_pos_target

    final_pred_prec = TP_prec_target
    final_pred_prec_prop = TP_prec_target_prop
    final_pred_prec_exact = TP_prec_target_exact
    final_correct_expr_prec = All_ans_target

    print "-------------"
    print "Recall\t"+str(final_pred)+"\t"+str(final_pred_prop)+"\t"+str(final_pred_exact)+"\t"+str(final_correct_expr)
    print "Precision\t"+str(final_pred_prec)+"\t"+str(final_pred_prec_prop)+"\t"+str(final_pred_prec_exact)+"\t"+str(final_correct_expr_prec)

    recall = (numpy_floatX(final_pred)/final_correct_expr)
    precision = (numpy_floatX(final_pred_prec)/final_correct_expr_prec)
    f_measure = 2*recall*precision/(recall+precision)

    recall_prop = (numpy_floatX(final_pred_prop)/final_correct_expr)
    precision_prop = (numpy_floatX(final_pred_prec_prop)/final_correct_expr_prec)
    f_measure_prop = 2*recall_prop*precision_prop/(recall_prop+precision_prop)
    
    recall_exact = (numpy_floatX(final_pred_exact)/final_correct_expr)
    precision_exact = (numpy_floatX(final_pred_prec_exact)/final_correct_expr_prec)
    f_measure_exact = 2*recall_exact*precision_exact/(recall_exact+precision_exact)

    print "Recall\t"+str(recall)+"\t"+str(recall_prop)+"\t"+str(recall_exact)
    print "Precision\t"+str(precision)+"\t"+str(precision_prop)+"\t"+str(precision_exact)
    
    print "F-score\t Target\t"+str(f_measure)+"\t"+str(f_measure_prop)+"\t"+str(f_measure_exact)

    f_measure_all += f_measure
    f_measure_prop_all += f_measure_prop
    f_measure_exact_all += f_measure_exact

    print "Relations!!====================================================="
    
    f_measure1 = 0.0
    
    recall = (numpy_floatX(TP_holder_rec)/All_gold_holder)
    precision = (numpy_floatX(TP_holder_prec)/All_pred_holder)
    f_measure_holder = 2*recall*precision/(recall+precision)
    
    print "Recall\t"+str(TP_holder_rec)+"\t"+str(All_gold_holder)+"\t"+str(recall)
    print "Precision\t"+str(TP_holder_prec)+"\t"+str(All_pred_holder)+"\t"+str(precision)

    print "F-score\t Holder\t"+str(f_measure_holder)
    
    recall = (numpy_floatX(TP_target_rec)/All_gold_target)
    precision = (numpy_floatX(TP_target_prec)/All_pred_target)
    f_measure_target = 2*recall*precision/(recall+precision)

    print "Recall\t"+str(TP_target_rec)+"\t"+str(All_gold_target)+"\t"+str(recall)
    print "Precision\t"+str(TP_target_prec)+"\t"+str(All_pred_target)+"\t"+str(precision)
    print "F-score\t Target\t"+str(f_measure_target)
   
    f_measure1 = 2*f_measure_target*f_measure_holder/(f_measure_target+f_measure_holder)
    
    if f_measure1 >= 0 : 
        valid_err = 1. - f_measure1
    else : 
        valid_err = 1.

    predictions.close()
    
    return valid_err



def pred_error_relation4(f_pred, f_pred_hrel, f_pred_trel, prepare_data, prepare_data_words, find_relations_lrE, data, word_labels, iterator, tparams, window, relation,  verbose = False) :

    All_pos_exp = 0
    TP_prec_exp = 0
    All_ans_exp = 0
    TP_rec_exp = 0
    
    TP_prec_exp_prop = 0.0
    TP_rec_exp_prop = 0.0
    TP_prec_exp_exact = 0.0
    TP_rec_exp_exact = 0.0


    All_pos_holder = 0
    TP_prec_holder = 0
    All_ans_holder = 0
    TP_rec_holder = 0
    
    TP_prec_holder_prop = 0.0
    TP_rec_holder_prop = 0.0
    TP_prec_holder_exact = 0.0
    TP_rec_holder_exact = 0.0


    All_pos_target = 0
    TP_prec_target = 0
    All_ans_target = 0
    TP_rec_target = 0
    
    TP_prec_target_prop = 0.0
    TP_rec_target_prop = 0.0
    TP_prec_target_exact = 0.0
    TP_rec_target_exact = 0.0


    All_gold_holder = 0
    All_gold_target = 0
    All_pred_holder = 0
    All_pred_target = 0

    TP_holder_prec = 0
    TP_holder_rec = 0
    TP_target_prec = 0
    TP_target_rec = 0

    pred_all_relations_full = []
    pred_all_relations_y_full = []

    gold_holder_relations_full = []
    gold_target_relations_full = []

    predictions = open('./lstm_results_4_'+str(relation), 'w+')                                             

    for _, valid_index in iterator : 
        #print valid_index
        #print data
        #print len(data[2])

        x = [data[0][t] for t in valid_index]
        y = numpy.array(data[1])[valid_index]

        word_y = [word_labels[t] for t in valid_index]
        
        all_holder_relations = []
        all_target_relations = []
        
        holder_dim  = 1
        target_dim = 1

        x, mask, y = prepare_data([data[0][t] for t in valid_index], 
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)

        #print y
        
        #print "==========================="
                
        #preds = []


        preds1 = f_pred(x, mask)
        preds_h, preds_hw = f_pred_hrel(x, mask)
        preds_t, preds_tw = f_pred_trel(x, mask)

        old_preds = copy.deepcopy(preds1)
        preds = numpy.zeros((len(y), len(y[0])))


        #for every preds_h and preds_tw, mention if it is related or not related
        preds1 = correct_relations(y, preds1)
        preds_h = correct_relations(y, preds_h)
        preds_t = correct_relations(y, preds_t)

        for j in range(len(y[0])) : 
            for i in range(len(y)) : 
                #if two of these are the same then assign that label to preds
                if preds1[i][j] > 0 and preds_h[i][j] > 0 and  same_class(preds1[i][j], preds_h[i][j]) == 1:
                    preds[i][j] = preds1[i][j]
                if preds1[i][j] > 0 and preds_t[i][j] > 0 and  same_class(preds1[i][j], preds_t[i][j]) == 1:
                    preds[i][j] = preds1[i][j]
                if preds_h[i][j] > 0 and preds_t[i][j] > 0 and  same_class(preds_h[i][j], preds_t[i][j]) == 1:
                    preds[i][j] = preds_h[i][j]


        
        preds_h_entities = dict()
        preds_t_entities = dict()
        preds_h_spans = dict()
        preds_t_spans = dict()
        for j in range(len(y[0])) :
            entities_h = dict()
            entity_spans_h = dict()
            index = 0
            related_h = numpy.zeros((len(y)))
            for i in range(len(y)) :
                if preds_h[i][j] == 1 :
                    start_e = i
                    end_e = i
                    if i+1 < len(y) : 
                        while preds_h[i+1][j] == 2 : 
                            i += 1
                            end_e = i
                            if i+1 == len(y) :
                                break
                    index+=1
                    for p in range(start_e, end_e+1) : 
                        entities_h[p] = index
                    entity_spans_h[index] = (start_e, end_e)

                if preds_h[i][j] == 3 :                            
                    start_e = i
                    end_e = i
                    if i+1 < len(y) :
                        while preds_h[i+1][j] == 4 : 
                            i += 1
                            end_e = i
                            if i+1 == len(y) :
                                break
                    index+=1
                    for p in range(start_e, end_e+1) : 
                        entities_h[p] = index
                    entity_spans_h[index] = (start_e, end_e)

                if preds_h[i][j] == 5 :                            
                    start_e = i
                    end_e = i
                    if i+1 < len(y) :
                        while preds_h[i+1][j] == 6 : 
                            i += 1
                            end_e = i
                            if i+1 == len(y) :
                                break
                    index+=1
                    for p in range(start_e, end_e+1) : 
                        entities_h[p] = index
                    entity_spans_h[index] = (start_e, end_e)


            entities_t = dict()
            entity_spans_t = dict()
            index = 0
            related_t = numpy.zeros((len(y)))
            for i in range(len(y)) :
                if preds_t[i][j] == 1 :
                    start_e = i
                    end_e = i
                    if i+1 < len(y) :
                        while preds_t[i+1][j] == 2 : 
                            i += 1
                            end_e = i
                            if i+1 == len(y) :
                                break
                    index+=1
                    for p in range(start_e, end_e+1) : 
                        entities_t[p] = index
                    entity_spans_t[index] = (start_e, end_e)

                if preds_t[i][j] == 3 :                            
                    start_e = i
                    end_e = i
                    if i+1 < len(y) :
                        while preds_t[i+1][j] == 4 : 
                            i += 1
                            end_e = i
                            if i+1 == len(y) :
                                break
                    index+=1
                    for p in range(start_e, end_e+1) : 
                        entities_t[p] = index
                    entity_spans_t[index] = (start_e, end_e)

                if preds_t[i][j] == 5 :                            
                    start_e = i
                    end_e = i
                    if i+1 < len(y) :
                        while preds_t[i+1][j] == 6 : 
                            i += 1
                            end_e = i
                            if i+1 == len(y) :
                                break
                    index+=1
                    for p in range(start_e, end_e+1) : 
                        entities_t[p] = index
                    entity_spans_t[index] = (start_e, end_e)
            preds_h_entities[j] = entities_h
            preds_t_entities[j] = entities_t
            preds_h_spans[j] = entity_spans_h
            preds_t_spans[j] = entity_spans_t

            
                        
        #for k in numpy.arange(len(preds1)) : 
        #    preds.append(viterbi_segment(preds1[k], tparams))
        #print preds
        #x, mask, preds = prepare_data([data[0][t] for t in valid_index], 
        #                         preds,
        #                                                      maxlen=None)
        #print preds
        #print y
        targets = numpy.array(data[1])[valid_index]  #this is most probably is the array of the arrays!
        #is complicated. Either hard_target or the soft_target. Only hard_target implemented for now
        #print y        


        '''x, mask, y = prepare_data([data[0][t] for t in valid_index], 
                                  numpy.array(data[1])[valid_index],
                                                               maxlen=None)
        preds = f_pred(x, mask)
        #print preds
        targets = numpy.array(data[1])[valid_index]  #this is most probably is the array of the arrays!
        #is complicated. Either hard_target or the soft_target. Only hard_target implemented for now
        #print y      

        if "train" in file_data : 
            word_targets = numpy.array(word_labels[0])[valid_index]
        if "valid" in file_data : 
            word_targets = numpy.array(word_labels[1])[valid_index]
        if "test" in file_data : 
            word_targets = numpy.array(word_labels[2])[valid_index]

        temp, mask, gold = prepare_data_words([data[0][t] for t in valid_index],
                                                      word_targets, 
                                                          maxlen=None)'''


        word_targets = numpy.array(word_labels)[valid_index]
        temp, mask, gold = prepare_data_words([data[0][t] for t in valid_index],
                                                      word_targets, 
                                                          maxlen=None)

        for j in range(len(y[0])) :
            for i in range(len(y)) :
                if preds[i][j] > 6 : 
                    preds[i][j] = 0

        for j in range(len(y[0])) :
            for i in range(len(y)) :
                if y[i][j] > 6 : 
                    y[i][j] = 0



        for j in range(len(y[0])) :
            for i in range(len(y)) :
                if preds[i][j] == 0 : 
                    if (i+1) < len(y)-1 : 
                        if preds[i+1][j] > 0 and preds[i+1][j] < 3: 
                            preds[i+1][j] = 1
                        elif preds[i+1][j] > 2 and preds[i+1][j] < 5:
                            preds[i+1][j] = 3
                        elif preds[i+1][j] > 4 : 
                            preds[i+1][j] = 5

                if preds[i][j] > 0 and preds[i][j] < 3:
                    if (i+1) < len(y)-1 : 
                        if preds[i+1][j] > 0 and preds[i+1][j] < 3:
                            preds[i+1][j] = 2
                elif preds[i][j] > 2 and preds[i][j] < 5: 
                    if (i+1) < len(y)-1 : 
                        if preds[i+1][j] > 2 :
                            preds[i+1][j] = 4
                elif preds[i][j] > 4: 
                    if (i+1) < len(y)-1 : 
                        if preds[i+1][j] > 4 :
                            preds[i+1][j] = 6


                            
        new_preds = preds

        for j in range(len(x[0])) : 
            pred_holder = []
            pred_target = []
            pred_expr = []
            
            gold_holder_start = dict()
            gold_holder_end = dict()
            gold_target_start = dict()
            gold_target_end = dict()
            gold_expr_start = dict()
            gold_expr_end = dict()
            
            start_index_h = -1
            end_index_h = -1

            start_index_t = -1
            end_index_t = -1

            start_index_e = -1
            end_index_e = -1
            
            
            for i in range(len(x)) : 
                #if x[i][j] > 0 : 
                #    print str(index_dict[str(x[i][j])])+"\t"+str(gold[i][j])+"\t"+str(preds[i][j])
                if preds[i][j] == 3 and x[i][j] > 0: #beginning of a holder
                    start_index_h = i
                    if i+1 >= len(x) : 
                        pred_holder.append([start_index_h, start_index_h])
                        start_index_h = -1
                        end_index_h = -1
                        break
                    while (i+1) < len(x) and preds[i+1][j] == 4 : 
                        i+=1
                        end_index_h = i

                    if end_index_h == -1 : 
                        end_index_h = start_index_h
                    pred_holder.append([start_index_h, end_index_h])
                    start_index_h = -1
                    end_index_h = -1


                if preds[i][j] == 5 and x[i][j] > 0: 
                    start_index_t = i
                    if i+1 >= len(x) : 
                        pred_target.append([start_index_t, start_index_t])
                        start_index_t = -1
                        end_index_t = -1
                        break

                    while (i+1) < len(x) and preds[i+1][j] == 6 : 
                        i+=1
                        end_index_t = i

                    if end_index_t == -1 : 
                        end_index_t = start_index_t
                    pred_target.append([start_index_t, end_index_t])
                    start_index_t = -1
                    end_index_t = -1


                if preds[i][j] == 1 and x[i][j] > 0: 
                    start_index_e = i
                    if i+1 >= len(x) : 
                        pred_expr.append([start_index_e, start_index_e])
                        start_index_e = -1
                        end_index_e = -1
                        break

                    while (i+1) < len(x) and preds[i+1][j] == 2 : 
                        i+=1
                        end_index_e = i

                    if end_index_e == -1 : 
                        end_index_e = start_index_e
                    pred_expr.append([start_index_e, end_index_e])
                    start_index_e = -1
                    end_index_e = -1

                    
            for i in range(len(x)) : 
                if gold[i][j].startswith("B_AGENT") and len(gold[i][j].split("_")) > 2 : 
                    #print gold[i][j].split("_")
                    rel = []
                    for k in numpy.arange(1, len(gold[i][j].split("_REL"))) :
                        rel.append(int(gold[i][j].split("_REL")[k]))
                        gold_holder_start[rel[-1]] = i
                    
                    if i+1 >= len(x) : 
                        for r in rel : 
                            gold_holder_end[r] = i
                        break
                    
                    while i+1 < len(x) and gold[i+1][j].startswith("AGENT") :
                        i+=1
                        for r in rel : 
                            gold_holder_end[r] = i
                            
                    for r in rel  :
                        if r not in gold_holder_end.keys() : 
                            gold_holder_end[r] = gold_holder_start[r]
                        
                if gold[i][j].startswith("B_TARGET") and len(gold[i][j].split("_")) > 2 : 
                    #print gold[i][j].split("_")
                    #rel = int(gold[i][j].split("_")[-1][3:])
                    #gold_target_start[rel] = i
                    rel = []
                    for k in numpy.arange(1, len(gold[i][j].split("_REL"))) : 
                        rel.append(int(gold[i][j].split("_REL")[k]))
                        gold_target_start[rel[-1]] = i

                    if i+1 >= len(x) :
                        for  r in rel : 
                            gold_target_end[r] = i
                        break

                    while i+1 < len(x) and gold[i+1][j].startswith("TARGET") :
                        i+=1
                        for  r in rel : 
                            gold_target_end[r] = i

                    for r in rel : 
                        if r not in gold_target_end.keys() : 
                            gold_target_end[r] = gold_target_start[r]

                if gold[i][j].startswith("B_DSE") and len(gold[i][j].split("_")) > 2: 
                    #print gold[i][j].split("_")
                    #rel = int(gold[i][j].split("_")[-1][3:])
                    #gold_expr_start[rel] = i

                    rel = []
                    for k in numpy.arange(1, len(gold[i][j].split("_REL"))) : 
                        rel.append(int(gold[i][j].split("_REL")[k]))
                        gold_expr_start[rel[-1]] = i

                    
                    if i+1 >= len(x) : 
                        for r in rel : 
                            gold_expr_end[r] = i
                        break

                    while i+1 < len(x) and gold[i+1][j].startswith("DSE") :
                        i+=1
                        for r in rel : 
                            gold_expr_end[r] = i

                    for r in rel : 
                        if r not in gold_expr_end.keys() : 
                            gold_expr_end[r] = gold_expr_start[r]

            #######Now all the respective annotations are done!!!!

            gold_holder_relations = []
            gold_target_relations = []

            #pred_holder_relations = []
            #pred_target_relations = []

            pred_all_relations = []
            pred_all_relations_y = []

            for key in gold_expr_start.keys() : 
                #print key
                start_e = gold_expr_start[key]
                #print gold_expr_start.keys()
                #print gold_expr_end.keys()
                end_e = gold_expr_end[key]
                if key in gold_holder_start.keys() : 
                    start_h = gold_holder_start[key]
                    end_h = gold_holder_end[key]
                    gold_holder_relations.append([start_h, end_h, start_e, end_e])

                if key in gold_target_start.keys() : 
                    start_t = gold_target_start[key]
                    end_t = gold_target_end[key]
                    gold_target_relations.append([start_t, end_t, start_e, end_e])

            #gold_holder_relations_full.append(gold_holder_relations)
            #gold_target_relations_full.append(gold_target_relations)
            
            #For the predicted all pairs are the ones

            
            holder_index = dict()
            target_index = dict()
            expr_index = dict()
            
            for i in range(len(x)) : 
                holder_index[i] = []
                target_index[i] = []
                expr_index[i] = []


            for i in range(len(x)) : 
                if preds[i][j] > 0  and x[i][j] > 0 : 
                    if preds[i][j] == 1 or preds[i][j] == 2 : 
                        for p in range(len(pred_expr)) :
                            [start_e, end_e] = pred_expr[p]
                            if start_e <= i and i<= end_e : 
                                expr_index[i].append(p)
                    if preds[i][j] == 3 or preds[i][j] == 4 : 
                        for p in range(len(pred_holder)) :
                            [start_h, end_h] = pred_holder[p]
                            if start_h <= i and i<= end_h : 
                                holder_index[i].append(p)
                    if preds[i][j] == 5 or preds[i][j] == 6 : 
                        for p in range(len(pred_target)) :
                            [start_t, end_t] = pred_target[p]
                            if start_t <= i and i<= end_t : 
                                target_index[i].append(p)
            

            #preds_h, preds_hw = f_pred_hrel(x, mask)
            #preds_t, preds_tw = f_pred_trel(x, mask)

            
            def Concat_append(pair, entity) : 
                [start_index, end_index] = pair
                
                for i in range(len(entity)) :
                    [start_e, end_e] = entity[i]
                    if end_e == start_index-1 : 
                        entity[i] = [start_e, end_index]
                        return [start_e, end_index], entity

                    if start_e == end_index+1 : 
                        entity[i] = [start_index, end_e]
                        return [start_index, end_e], entity
                        
                entity.append([start_index, end_index])
                return [start_index, end_index], entity


            pred_holder_relations = []
            pred_target_relations = []

            #preds_h_entities[j] = entities_h
            #preds_t_entities[j] = entities_t
            #preds_h_spans[j] = entity_span_h
            #preds_t_spans = entity_span_t


            
            #these many windows to the left
            for i in range(len(x)) : 
                if (i-preds_hw[i][j]) < 0 : 
                    continue
                if preds_hw[i][j] > 0 and x[i][j] > 0 : 
                    if len(expr_index[i]) > 0 : 
                        if (i-preds_hw[i][j]) >= 0 and len(holder_index[i-preds_hw[i][j]]) > 0 :
                            for ei in expr_index[i] : 
                                [start_e, end_e] = pred_expr[ei]
                                for hi in holder_index[i-preds_hw[i][j]] : 
                                    [start_h, end_h] = pred_holder[hi]
                                    if [start_h, end_h, start_e, end_e] not in pred_holder_relations : 
                                        pred_holder_relations.append([start_h, end_h, start_e, end_e])
                        elif (i-preds_hw[i][j]) >= 0 and (preds_h[i-preds_hw[i][j]][j] == 3 or preds_h[i-preds_hw[i][j]][j] == 4) and preds[i-preds_hw[i][j]][j] == 0: 
                            if (i-preds_hw[i][j]) in preds_h_entities[j].keys() : 
                                (start_h, end_h) = preds_h_spans[j][preds_h_entities[j][(i-preds_hw[i][j])]]
                                [start_h, end_h], pred_holder = Concat_append([start_h, end_h], pred_holder)
                                for index in range(start_h, end_h+1) : 
                                    if preds[index][j] == 0  :
                                        preds[index][j] = preds_h[index][j]
                                for ei in expr_index[i] : 
                                    [start_e, end_e] = pred_expr[ei]
                                    if [start_h, end_h, start_e, end_e] not in pred_holder_relations : 
                                        pred_holder_relations.append([start_h, end_h, start_e, end_e])
                    else : 
                        if (i-preds_hw[i][j]) >= 0 and len(holder_index[i-preds_hw[i][j]]) > 0 and (preds_h[i][j] == 1 or preds_h[i][j]==2) and preds[i][j] == 0:
                            if (i) in preds_h_entities[j].keys() : 
                                (start_e, end_e) = preds_h_spans[j][preds_h_entities[j][(i)]]
                                [start_e, end_e], pred_expr = Concat_append([start_e, end_e], pred_expr)
                                for index in range(start_e, end_e+1) : 
                                    if preds[index][j] == 0  :
                                        preds[index][j] = preds_h[index][j]
                                for hi in holder_index[i-preds_hw[i][j]] : 
                                    [start_h, end_h] = pred_holder[hi]
                                    if [start_h, end_h, start_e, end_e] not in pred_holder_relations : 
                                        pred_holder_relations.append([start_h, end_h, start_e, end_e])

                                    
                    if len(target_index[i]) > 0 : 
                        if (i-preds_hw[i][j]) >= 0 and len(expr_index[i-preds_hw[i][j]]) > 0 :
                            for ti in target_index[i] : 
                                [start_t, end_t] = pred_target[ti]
                                for ei in expr_index[i-preds_hw[i][j]] : 
                                    [start_e, end_e] = pred_expr[ei]
                                    if [start_t, end_t, start_e, end_e] not in pred_target_relations : 
                                        pred_target_relations.append([start_t, end_t, start_e, end_e])

                        elif (i-preds_hw[i][j]) >= 0 and (preds_h[i-preds_hw[i][j]][j] == 1 or preds_h[i-preds_hw[i][j]][j] == 2) and preds[i-preds_hw[i][j]][j] == 0: 
                            if (i-preds_hw[i][j]) in preds_h_entities[j].keys() : 
                                (start_e, end_e) = preds_h_spans[j][preds_h_entities[j][(i-preds_hw[i][j])]]
                                [start_e, end_e], pred_expr = Concat_append([start_e, end_e], pred_expr)
                                for index in range(start_e, end_e+1) : 
                                    if preds[index][j] == 0  :
                                        preds[index][j] = preds_h[index][j]
                                for ti in target_index[i] : 
                                    [start_t, end_t] = pred_target[ti]
                                    if [start_t, end_t, start_e, end_e] not in pred_target_relations : 
                                        pred_target_relations.append([start_t, end_t, start_e, end_e])

                    else : 
                        if (i-preds_hw[i][j]) >= 0 and len(expr_index[i-preds_hw[i][j]]) > 0 and (preds_h[i][j] == 5 or preds_h[i][j]==6) and preds[i][j] == 0:
                            if i in preds_h_entities[j].keys() : 
                                (start_t, end_t) = preds_h_spans[j][preds_h_entities[j][(i)]]
                                [start_t, end_t], pred_target = Concat_append([start_t, end_t], pred_target)
                                for index in range(start_t, end_t+1) : 
                                    if preds[index][j] == 0  :
                                        preds[index][j] = preds_h[index][j]
                                for ei in expr_index[i-preds_hw[i][j]] : 
                                    [start_e, end_e] = pred_expr[ei]
                                    if [start_t, end_t, start_e, end_e] not in pred_target_relations : 
                                        pred_target_relations.append([start_t, end_t, start_e, end_e])

                        

            for i in range(len(x)) : 
                if (i+preds_tw[i][j]) >= len(x) : 
                    continue
                if preds_tw[i][j] > 0 and x[i][j] > 0 : 
                    if len(expr_index[i]) > 0 : 
                        if (i+preds_tw[i][j]) < len(x)  and len(holder_index[i+preds_tw[i][j]]) > 0 :
                            for ei in expr_index[i] : 
                                [start_e, end_e] = pred_expr[ei]
                                for hi in holder_index[i+preds_tw[i][j]] : 
                                    [start_h, end_h] = pred_holder[hi]
                                    if [start_h, end_h, start_e, end_e] not in pred_holder_relations : 
                                        pred_holder_relations.append([start_h, end_h, start_e, end_e])
                        elif (i+preds_tw[i][j]) < len(x) and (preds_t[i+preds_tw[i][j]][j] == 3 or preds_t[i+preds_tw[i][j]][j] == 4) and preds[i+preds_tw[i][j]][j] == 0: 
                            if (i+preds_tw[i][j]) in preds_t_entities[j].keys() : 
                                (start_h, end_h) = preds_t_spans[j][preds_t_entities[j][(i+preds_tw[i][j])]]
                                [start_h, end_h], pred_holder = Concat_append([start_h, end_h], pred_holder)
                                for index in range(start_h, end_h+1) : 
                                    if preds[index][j] == 0  :
                                        preds[index][j] = preds_t[index][j]
                                for ei in expr_index[i] : 
                                    [start_e, end_e] = pred_expr[ei]
                                    if [start_h, end_h, start_e, end_e] not in pred_holder_relations : 
                                        pred_holder_relations.append([start_h, end_h, start_e, end_e])

                    else : 
                        if (i+preds_tw[i][j]) < len(x) and len(holder_index[i+preds_tw[i][j]]) > 0 and (preds_t[i][j] == 1 or preds_t[i][j]==2) and preds[i][j] == 0:
                            if (i) in preds_t_entities[j].keys() : 
                                (start_e, end_e) = preds_t_spans[j][preds_t_entities[j][(i)]]
                                [start_e, end_e], pred_expr = Concat_append([start_e, end_e], pred_expr)
                                for index in range(start_e, end_e+1) : 
                                    if preds[index][j] == 0  :
                                        preds[index][j] = preds_t[index][j]
                                for hi in holder_index[i+preds_tw[i][j]] : 
                                    [start_h, end_h] = pred_holder[hi]
                                    if [start_h, end_h, start_e, end_e] not in pred_holder_relations : 
                                        pred_holder_relations.append([start_h, end_h, start_e, end_e])
                                
                                    
                    if len(target_index[i]) > 0 : 
                        if (i+preds_tw[i][j]) < len(x) and len(expr_index[i+preds_tw[i][j]]) > 0 :
                            for ti in target_index[i] : 
                                [start_t, end_t] = pred_target[ti]
                                for ei in expr_index[i+preds_tw[i][j]] : 
                                    [start_e, end_e] = pred_expr[ei]
                                    if [start_t, end_t, start_e, end_e] not in pred_target_relations : 
                                        pred_target_relations.append([start_t, end_t, start_e, end_e])
                                
                        elif (i+preds_tw[i][j]) < len(x) and (preds_t[i+preds_tw[i][j]][j] == 1 or preds_t[i+preds_tw[i][j]][j] == 2) and preds[i+preds_tw[i][j]][j] == 0: 
                            if (i+preds_tw[i][j]) in preds_t_entities[j].keys() : 
                                (start_e, end_e) = preds_t_spans[j][preds_t_entities[j][(i+preds_tw[i][j])]]
                                [start_e, end_e], pred_expr = Concat_append([start_e, end_e], pred_expr)
                                for index in range(start_e, end_e+1) : 
                                    if preds[index][j] == 0  :
                                        preds[index][j] = preds_t[index][j]
                                for ti in target_index[i] : 
                                    [start_t, end_t] = pred_target[ti]
                                    if [start_t, end_t, start_e, end_e] not in pred_target_relations : 
                                        pred_target_relations.append([start_t, end_t, start_e, end_e])

                    else : 
                        if (i+preds_tw[i][j]) < len(x) and len(expr_index[i+preds_tw[i][j]]) > 0 and (preds_t[i][j] == 5 or preds_t[i][j]==6) and preds[i][j] == 0:
                            if i in preds_t_entities[j].keys() : 
                                (start_t, end_t) = preds_t_spans[j][preds_t_entities[j][(i)]]
                                [start_t, end_t], pred_target = Concat_append([start_t, end_t], pred_target)
                                for index in range(start_t, end_t+1) : 
                                    if preds[index][j] == 0  :
                                        preds[index][j] = preds_t[index][j]
                                for ei in expr_index[i+preds_tw[i][j]] : 
                                    [start_e, end_e] = pred_expr[ei]
                                    if [start_t, end_t, start_e, end_e] not in pred_target_relations : 
                                        pred_target_relations.append([start_t, end_t, start_e, end_e])
                

            '''for [start_e, end_e] in pred_expr : 
                for [start_h, end_h] in pred_holder : 
                    pred_all_relations.append([start_h, end_h, start_e, end_e])
                    pred_all_relations_y.append(1)

                for [start_t, end_t] in pred_target : 
                    pred_all_relations.append([start_t, end_t, start_e, end_e])
                    pred_all_relations_y.append(2)'''

            
            #pred_all_relations_full.append(pred_all_relations)
            #pred_all_relations_y_full.append(pred_all_relations_y)


            #print "Gold=========================================="
            #print pred_holder_relations
            #print gold_holder_relations
            #print gold_target_relations

            
            #print "Pred=========================================="
            #print pred_holder_relations
            #print pred_all_relations
            #print pred_all_relations_y

            
            #print pred_all_relations
            #print pred_all_relations_y

            #if len(pred_all_relations) == 0 : 
            #    temp = []
            #    pred_all_relations.append(temp)

            #pred_all_relations_full = []
            #pred_all_relations_full_y = []

            #if not len(pred_all_relations[0]) == 0 : 
            #pred_all_relations_full.append(pred_all_relations)
            #pred_all_relations_full_y.append(pred_all_relations_y)

            #print "Before===="
            #print pred_all_relations_full
            #print pred_all_relations_full_y
            
            #relations_x, mask_t, relations_y = prepare_data_relations(pred_all_relations_full, pred_all_relations_full_y)

            #print "After relations prep==="
            #print relations_x
            #print relations_y

            #pred_relations = f_pred_relations(x, mask, relations_x)

            #print pred_relations

            #pred_holder_relations = []
            #pred_target_relations = []

            '''for l in range(len(pred_relations[0])) :
                for k in range(len(pred_relations)) :
                    if pred_relations[k][l] == 1 : 
                        pred_holder_relations.append(relations_x[k][l])
                    if pred_relations[k][l] == 2 : 
                        pred_target_relations.append(relations_x[k][l])
                        #pred_holder_relations_all.append(pred_holder_relations)
                        #pred_target_relations_all.append(pred_target_relations)'''
            
            #print "-----------------------------------------------------------------"
            #print gold_holder_relations
            #print gold_target_relations
            #print "Predicted relation=========================================="
            #print pred_holder_relations
            #print pred_target_relations


            #preds = new_preds

            pred_holder_later = []
            for [sh1, eh1, se1, ee1] in pred_holder_relations : 
                for [sh2, eh2, se2, ee2] in pred_holder_later : 
                    if sh1==sh2 and eh1 == eh2 and se1 == se2 and ee1 == ee2 : 
                        break
                pred_holder_later.append([sh1, eh1, se1, ee1])

            pred_holder_relations = pred_holder_later

            pred_target_later = []
            for [sh1, eh1, se1, ee1] in pred_target_relations : 
                for [sh2, eh2, se2, ee2] in pred_target_later : 
                    if sh1==sh2 and eh1 == eh2 and se1 == se2 and ee1 == ee2 : 
                        break
                pred_target_later.append([sh1, eh1, se1, ee1])

            pred_target_relations = pred_target_later
            
            
            
            final_tag = []
            for i in range(len(x)) :
                if preds[i][j] == 1 :
                    final_tag.append("B_DSE")
                if preds[i][j] == 2 :
                    final_tag.append("DSE")
                if preds[i][j] == 3 :
                    final_tag.append("B_AGENT")
                if preds[i][j] == 4 :
                    final_tag.append("AGENT")
                if preds[i][j] == 5 :
                    final_tag.append("B_TARGET")
                if preds[i][j] == 6 :
                    final_tag.append("TARGET")
                if preds[i][j] == 0 :
                    final_tag.append("O")
                if preds[i][j] == 7 : 
                    final_tag.append("B_ESE")
                if preds[i][j] == 8 : 
                    final_tag.append("ESE")
                if preds[i][j] == 9 :
                    final_tag.append("B_OBJ")
                if preds[i][j] == 10 : 
                    final_tag.append("OBJ")
            
            count_h = 1
            count_t = 1
            
            for [sh, eh, se, ee] in pred_holder_relations : 
                for ind_h in numpy.arange(sh, eh+1) : 
                    final_tag[ind_h] = final_tag[ind_h]+"_-"+str(count_h)
                for ind_e in numpy.arange(se, ee+1) : 
                    final_tag[ind_e] = final_tag[ind_e]+"_-"+str(count_h)
                count_h+=1
            
            #count_t = 1
            for [st, et, se, ee] in pred_target_relations :
                for ind_t in numpy.arange(st, et+1) :
                    final_tag[ind_t] = final_tag[ind_t]+"_"+str(count_t)
                for ind_e in numpy.arange(se, ee+1) :
                    final_tag[ind_e] = final_tag[ind_e]+"_"+str(count_t)
                count_t+=1

            #remove all the ones with no relations at place of holders and targets!!

            '''for i in range(len(x)) : 
                if final_tag[i] == 'B_AGENT' or final_tag[i] == 'AGENT' or final_tag[i] == 'B_TARGET' or final_tag[i] == 'TARGET': 
                    final_tag[i] = 'O'
                    preds[i][j] = 0'''


            for i in range(len(x)) :
                predictions.write(str(index_dict[str(x[i][j])]))
                predictions.write("\t")
                '''predictions.write(str(y[i][j]))
                predictions.write("\t")
                predictions.write(str(int(preds[i][j])))
                predictions.write("\t")
                predictions.write(str(old_preds[i][j]))
                predictions.write("\t")
                predictions.write(str(preds_h[i][j]))
                predictions.write("\t")
                predictions.write(str(preds_t[i][j]))
                predictions.write("\t")
                predictions.write(str(preds_hw[i][j]))
                predictions.write("\t")
                predictions.write(str(preds_tw[i][j]))
                predictions.write("\t")'''
                predictions.write(str(word_y[0][i]))
                predictions.write("\t")
                predictions.write(str(final_tag[i]))

                predictions.write("\n")
            #predictions.write(str(word_y))
            #predictions.write("\n")
            predictions.write("\n")


            for [pstart_h, pend_h, pstart_e, pend_e] in pred_holder_relations : 
                for [gstart_h, gend_h, gstart_e, gend_e] in gold_holder_relations : 
                    list1 = numpy.arange(pstart_h, pend_h+1)
                    list2 = numpy.arange(gstart_h, gend_h+1)

                    list3 = numpy.arange(pstart_e, pend_e+1)
                    list4 = numpy.arange(gstart_e, gend_e+1)


                    if len(list(set(list1) & set(list2))) > 0 and len(list(set(list3) & set(list4))) > 0 :
                        TP_holder_prec+=1


            for [gstart_h, gend_h, gstart_e, gend_e] in gold_holder_relations : 
                for [pstart_h, pend_h, pstart_e, pend_e] in pred_holder_relations : 
                    list1 = numpy.arange(pstart_h, pend_h+1)
                    list2 = numpy.arange(gstart_h, gend_h+1)

                    list3 = numpy.arange(pstart_e, pend_e+1)
                    list4 = numpy.arange(gstart_e, gend_e+1)

                    if len(list(set(list1) & set(list2))) > 0 and len(list(set(list3) & set(list4))) > 0 :
                        TP_holder_rec+=1

            for [pstart_t, pend_t, pstart_e, pend_e] in pred_target_relations : 
                for [gstart_t, gend_t, gstart_e, gend_e] in gold_target_relations : 
                    list1 = numpy.arange(pstart_t, pend_t+1)
                    list2 = numpy.arange(gstart_t, gend_t+1)

                    list3 = numpy.arange(pstart_e, pend_e+1)
                    list4 = numpy.arange(gstart_e, gend_e+1)

                    if len(list(set(list1) & set(list2))) > 0 and len(list(set(list3) & set(list4))) > 0 : 
                        TP_target_prec+=1

            for [gstart_t, gend_t, gstart_e, gend_e] in gold_target_relations : 
                for [pstart_t, pend_t, pstart_e, pend_e] in pred_target_relations : 
                    list1 = numpy.arange(pstart_t, pend_t+1)
                    list2 = numpy.arange(gstart_t, gend_t+1)

                    list3 = numpy.arange(pstart_e, pend_e+1)
                    list4 = numpy.arange(gstart_e, gend_e+1)

                    if len(list(set(list1) & set(list2))) > 0 and len(list(set(list3) & set(list4))) > 0 :
                        TP_target_rec+=1

            All_gold_holder += len(gold_holder_relations)
            All_gold_target += len(gold_target_relations)
            All_pred_holder += len(pred_holder_relations)
            All_pred_target += len(pred_target_relations)
            
            #print "HOlder\t"+str(temp_holder_pred)
            #print "Gold\t"+str(temp_holder_gold)
        
        #preds = new_preds
        

        #########################################################Relation part!!!###########################################

        for j in range(len(y[0])) :
            for i in range(len(y)) :
                if preds[i][j] == 0 : 
                    if (i+1) < len(y)-1 : 
                        if preds[i+1][j] > 0 and preds[i+1][j] < 3: 
                            preds[i+1][j] = 1
                        elif preds[i+1][j] > 2 and preds[i+1][j] < 5:
                            preds[i+1][j] = 3
                        elif preds[i+1][j] > 4 : 
                            preds[i+1][j] = 5

                if preds[i][j] > 0 and preds[i][j] < 3:
                    if (i+1) < len(y)-1 : 
                        if preds[i+1][j] > 0 and preds[i+1][j] < 3:
                            preds[i+1][j] = 2
                elif preds[i][j] > 2 and preds[i][j] < 5: 
                    if (i+1) < len(y)-1 : 
                        if preds[i+1][j] > 2 :
                            preds[i+1][j] = 4
                elif preds[i][j] > 4: 
                    if (i+1) < len(y)-1 : 
                        if preds[i+1][j] > 4 :
                            preds[i+1][j] = 6


        for j in range(len(x[0])) : 

            pred_holder = []
            pred_target = []
            pred_expr = []
                        
            start_index_h = -1
            end_index_h = -1

            start_index_t = -1
            end_index_t = -1

            start_index_e = -1
            end_index_e = -1
            
            
            for i in range(len(x)) : 
                #if x[i][j] > 0 : 
                #    print str(index_dict[str(x[i][j])])+"\t"+str(gold[i][j])+"\t"+str(preds[i][j])
                if preds[i][j] == 3 and x[i][j] > 0: #beginning of a holder
                    start_index_h = i
                    if i+1 >= len(x) : 
                        pred_holder.append([start_index_h, start_index_h])
                        start_index_h = -1
                        end_index_h = -1
                        break
                    while (i+1) < len(x) and preds[i+1][j] == 4 : 
                        i+=1
                        end_index_h = i

                    if end_index_h == -1 : 
                        end_index_h = start_index_h
                    pred_holder.append([start_index_h, end_index_h])
                    start_index_h = -1
                    end_index_h = -1


                if preds[i][j] == 5 and x[i][j] > 0: 
                    start_index_t = i
                    if i+1 >= len(x) : 
                        pred_target.append([start_index_t, start_index_t])
                        start_index_t = -1
                        end_index_t = -1
                        break

                    while (i+1) < len(x) and preds[i+1][j] == 6 : 
                        i+=1
                        end_index_t = i

                    if end_index_t == -1 : 
                        end_index_t = start_index_t
                    pred_target.append([start_index_t, end_index_t])
                    start_index_t = -1
                    end_index_t = -1


                if preds[i][j] == 1 and x[i][j] > 0: 
                    start_index_e = i
                    if i+1 >= len(x) : 
                        pred_expr.append([start_index_e, start_index_e])
                        start_index_e = -1
                        end_index_e = -1
                        break

                    while (i+1) < len(x) and preds[i+1][j] == 2 : 
                        i+=1
                        end_index_e = i

                    if end_index_e == -1 : 
                        end_index_e = start_index_e
                    pred_expr.append([start_index_e, end_index_e])
                    start_index_e = -1
                    end_index_e = -1


            gold_holder = []
            gold_target = []
            gold_expr = []
                        
            start_index_h = -1
            end_index_h = -1

            start_index_t = -1
            end_index_t = -1

            start_index_e = -1
            end_index_e = -1
            
            
            for i in range(len(x)) : 
                #if x[i][j] > 0 : 
                #    print str(index_dict[str(x[i][j])])+"\t"+str(gold[i][j])+"\t"+str(preds[i][j])
                if y[i][j] == 3 and x[i][j] > 0: #beginning of a holder
                    start_index_h = i
                    if i+1 >= len(x) : 
                        gold_holder.append([start_index_h, start_index_h])
                        start_index_h = -1
                        end_index_h = -1
                        break
                    while (i+1) < len(x) and y[i+1][j] == 4 : 
                        i+=1
                        end_index_h = i

                    if end_index_h == -1 : 
                        end_index_h = start_index_h
                    gold_holder.append([start_index_h, end_index_h])
                    start_index_h = -1
                    end_index_h = -1


                if y[i][j] == 5 and x[i][j] > 0: 
                    start_index_t = i
                    if i+1 >= len(x) : 
                        gold_target.append([start_index_t, start_index_t])
                        start_index_t = -1
                        end_index_t = -1
                        break

                    while (i+1) < len(x) and y[i+1][j] == 6 : 
                        i+=1
                        end_index_t = i

                    if end_index_t == -1 : 
                        end_index_t = start_index_t
                    gold_target.append([start_index_t, end_index_t])
                    start_index_t = -1
                    end_index_t = -1


                if y[i][j] == 1 and x[i][j] > 0: 
                    start_index_e = i
                    if i+1 >= len(x) : 
                        gold_expr.append([start_index_e, start_index_e])
                        start_index_e = -1
                        end_index_e = -1
                        break

                    while (i+1) < len(x) and y[i+1][j] == 2 : 
                        i+=1
                        end_index_e = i

                    if end_index_e == -1 : 
                        end_index_e = start_index_e
                    gold_expr.append([start_index_e, end_index_e])
                    start_index_e = -1
                    end_index_e = -1


            #do analysis and add to the global counts
            #TP_prec_target_exact = 0.0
            #TP_rec_target_exact = 0.0

            
            for [pstarte, pende] in pred_holder : 
                for [gstarte, gende] in gold_holder : 
                    if pstarte == gstarte and pende == gende : 
                        TP_prec_holder_exact += 1
                        TP_rec_holder_exact += 1
                        break

            for [pstarte, pende] in pred_target : 
                for [gstarte, gende] in gold_target : 
                    if pstarte == gstarte and pende == gende : 
                        TP_prec_target_exact += 1
                        TP_rec_target_exact += 1
                        break

            for [pstarte, pende] in pred_expr : 
                for [gstarte, gende] in gold_expr : 
                    if pstarte == gstarte and pende == gende : 
                        TP_prec_exp_exact += 1
                        TP_rec_exp_exact += 1
                        break
            
            
            
        for j in range(len(y[0])) :
            list_expr = []
            num_overlap = 0
            total_expr = 0

            gold_holders = []
            gold_targets = []
            gold_exp = []

            gold_present_pred_holder = []
            gold_present_pred_target = []
            gold_present_pred_exp = []

            gold_present_pred_holder_prop = []
            gold_present_pred_target_prop = []
            gold_present_pred_exp_prop = []



            gold_present_holder = []
            gold_present_target = []
            gold_present_exp = []

            current_holder_gold = []
            current_target_gold = []
            current_exp_gold = []

            pred_holders = []
            pred_targets = []
            pred_exp = []

            pred_present_gold_holder = []
            pred_present_gold_target = []
            pred_present_gold_exp = []

            pred_present_gold_holder_prop = []
            pred_present_gold_target_prop = []
            pred_present_gold_exp_prop = []

            pred_present_holder = []
            pred_present_target = []
            pred_present_exp = []

            current_holder_pred = []
            current_target_pred = []
            current_exp_pred = []


            for i in range(len(y)) :

                #So preds[i][j] contains the true labels and y[i][j] contains the predicted labels
                if y[i][j] == 1 and x[i][j] > 0 : 
                    #start another and keep this in the gold_holders
                    if len(current_exp_gold) > 0 : 
                        gold_exp.append(current_exp_gold)
                        if sum(gold_present_exp) > 0 :
                            gold_present_pred_exp.append(1)
                            gold_present_pred_exp_prop.append(float(sum(gold_present_exp))/len(gold_present_exp))
                        else : 
                            gold_present_pred_exp.append(0)
                            gold_present_pred_exp_prop.append(0.0)

                    gold_present_exp = []
                    current_exp_gold = []
                    if preds[i][j] == 1 or preds[i][j] == 2 : 
                        gold_present_exp.append(1)
                    else : 
                        gold_present_exp.append(0)
                    current_exp_gold.append(i)
                if y[i][j] == 2 and x[i][j] > 0 : 
                    if len(current_exp_gold) > 0 and current_exp_gold[-1] == i-1 : 
                        #Should never be the case!! but still need to troubleshoot it!
                        if preds[i][j] == 1 or preds[i][j] == 2 : 
                            gold_present_exp.append(1)
                        else : 
                            gold_present_exp.append(0)

                        current_exp_gold.append(i)
                    else : 
                        if len(current_exp_gold) > 0 : 
                            gold_exp.append(current_exp_gold)
                            if sum(gold_present_exp) > 0 :
                                gold_present_pred_exp.append(1)
                                gold_present_pred_exp_prop.append(float(sum(gold_present_exp))/len(gold_present_exp))

                            else : 
                                gold_present_pred_exp.append(0)
                                gold_present_pred_exp_prop.append(0.0)

                        gold_present_exp = []

                        current_exp_gold = []

                        if preds[i][j] == 1 or preds[i][j] == 2 : 
                            gold_present_exp.append(1)
                        else : 
                            gold_present_exp.append(0)

                        current_exp_gold.append(i)

                        
                if y[i][j] == 3 and x[i][j] > 0 : 
                    #start another and keep this in the gold_holders
                    if len(current_holder_gold) > 0 : 
                        gold_holders.append(current_holder_gold)
                        if sum(gold_present_holder) > 0 :
                            gold_present_pred_holder.append(1)
                            gold_present_pred_holder_prop.append(float(sum(gold_present_holder))/len(gold_present_holder))
                            
                        else : 
                            gold_present_pred_holder.append(0)
                            gold_present_pred_exp_prop.append(0.0)


                    gold_present_holder = []

                    current_holder_gold = []
                    
                    if preds[i][j] == 3 or preds[i][j] == 4 : 
                        gold_present_holder.append(1)
                    else : 
                        gold_present_holder.append(0)


                    current_holder_gold.append(i)
                if y[i][j] == 4 and x[i][j] > 0 : 
                    if len(current_holder_gold) > 0 and current_holder_gold[-1] == i-1 : 
                        #Should never be the case!! but still need to troubleshoot it!
                        if preds[i][j] == 3 or preds[i][j] == 4 : 
                            gold_present_holder.append(1)
                        else : 
                            gold_present_holder.append(0)

                        current_holder_gold.append(i)
                    else : 
                        if len(current_holder_gold) > 0 : 
                            gold_holders.append(current_holder_gold)
                            if sum(gold_present_holder) > 0 :
                                gold_present_pred_holder.append(1)
                                gold_present_pred_holder_prop.append(float(sum(gold_present_holder))/len(gold_present_holder))
                                
                            else : 
                                gold_present_pred_holder.append(0)
                                gold_present_pred_holder_prop.append(0.0)


                        gold_present_holder = []

                        current_holder_gold = []

                        if preds[i][j] == 3 or preds[i][j] == 4 : 
                            gold_present_holder.append(1)
                        else : 
                            gold_present_holder.append(0)

                        current_holder_gold.append(i)
                
                        
                if y[i][j] == 5 and x[i][j] > 0 : 
                    #start another and keep this in the gold_holders
                    if len(current_target_gold) > 0 : 
                        gold_targets.append(current_target_gold)
                        if sum(gold_present_target) > 0 :
                            gold_present_pred_target.append(1)
                            gold_present_pred_target_prop.append(float(sum(gold_present_target))/len(gold_present_target))
                            
                        else : 
                            gold_present_pred_target.append(0)
                            gold_present_pred_target_prop.append(0.0)


                    gold_present_target = []

                    current_target_gold = []
                    
                    if preds[i][j] == 5 or preds[i][j] == 6 : 
                        gold_present_target.append(1)
                    else : 
                        gold_present_target.append(0)

                    current_target_gold.append(i)

                if y[i][j] == 6 and x[i][j] > 0 : 
                    if len(current_target_gold) > 0 and current_target_gold[-1] == i-1 : 
                        #Should never be the case!! but till need to troubleshoot it!
                        if preds[i][j] == 5 or preds[i][j] == 6 : 
                            gold_present_target.append(1)
                        else : 
                            gold_present_target.append(0)

                        current_target_gold.append(i)
                    else : 

                        if len(current_target_gold) > 0 : 
                            gold_targets.append(current_target_gold)

                            if sum(gold_present_target) > 0 :
                                gold_present_pred_target.append(1)
                                gold_present_pred_target_prop.append(float(sum(gold_present_target))/len(gold_present_target))
                                
                            else : 
                                gold_present_pred_target.append(0)
                                gold_present_pred_target_prop.append(0.0)
                                                    
                        gold_present_target = []

                        current_holder_gold = []

                        if preds[i][j] == 5 or preds[i][j] == 6 : 
                            gold_present_target.append(1)
                        else : 
                            gold_present_target.append(0)
                        
                        current_target_gold.append(i)


                if preds[i][j] == 1 and x[i][j] > 0 : 
                    #start another and keep this in the gold_holders
                    if len(current_exp_pred) > 0 : 
                        pred_exp.append(current_exp_pred)
                        if sum(pred_present_exp) > 0 :
                            pred_present_gold_exp.append(1)
                            pred_present_gold_exp_prop.append(float(sum(pred_present_exp))/len(pred_present_exp))
                            
                        else : 
                            pred_present_gold_exp.append(0)
                            pred_present_gold_exp_prop.append(0.0)


                    pred_present_exp = []
                    current_exp_pred = []
                    
                    if y[i][j] == 1 or y[i][j] == 2 : 
                        pred_present_exp.append(1)
                    else : 
                        pred_present_exp.append(0)

                    current_exp_pred.append(i)
                if preds[i][j] == 2 and x[i][j] > 0 : 
                    if len(current_exp_pred) > 0 and current_exp_pred[-1] == i-1 : 
                        #Should never be the case!! but still need to troubleshoot it!
                        if y[i][j] == 1 or y[i][j] == 2 : 
                            pred_present_exp.append(1)
                        else : 
                            pred_present_exp.append(0)

                        current_exp_pred.append(i)
                    else : 
                        if len(current_exp_pred) > 0 : 
                            pred_exp.append(current_exp_pred)
                            if sum(pred_present_exp) > 0 :
                                pred_present_gold_exp.append(1)
                                pred_present_gold_exp_prop.append(float(sum(pred_present_exp))/len(pred_present_exp))
                            else : 
                                pred_present_gold_exp.append(0)
                                pred_present_gold_exp_prop.append(0.0)

                        pred_present_exp = []

                        current_exp_pred = []

                        if y[i][j] == 1 or y[i][j] == 2 : 
                            pred_present_exp.append(1)
                        else : 
                            pred_present_exp.append(0)

                        current_exp_pred.append(i)

                        
                if preds[i][j] == 3 and x[i][j] > 0 : 
                    #start another and keep this in the gold_holders
                    if len(current_holder_pred) > 0 : 
                        pred_holders.append(current_holder_pred)
                        if sum(pred_present_holder) > 0 :
                            pred_present_gold_holder.append(1)
                            pred_present_gold_holder_prop.append(float(sum(pred_present_holder))/len(pred_present_holder))
                        else : 
                            pred_present_gold_holder.append(0)
                            pred_present_gold_holder_prop.append(0.0)

                    pred_present_holder = []

                    current_holder_pred = []
                    
                    if y[i][j] == 3 or y[i][j] == 4 : 
                        pred_present_holder.append(1)
                    else : 
                        pred_present_holder.append(0)

                    current_holder_pred.append(i)
                if preds[i][j] == 4 and x[i][j] > 0 : 
                    if len(current_holder_pred) > 0 and current_holder_pred[-1] == i-1 : 
                        #Should never be the case!! but still need to troubleshoot it!
                        current_holder_pred.append(i)
                        if y[i][j] == 3 or y[i][j] == 4 : 
                            pred_present_holder.append(1)
                        else : 
                            pred_present_holder.append(0)

                    else : 
                        if len(current_holder_pred) > 0 : 
                            pred_holders.append(current_holder_pred)
                            if sum(pred_present_holder) > 0 :
                                pred_present_gold_holder.append(1)
                                pred_present_gold_holder_prop.append(float(sum(pred_present_holder))/len(pred_present_holder))
                                
                            else : 
                                pred_present_gold_holder.append(0)
                                pred_present_gold_exp_prop.append(0.0)


                        pred_present_holder = []

                        current_holder_pred = []
                        if y[i][j] == 3 or y[i][j] == 4 : 
                            pred_present_holder.append(1)
                        else : 
                            pred_present_holder.append(0)

                        current_holder_pred.append(i)
                
                        
                if preds[i][j] == 5 and x[i][j] > 0 : 
                    #start another and keep this in the gold_holders
                    if len(current_target_pred) > 0 : 
                        pred_targets.append(current_target_pred)
                        if sum(pred_present_target) > 0 :
                            pred_present_gold_target.append(1)
                            pred_present_gold_target_prop.append(float(sum(pred_present_target))/len(pred_present_target))
                            
                        else : 
                            pred_present_gold_target.append(0)
                            pred_present_gold_target_prop.append(0.0)


                    pred_present_target = []

                    current_target_pred = []
                    if y[i][j] == 5 or y[i][j] == 6 : 
                        pred_present_target.append(1)
                    else : 
                        pred_present_target.append(0)

                    current_target_pred.append(i)
                if preds[i][j] == 6 and x[i][j] > 0 : 
                    if len(current_target_pred) > 0 and current_target_pred[-1] == i-1 : 
                        #Should never be the case!! but still need to troubleshoot it!
                        if y[i][j] == 5 or y[i][j] == 6 : 
                            pred_present_target.append(1)
                        else : 
                            pred_present_target.append(0)

                        current_target_pred.append(i)
                    else : 
                        if len(current_target_pred) > 0 : 
                            pred_targets.append(current_target_pred)
                            
                            if sum(pred_present_target) > 0 :
                                pred_present_gold_target.append(1)
                                pred_present_gold_target_prop.append(float(sum(pred_present_target))/len(pred_present_target))
                            else : 
                                pred_present_gold_target.append(0)
                                pred_present_gold_target_prop.append(0.0)

                        pred_present_target = []

                        current_holder_pred = []
                        if y[i][j] == 5 or y[i][j] == 6 : 
                            pred_present_target.append(1)
                        else : 
                            pred_present_target.append(0)

                        current_holder_pred.append(i)
                
            #end here
            if len(current_exp_gold) > 0 : 
                gold_exp.append(current_exp_gold)
                if sum(gold_present_exp) > 0 :
                    gold_present_pred_exp_prop.append(float(sum(gold_present_exp))/len(gold_present_exp))
                    gold_present_pred_exp.append(1)

                else : 
                    gold_present_pred_exp_prop.append(0)
                    gold_present_pred_exp.append(0)

                    
            if len(current_holder_gold) > 0 : 
                gold_holders.append(current_holder_gold)
                if sum(gold_present_holder) > 0 :
                    gold_present_pred_holder_prop.append(float(sum(gold_present_holder))/len(gold_present_holder))
                    gold_present_pred_holder.append(1)
                else : 
                    gold_present_pred_holder_prop.append(0.0)
                    gold_present_pred_holder.append(0)

            if len(current_target_gold) > 0 : 
                gold_targets.append(current_target_gold)
                if sum(gold_present_target) > 0 :
                    gold_present_pred_target_prop.append(float(sum(gold_present_target))/len(gold_present_target))
                    gold_present_pred_target.append(1)
                else : 
                    gold_present_pred_target_prop.append(0.0)
                    gold_present_pred_target.append(0)


            if len(current_exp_pred) > 0 : 
                pred_exp.append(current_exp_pred)
                if sum(pred_present_exp) > 0 :
                    pred_present_gold_exp_prop.append(float(sum(pred_present_exp))/len(pred_present_exp))
                    pred_present_gold_exp.append(1)

                else : 
                    pred_present_gold_exp_prop.append(0.0)
                    pred_present_gold_exp.append(0)

                    
            if len(current_holder_pred) > 0 : 
                pred_holders.append(current_holder_pred)
                if sum(pred_present_holder) > 0 :
                    pred_present_gold_holder.append(1)
                    pred_present_gold_holder_prop.append(float(sum(pred_present_holder))/len(pred_present_holder))
                else : 
                    pred_present_gold_holder.append(0)
                    pred_present_gold_holder_prop.append(0)

            if len(current_target_pred) > 0 : 
                pred_targets.append(current_target_pred)
                if sum(pred_present_target) > 0 :
                    pred_present_gold_target.append(1)
                    pred_present_gold_target_prop.append(float(sum(pred_present_target))/len(pred_present_target))
                else : 
                    pred_present_gold_target.append(0)
                    pred_present_gold_target_prop.append(0.0)

            ####Now here do the analysiss!!!################
            All_pos_exp += len(gold_exp)
            TP_prec_exp += sum(pred_present_gold_exp)
            All_ans_exp += len(pred_exp)
            TP_rec_exp += sum(gold_present_pred_exp)
            
            TP_prec_exp_prop += sum(pred_present_gold_exp_prop)
            TP_rec_exp_prop += sum(gold_present_pred_exp_prop)

            All_pos_holder += len(gold_holders)
            TP_prec_holder += sum(pred_present_gold_holder)
            All_ans_holder += len(pred_holders)
            TP_rec_holder += sum(gold_present_pred_holder)
            
            TP_prec_holder_prop += sum(pred_present_gold_holder_prop)
            TP_rec_holder_prop += sum(gold_present_pred_holder_prop)

            All_pos_target += len(gold_targets)
            TP_prec_target += sum(pred_present_gold_target)
            All_ans_target += len(pred_targets)
            TP_rec_target += sum(gold_present_pred_target)
            
            TP_prec_target_prop += sum(pred_present_gold_target_prop)
            TP_rec_target_prop += sum(gold_present_pred_target_prop)

            
            #print str(TP_prec_exp)+"\t"+str(TP_prec_exp_prop)+"\t"+str(All_ans_exp)
            #print str(TP_rec_exp)+"\t"+ str(TP_rec_exp_prop)+"\t"+str(All_pos_exp)
            
            #if can_break == 1 : 
            #    break
            #if can_break == 1 : 
            #   break


    
    f_measure_all = 0.0
    f_measure_prop_all = 0.0
    f_measure_exact_all = 0.0
    
    final_pred = TP_rec_exp
    final_pred_prop = TP_rec_exp_prop
    #TP_prec_expr_exact += 1
    #TP_rec_expr_exact += 1
    final_pred_exact = TP_rec_exp_exact
    final_correct_expr = All_pos_exp

    final_pred_prec = TP_prec_exp
    final_pred_prec_prop = TP_prec_exp_prop
    final_pred_prec_exact = TP_prec_exp_exact
    final_correct_expr_prec = All_ans_exp

    print "Recall\t"+str(final_pred)+"\t"+str(final_pred_prop)+"\t"+str(final_pred_exact)+"\t"+str(final_correct_expr)
    print "Precision\t"+str(final_pred_prec)+"\t"+str(final_pred_prec_prop)+"\t"+str(final_pred_prec_exact)+"\t"+str(final_correct_expr_prec)
    
    recall = (numpy_floatX(final_pred)/final_correct_expr)
    precision = (numpy_floatX(final_pred_prec)/final_correct_expr_prec)
    f_measure = 2*recall*precision/(recall+precision)

    recall_prop = (numpy_floatX(final_pred_prop)/final_correct_expr)
    precision_prop = (numpy_floatX(final_pred_prec_prop)/final_correct_expr_prec)
    f_measure_prop = 2*recall_prop*precision_prop/(recall_prop+precision_prop)

    recall_exact = (numpy_floatX(final_pred_exact)/final_correct_expr)
    precision_exact = (numpy_floatX(final_pred_prec_exact)/final_correct_expr_prec)
    f_measure_exact = 2*recall_exact*precision_exact/(recall_exact+precision_exact)

    print "Recall\t"+str(recall)+"\t"+str(recall_prop)+"\t"+str(recall_exact)
    print "Precision\t"+str(precision)+"\t"+str(precision_prop)+"\t"+str(precision_exact)

    print "F-score\t Exp \t"+str(f_measure)+"\t"+str(f_measure_prop)+"\t"+str(f_measure_exact)

    f_measure_all += f_measure
    f_measure_prop_all += f_measure_prop
    f_measure_exact_all += f_measure_exact

    final_pred = TP_rec_holder
    final_pred_prop = TP_rec_holder_prop
    final_pred_exact = TP_rec_holder_exact
    final_correct_expr = All_pos_holder

    final_pred_prec = TP_prec_holder
    final_pred_prec_prop = TP_prec_holder_prop
    final_pred_prec_exact = TP_prec_holder_exact
    final_correct_expr_prec = All_ans_holder
    print "-------------"

    print "Recall\t"+str(final_pred)+"\t"+str(final_pred_prop)+"\t"+str(final_pred_exact)+"\t"+str(final_correct_expr)
    print "Precision\t"+str(final_pred_prec)+"\t"+str(final_pred_prec_prop)+"\t"+str(final_pred_prec_exact)+"\t"+str(final_correct_expr_prec)
    
    recall = (numpy_floatX(final_pred)/final_correct_expr)
    precision = (numpy_floatX(final_pred_prec)/final_correct_expr_prec)
    f_measure = 2*recall*precision/(recall+precision)

    recall_prop = (numpy_floatX(final_pred_prop)/final_correct_expr)
    precision_prop = (numpy_floatX(final_pred_prec_prop)/final_correct_expr_prec)
    f_measure_prop = 2*recall_prop*precision_prop/(recall_prop+precision_prop)

    recall_exact = (numpy_floatX(final_pred_exact)/final_correct_expr)
    precision_exact = (numpy_floatX(final_pred_prec_exact)/final_correct_expr_prec)
    f_measure_exact = 2*recall_exact*precision_exact/(recall_exact+precision_exact)

    print "Recall\t"+str(recall)+"\t"+str(recall_prop)+"\t"+str(recall_exact)
    print "Precision\t"+str(precision)+"\t"+str(precision_prop)+"\t"+str(precision_exact)
    
    print "F-score\t Holder \t"+str(f_measure)+"\t"+str(f_measure_prop)+"\t"+str(f_measure_exact)

    f_measure_all += f_measure
    f_measure_prop_all += f_measure_prop
    f_measure_exact_all += f_measure_exact

    final_pred = TP_rec_target
    final_pred_prop = TP_rec_target_prop
    final_pred_exact = TP_rec_target_exact
    final_correct_expr = All_pos_target

    final_pred_prec = TP_prec_target
    final_pred_prec_prop = TP_prec_target_prop
    final_pred_prec_exact = TP_prec_target_exact
    final_correct_expr_prec = All_ans_target

    print "-------------"
    print "Recall\t"+str(final_pred)+"\t"+str(final_pred_prop)+"\t"+str(final_pred_exact)+"\t"+str(final_correct_expr)
    print "Precision\t"+str(final_pred_prec)+"\t"+str(final_pred_prec_prop)+"\t"+str(final_pred_prec_exact)+"\t"+str(final_correct_expr_prec)

    recall = (numpy_floatX(final_pred)/final_correct_expr)
    precision = (numpy_floatX(final_pred_prec)/final_correct_expr_prec)
    f_measure = 2*recall*precision/(recall+precision)

    recall_prop = (numpy_floatX(final_pred_prop)/final_correct_expr)
    precision_prop = (numpy_floatX(final_pred_prec_prop)/final_correct_expr_prec)
    f_measure_prop = 2*recall_prop*precision_prop/(recall_prop+precision_prop)
    
    recall_exact = (numpy_floatX(final_pred_exact)/final_correct_expr)
    precision_exact = (numpy_floatX(final_pred_prec_exact)/final_correct_expr_prec)
    f_measure_exact = 2*recall_exact*precision_exact/(recall_exact+precision_exact)

    print "Recall\t"+str(recall)+"\t"+str(recall_prop)+"\t"+str(recall_exact)
    print "Precision\t"+str(precision)+"\t"+str(precision_prop)+"\t"+str(precision_exact)
    
    print "F-score\t Target\t"+str(f_measure)+"\t"+str(f_measure_prop)+"\t"+str(f_measure_exact)

    f_measure_all += f_measure
    f_measure_prop_all += f_measure_prop
    f_measure_exact_all += f_measure_exact

    print "Relations!!====================================================="
    
    f_measure1 = 0.0
    
    recall = (numpy_floatX(TP_holder_rec)/All_gold_holder)
    precision = (numpy_floatX(TP_holder_prec)/All_pred_holder)
    f_measure_holder = 2*recall*precision/(recall+precision)
    
    print "Recall\t"+str(TP_holder_rec)+"\t"+str(All_gold_holder)+"\t"+str(recall)
    print "Precision\t"+str(TP_holder_prec)+"\t"+str(All_pred_holder)+"\t"+str(precision)

    print "F-score\t Holder\t"+str(f_measure_holder)
    
    recall = (numpy_floatX(TP_target_rec)/All_gold_target)
    precision = (numpy_floatX(TP_target_prec)/All_pred_target)
    f_measure_target = 2*recall*precision/(recall+precision)

    print "Recall\t"+str(TP_target_rec)+"\t"+str(All_gold_target)+"\t"+str(recall)
    print "Precision\t"+str(TP_target_prec)+"\t"+str(All_pred_target)+"\t"+str(precision)
    print "F-score\t Target\t"+str(f_measure_target)
   
    f_measure1 = 2*f_measure_target*f_measure_holder/(f_measure_target+f_measure_holder)
    
    if f_measure1 >= 0 : 
        valid_err = 1. - f_measure1
    else : 
        valid_err = 1.

    predictions.close()
    
    return valid_err



def train_lstm(
    dim_proj = 50, #word emebedding dimension and LSTM number of hidden units
    window = 15, #context window size
    patience=30, #NUmber of epoch to wait before early stop if no progress
    max_epochs = 200, #the maximum number of epochs to run
    dispFreq=1000, #Display to stdout the training progress every N updates
    decay_c = 0., #weight decay for the classifier applied to the U weights
    lrate=0.0005, #learning rate for sgd (not used for adadelta and rmsprop)
    n_words=40000, #Vocbulary size
    optimizer = adadelta, #sgd, adadelta and rmsprop available, sgd very hard to use, not recommended (probably need momentun and decaying learning rate).
    encoder = 'lstm', #TODO: can be removed must be lstm.
    validFreq = -1, #compute the validation error after this number of upadte.
    saveFreq = 6000, #save the parameters after every saveFreq updates
    maxlen=100, #sequence longer than this get ignored (need to change this!!! since we are trying to test it on longer strings)
    batch_size = 1, #the batch size during training
    valid_batch_size=1, #the batch size used for validation/test set
    dataset='mpqa_ht0', #will probably not need them

    #Parameter for extra option
    noise_std=0.,
    use_dropout=True, #if False slightly faster, but worst test error (Check this!)
    reload_model = '', #Path to a saved model we want to start from
    test_size = -1, #If >0, we keep only this number of test examples
    hidden_layers = 3,
    saveto='sll+rll0.npz', #the best model will be saved here
):

    #Model options
    model_options = locals().copy()
    print "model_options", model_options

    load_data, prepare_data, prepare_data_words, find_relations_lrE = get_dataset(dataset)

    print 'Loading data'
    global word_labels
    train, valid, test, word_labels = load_data(n_words=n_words, valid_portion=0.05, maxlen=maxlen)

    
    if test_size > 0 : 
        #the test set is sorted by size, but we want to keep random size example. So we must select a random selection of examples.
        idx = numpy.arange(len(test[0]))
        random.shuffle(idx)
        idx = idx[:test_size]
        test = ([test[0][n] for n in idx], [test[1][n] for n in idx])

    #for p in (train[1][n][p] for p in numpy.arange(len(train[1][n] for n in numpy.arange(len(train[1]))))) : 
     #   print p

    #print (p for p in (n for n in train[1]))
    

    #ydim = numpy.max(train[1][n] for n in numpy.arange(len(train[1]))) + 1 #Not quite sure about this! Our y is an array of arrays, same as x
    ydim = numpy.max(numpy.max([p for p in (n for n in train[1])])) + 1
 
    model_options['ydim'] = ydim #I guess this should be certainly greater than this dimension! Will come back to it after writin some code/..
    print 'Building model'
    # This creates the initial parameters as numpy darrays. 
    # Doct name (string) -> numpy darray
    
    params = init_params(model_options)

    if reload_model : 
        load_params(reload_model, params)

    #This creates Theano shared variable from the parameters. Dict name (string) -> Theano Tensor Shared variable
    # params and tparams have different copy of the weights.
        
    tparams = init_tparams(params)
    #theano.shared(params[kk], name=kk)
    Wemb = theano.shared(Wemb1)

    #use_noise is for dropout
    (use_noise, x, mask, y, all_holder_relations, all_target_relations, holder_indices, target_indices, f_pred, f_pred_hrel, f_pred_trel, final_output, cost, tparams, temp_func) = build_model(tparams, model_options, Wemb)

    if decay_c > 0. : 
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += (tparams['U'] ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    f_cost = theano.function([x, mask, y, all_holder_relations, all_target_relations, holder_indices, target_indices], cost, name='f_cost')

    grads = tensor.grad(cost, wrt=tparams.values())

    f_grad = theano.function([x, mask, y, all_holder_relations, all_target_relations, holder_indices, target_indices], grads, name='f_grad')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, tparams, grads, x, mask, y, all_holder_relations, all_target_relations, holder_indices, target_indices, cost)
    
    print 'Optimization'

    
    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)

    #print "Valid error =========="
    #all_valid_err =  pred_error_relation2(f_pred, f_pred_hrel, f_pred_trel, prepare_data, prepare_data_words, find_relations_lrE, valid, word_labels[1], kf_valid, tparams, window, 0)
    '''print "=============="
    print "Test error"
    all_test_err = pred_error_relation4(f_pred, f_pred_hrel, f_pred_trel, prepare_data, prepare_data_words, find_relations_lrE, test, word_labels[2], kf_test, tparams, window, 0)
    print '-------------------------'
    all_test_err =  pred_error_relation2(f_pred, f_pred_hrel, f_pred_trel, prepare_data, prepare_data_words, find_relations_lrE, test, word_labels[2], kf_test, tparams, window, 0)
    print '------------------------'
    all_test_err =  pred_error_relation3(f_pred, f_pred_hrel, f_pred_trel, prepare_data, prepare_data_words, find_relations_lrE, test, word_labels[2], kf_test, tparams, window, 0)
    print '------------'
    
    exit()'''

    print "%d train examples" %len(train[0])
    print "%d valid examples" % len(valid[0])
    print "%d test examples" % len(test[0])
    history_errs = []
    best_p = None
    bad_count = 0

    if validFreq == -1 : 
        validFreq = len(train[0]) / batch_size
    if saveFreq == -1 : 
        saveFreq = len(train[0]) / batch_size

    uidx = 0 # the number of updates done
    estop = False # early stop
    start_time = time.clock()

    try : 
        for eidx in xrange(max_epochs):
            n_samples = 0

            #Get new suffled index for the training set
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)
            #all_train_err =  pred_error_relation2(f_pred, f_pred_hrel, f_pred_trel, prepare_data, prepare_data_words, find_relations_lrE, train, word_labels[0], kf, tparams, window)


            for _, train_index in kf:
                uidx += 1
                use_noise.set_value(1.)
                
                #Select the random examples from this minibatch
                y = [train[1][t] for t in train_index]
                x = [train[0][t] for t in train_index]

                word_y = [word_labels[0][t] for t in train_index]
                
                #print "1"
                #print x
                #print len(x[0])
                #print y
                #print word_y
                
                all_holder_relations = []
                all_target_relations = []
                
                holder_dim  = 1
                target_dim = 1

                for i in range(len(x)) : 
                    holder_relation, target_relation, holder_dim, target_dim = find_relations_lrE(y[:][i], word_y[:][i])
                    all_holder_relations.append(holder_relation)
                    all_target_relations.append(target_relation)
                    #print all_holder_relations
                    #print all_target_relations
                                    
                all_holder_relations = numpy.asarray(all_holder_relations)
                all_target_relations = numpy.asarray(all_target_relations)
                
                if holder_dim > 1 : 
                    all_holder_relations = numpy.reshape(all_holder_relations, (len(x[0]), len(x)))
                else : 
                    all_holder_relations = numpy.reshape(all_holder_relations, (len(x[0]), len(x), 1))

                if target_dim > 1 :
                    all_target_relations = numpy.reshape(all_target_relations, (len(x[0]), len(x)))
                else : 
                    all_target_relations = numpy.reshape(all_target_relations, (len(x[0]), len(x), 1))


                new_all_holder_relations = numpy.zeros((len(all_holder_relations), len(all_holder_relations[0]), holder_dim)).astype('int32')
                new_all_target_relations = numpy.zeros((len(all_target_relations), len(all_target_relations[0]), target_dim)).astype('int32')
                holder_indices = numpy.zeros((len(all_holder_relations), len(all_holder_relations[0]), 1)).astype('int32')
                target_indices = numpy.zeros((len(all_target_relations), len(all_target_relations[0]), 1)).astype('int32')
                
                
                for p in numpy.arange(len(new_all_holder_relations)) : 
                    for q in numpy.arange(len(new_all_holder_relations[0])) : 
                        holder_indices[p, q, 0] = len(all_holder_relations[p][q])
                        #print len(forward_deps2[p][q])
                        for r in numpy.arange(len(all_holder_relations[p][q])) : 
                            if all_holder_relations[p, q][r] <= window : 
                                new_all_holder_relations[p, q, r] = all_holder_relations[p, q][r]
                            else : 
                                new_all_holder_relations[p, q, r] = 0
                            #print str(new_forward_deps2[p,q,r])+"\t"+str(forward_deps2[p,q,r])
                            

                for p in numpy.arange(len(new_all_target_relations)) : 
                    for q in numpy.arange(len(new_all_target_relations[0])) : 
                        target_indices[p, q, 0] = len(all_target_relations[p][q])
                        #print len(forward_deps2[p][q])
                        for r in numpy.arange(len(all_target_relations[p][q])) : 
                            if all_target_relations[p, q][r] <= window : 
                                new_all_target_relations[p, q, r] = all_target_relations[p, q][r]
                            else : 
                                new_all_target_relations[p, q, r] = 0
                            #print str(new_forward_deps2[p,q,r])+"\t"+str(forward_deps2[p,q,r])
 

                #print new_all_holder_relations
                #rint new_all_target_relations

                #Get the data in numpy.ndarray format
                #This swap the axis!
                #Return something of shape (minibatch maxlen, n samples)
                x, mask, y = prepare_data(x, y)
                n_samples += x.shape[1]

                #print y

                #print "==========================="

                cost = f_grad_shared(x, mask, y, new_all_holder_relations, new_all_target_relations, holder_indices, target_indices)
                
                #print f_grad(x, mask, y)
                f_update(lrate)
                #print temp_func(x, mask, y)
                #print y
                #pred = f_pred(x, mask)
                #print pred
                #print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost

                if numpy.isnan(cost) or numpy.isinf(cost) : 
                    print 'NaN detected'
                    return 1.,1.,1.

                if numpy.mod(uidx, dispFreq) == 0 : 
                    print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost

                    #preds_h, preds_hw = f_pred_hrel(x, mask)
                    #preds_t, preds_tw = f_pred_trel(x, mask)
                    #print preds_h
                    #print preds_hw
                    #print preds_t
                    #print preds_tw
                    #print "============="


                    #print len(y[0])
                    #print len(y)
                    #print x
                    #print y
                    #print word_y

                    #for i in range(len(y)) :
                    #    for j in range(len(y[0])) :
                    #        print str(index_dict[str(x[j][i])])+"\t"+str(word_y[j][i])+"\t"+str(y[j][i])
                    
                    #print "================="
                        
                    #print new_all_holder_relations
                    #print new_all_target_relations
                    #print y
                    #print all_holder_relations
                    #print all_target_relations

                    if cost <= 0 : 
                        '''print temp_func(x, mask, y, new_all_holder_relations, new_all_target_relations, holder_indices, target_indices)
                        print "---------------------------------------------------------"
                        print tparams['A_hrel'].eval()
                        print "|||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
                        print y'''
                        print "NEGATIVE COST"
                        #exit()
                    
                    '''final_norm = 0
                    norm = [(g ** 2).sum() for g in  f_prev_updates(lrate)]
                    for n in norm  :
                    final_norm += n
                    print norm
                    print "------------"
                    ''' 
                    '''for k, p in tparams.iteritems() : 
                        final_norm = 0
                        norm = [(g ** 2).sum() for g in p.get_value()]
                        for n in norm  :
                            final_norm += n
                        print final_norm
                        print "===="
                    '''

                    '''grads_temp =  f_grad(x, mask, y, new_all_holder_relations, new_all_target_relations, holder_indices, target_indices)   
                    grad_norm = tensor.sqrt(sum(map(lambda x: tensor.sqr(x).sum(), grads_temp)))
                    not_finite = tensor.or_(tensor.isnan(grad_norm), tensor.isinf(grad_norm))
                    #grad_norm = tensor.sqrt(grad_norm)
                    rescale = 5.
                    scaling_num = rescale
                    scaling_den = tensor.maximum(rescale, grad_norm)

                    new_grads = [g * (scaling_num/scaling_den) for g in grads_temp]
                    
                    final_norm = 0
                    norm = [(g ** 2).sum().eval() for g in new_grads]
                    for n in norm  :
                        final_norm += n
                    print norm
                    print final_norm'''

                    
                if numpy.mod(uidx, saveFreq) == 0 : 
                    print 'Saving...'

                    if best_p is not None : 
                        params = best_p
                    else : 
                        params = unzip(tparams)
                    numpy.savez(saveto, history_errs, **params)
                    pkl.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
                    print 'Done'


                    '''if numpy.mod(uidx, 60) == 0 : 
                    numpy.savez(saveto, train_err=train_err, valid_err=valid_err, test_err=test_err, history_errs=history_errs, **best_p)'''


                if numpy.mod(uidx, validFreq) == 0 :
                    #print "========================"
                    #print tparams['A'].eval() 
                    #print "---------------------------------------------------------"
                    #print tparams['A_hrel'].eval()
                    #print "---------------------------------------------------------"
                    #print tparams['A_trel'].eval()
                    use_noise.set_value(0.)

                    print "Train error =========="

                    #all_train_err = pred_error_all(f_pred, f_pred_hrel, f_pred_trel, prepare_data, find_relations_lrE, train, word_labels[0], kf, tparams, window)
                    all_train_err = 0
                    if numpy.mod(uidx, 15) == 0 : 
                        all_train_err =  pred_error_relation2(f_pred, f_pred_hrel, f_pred_trel, prepare_data, prepare_data_words, find_relations_lrE, train, word_labels[0], kf, tparams, window, 0)

                    print "Valid error =========="
                    
                    #all_valid_err = pred_error_all(f_pred, f_pred_hrel, f_pred_trel, prepare_data, find_relations_lrE, valid, word_labels[1], kf_valid, tparams, window)
                    all_valid_err =  pred_error_relation2(f_pred, f_pred_hrel, f_pred_trel, prepare_data, prepare_data_words, find_relations_lrE, valid, word_labels[1], kf_valid, tparams, window, 0)
                    

                    print "Test error =========="
                    
                    #all_test_err = pred_error_all(f_pred, f_pred_hrel, f_pred_trel, prepare_data, find_relations_lrE, test, word_labels[2], kf_test, tparams, window)
                    all_test_err =  pred_error_relation2(f_pred, f_pred_hrel, f_pred_trel, prepare_data, prepare_data_words, find_relations_lrE, test, word_labels[2], kf_test, tparams, window, 0)
                    #print "Wrt relation!!"
                    #all_test_err1 =  pred_error_relation2(f_pred, f_pred_hrel, f_pred_trel, prepare_data, prepare_data_words, find_relations_lrE, test, word_labels[2], kf_test, tparams, window, 1)
                    #print "Improved labels!!"
                    #all_test_err2 =  pred_error_relation2(f_pred, f_pred_hrel, f_pred_trel, prepare_data, prepare_data_words, find_relations_lrE, test, word_labels[2], kf_test, tparams, window, 2)

                    if numpy.mod(uidx, 30) == 0 :
                        numpy.savez(saveto, all_train_err=all_train_err, all_valid_err=all_valid_err, all_test_err=all_test_err, history_errs=history_errs, **best_p)
                    
                    if eidx > 65 : 
                        new_p = unzip(tparams)
                        numpy.savez(saveto+str(eidx), all_train_err=all_train_err, all_valid_err=all_valid_err, all_test_err=all_test_err, history_errs=history_errs, **new_p)

                    '''print "Train error"
                    train_err = pred_error(f_pred, prepare_data, train, kf)
                    train_err_H = pred_error_H(f_pred, prepare_data, train, kf)
                    train_err_T = pred_error_T(f_pred, prepare_data, train, kf)
                                        
                    print "Valid error"
                    valid_err = pred_error(f_pred, prepare_data, valid, kf_valid)
                    valid_err_H = pred_error_H(f_pred, prepare_data, valid, kf_valid)
                    valid_err_T = pred_error_T(f_pred, prepare_data, valid, kf_valid)

                    print "Test error"
                    test_err = pred_error(f_pred, prepare_data, test, kf_test)
                    test_err_H = pred_error_H(f_pred, prepare_data, test, kf_test)
                    test_err_T = pred_error_T(f_pred, prepare_data, test, kf_test)'''

                    history_errs.append([all_valid_err, all_test_err])
                    print "Min now\t"+str(numpy.array(history_errs)[:,0].min())+"\tlength\t"+str(len(history_errs))
                    #print "Stores\t"+str(numpy.array(history_errs[:,0]))
                    
                    

                    if (uidx == 0 or all_valid_err <= numpy.array(history_errs)[:,0].min()) : 
                        best_p = unzip(tparams)
                        bad_counter = 0

                    print ('Train ', all_train_err, 'Valid ', all_valid_err, 'Test ', all_test_err)

                    if (len(history_errs) > patience and all_valid_err >= numpy.array(history_errs)[:-patience, 0].min()) :
                        bad_counter += 1
                        if bad_counter > patience : 
                            print 'Early Stop!'
                            estop = True
                            break

                
            print 'Seen %d samples' % n_samples
            #predictions.write("Next epoch\n")
            #predictions.write("\n")
            

            if estop : 
                #break
                print "Should have stopped"
            
    except KeyboardInterrupt : 
        print "Training interrupted"

    end_time = time.clock()
    if best_p is not None : 
        zipp(best_p, tparams)
    else : 
        best_p = unzip(tparams)

    use_noise.set_value(0.)

    kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
    #train_err = pred_error_all(f_pred, f_pred_hrel, f_pred_trel, prepare_data, find_relations_lrE, train, word_labels[0], kf_train_sorted, tparams, window)
    train_err =  pred_error_relation2(f_pred, f_pred_hrel, f_pred_trel, prepare_data, prepare_data_words, find_relations_lrE, train, word_labels[0], kf_train_sorted, tparams, window, 0)


    #valid_err = pred_error_all(f_pred, f_pred_hrel, f_pred_trel, prepare_data, find_relations_lrE, valid, word_labels[1], kf_valid, tparams, window)
    valid_err =  pred_error_relation2(f_pred, f_pred_hrel, f_pred_trel, prepare_data, prepare_data_words, find_relations_lrE, valid, word_labels[1], kf_valid, tparams, window, 0)

    #test_err = pred_error_all(f_pred, f_pred_hrel, f_pred_trel, prepare_data, find_relations_lrE, test, word_labels[2], kf_test, tparams, window)
    test_err =  pred_error_relation2(f_pred, f_pred_hrel, f_pred_trel, prepare_data, prepare_data_words, find_relations_lrE, test, word_labels[2], kf_test, tparams, window, 0)
    #print "Wrt relation"
    #test_err1 =  pred_error_relation2(f_pred, f_pred_hrel, f_pred_trel, prepare_data, prepare_data_words, find_relations_lrE, test, word_labels[2], kf_test, tparams, window, 1)
    #print "Improved labels"
    #test_err2 =  pred_error_relation2(f_pred, f_pred_hrel, f_pred_trel, prepare_data, prepare_data_words, find_relations_lrE, test, word_labels[2], kf_test, tparams, window, 2)


    print 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err

    numpy.savez(saveto, train_err=train_err, valid_err=valid_err, test_err=test_err, history_errs=history_errs, **best_p)
    
    print 'The code ran for %d epochs, with %f sec/epochs' % ((eidx + 1), (end_time - start_time) / (1. * (eidx + 1)))
    print >> sys.stderr, ('Training took %.1fs' %(end_time - start_time))
    return train_err, valid_err, test_err


if __name__ == '__main__' : 
    #See function train for all possible parameter and their definition.
    train_lstm(
        #reload_model="lstm_model_opinion.npz",
        max_epochs=200,
        #test_size=500,
    )
                    
