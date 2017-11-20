import cPickle
import gzip
import os
import operator

import numpy
import theano


def prepare_data(seqs, labels, maxlen=None) :
    ''''Create the martices from the datasets.
    
    This pad each sequence to the same length : the length of the longest sequence or maxlen.
    If maxlen is set, we will cut all sequence to this maximum length.
    This swap the axis!
    '''
    # x : a list of sentences
    lengths = [len(s) for s in seqs]
    
    if maxlen is not None : 
        new_seqs = []
        new_labels = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, labels) :
            if l < maxlen : 
                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        labels = new_labels
        seqs = new_seqs

        if len(lengths) < 1 : 
            return None, None, None

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)

    x = numpy.zeros((maxlen, n_samples)).astype('int32')
    x_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    labels_new = numpy.zeros((maxlen, n_samples)).astype('int32')
    for idx, s in enumerate(seqs) : 
        x[:lengths[idx], idx] = s
        x_mask[:lengths[idx], idx] = 1.

    for idx, y in enumerate(labels) :
        labels_new[:lengths[idx], idx] = y

    #print labels
    #print labels_new

    return x, x_mask, labels_new 
    #return x, x_mask, labels 
        
def prepare_data_words(seqs, labels, maxlen=None) :
    ''''Create the martices from the datasets.
    
    This pad each sequence to the same length : the length of the longest sequence or maxlen.
    If maxlen is set, we will cut all sequence to this maximum length.
    This swap the axis!
    '''
    # x : a list of sentences
    lengths = [len(s) for s in seqs]
    
    if maxlen is not None : 
        new_seqs = []
        new_labels = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, labels) :
            if l < maxlen : 
                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        labels = new_labels
        seqs = new_seqs

        if len(lengths) < 1 : 
            return None, None, None

    n_samples = len(seqs)
    #print n_samples
    maxlen = numpy.max(lengths)
    #print maxlen

    x = numpy.zeros((maxlen, n_samples)).astype('int32')
    x_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    labels_new = numpy.empty((maxlen, n_samples), dtype=numpy.dtype('a50'))
    #print labels_new
    labels_new[:] = "O"
    #print labels_new
    #print "++++++++++++++++++++++++++++++++++++++++++"
    #print labels
    
    
    #print len(labels_new)
    
    for idx, s in enumerate(seqs) : 
        x[:lengths[idx], idx] = s
        x_mask[:lengths[idx], idx] = 1.
    for idx, y in enumerate(labels) :
        #print str(lengths[idx])+"\t"+str(idx)+"\ty\t"+str(y)
        labels_new[:lengths[idx], idx] = y
        '''for k in numpy.arange(lengths[idx]) : 
            labels_new[k, idx] = y[k][:]'''
        #print "New\t"+str(labels_new[:lengths[idx], idx])
    
    #print labels
    #print "----------------------------------------------"
    #print labels_new
    #exit()
    return x, x_mask, labels_new 
    #return x, x_mask, labels 


def find_relations(x, gold) : 
    
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
        #print i
        if gold[i].startswith("B_AGENT") and len(gold[i].split("_")) > 2 : 
            #print gold[i].split("_")
            rel = []
            for k in numpy.arange(1, len(gold[i].split("_REL"))) : 
                rel.append(int(gold[i].split("_REL")[k]))
                gold_holder_start[rel[-1]] = i
                    
            if i+1 >= len(x) : 
                for r in rel : 
                    gold_holder_end[r] = i
                
                    
            while i+1 < len(x) and gold[i+1].startswith("AGENT") :
                i+=1
                for r in rel : 
                    gold_holder_end[r] = i
                            
            for r in rel  :
                if r not in gold_holder_end.keys() : 
                    gold_holder_end[r] = gold_holder_start[r]
                        
        if gold[i].startswith("B_TARGET") and len(gold[i].split("_")) > 2 : 
            #print gold[i][j].split("_")
            #rel = int(gold[i][j].split("_")[-1][3:])
            #gold_target_start[rel] = i
            rel = []
            for k in numpy.arange(1, len(gold[i].split("_REL"))) : 
                rel.append(int(gold[i].split("_REL")[k]))
                gold_target_start[rel[-1]] = i

            if i+1 >= len(x) :
                for  r in rel : 
                    gold_target_end[r] = i
                break

            while i+1 < len(x) and gold[i+1].startswith("TARGET") :
                i+=1
                for  r in rel : 
                    gold_target_end[r] = i

            for r in rel : 
                if r not in gold_target_end.keys() : 
                    gold_target_end[r] = gold_target_start[r]

        if gold[i].startswith("B_DSE") and len(gold[i].split("_")) > 2: 
            #print gold[i][j].split("_")
            #rel = int(gold[i][j].split("_")[-1][3:])
            #gold_expr_start[rel] = i
            
            rel = []
            for k in numpy.arange(1, len(gold[i].split("_REL"))) : 
                rel.append(int(gold[i].split("_REL")[k]))
                gold_expr_start[rel[-1]] = i

                    
            if i+1 >= len(x) : 
                for r in rel : 
                    gold_expr_end[r] = i
                break

            while i+1 < len(x) and gold[i+1].startswith("DSE") :
                i+=1
                for r in rel : 
                    gold_expr_end[r] = i

            for r in rel : 
                if r not in gold_expr_end.keys() : 
                    gold_expr_end[r] = gold_expr_start[r]

        #######Now all the respective annotations are done!!!!
        
        #print i

        #print gold_holder_start
        #print gold_holder_end
        #print gold_target_start
        #print gold_target_end

    #print gold_holder_start
    #print gold_holder_end
    #print gold_target_start
    #print gold_target_end
    #print gold_expr_start
    #print gold_expr_end


    relation_holder = [[0] for i in range(len(x))]
    relation_target = [[0] for i in range(len(x))]

    holder_dim = 1
    target_dim = 1

    #print len(x)
    
    for key in gold_expr_start.keys() : 
        #print key
        start_e = gold_expr_start[key]
        #print gold_expr_start.keys()
        #print gold_expr_end.keys()
        end_e = gold_expr_end[key]
        if key in gold_holder_start.keys() : 
            start_h = gold_holder_start[key]
            end_h = gold_holder_end[key]
            liste = numpy.arange(start_e, end_e+1)
            listh = numpy.arange(start_h, end_h+1)
                        
            for e_ind in liste : 
                for h_ind in listh : 
                    if e_ind < h_ind : 
                        if 0 in relation_holder[h_ind] : 
                            relation_holder[h_ind].remove(0)
                        relation_holder[h_ind].append(h_ind-e_ind)
                        if holder_dim < len(relation_holder[h_ind]) : 
                            holder_dim = len(relation_holder[h_ind])
                    if e_ind > h_ind : 
                        if 0 in relation_holder[e_ind] : 
                            relation_holder[e_ind].remove(0)
                        relation_holder[e_ind].append(e_ind-h_ind)
                        if holder_dim < len(relation_holder[e_ind]) : 
                            holder_dim = len(relation_holder[e_ind])
            
        if key in gold_target_start.keys() : 
            start_t = gold_target_start[key]
            end_t = gold_target_end[key]
            liste = numpy.arange(start_e, end_e+1)
            listt = numpy.arange(start_t, end_t+1)
            
            for e_ind in liste : 
                for t_ind in listt : 
                    if e_ind < t_ind : 
                        if 0 in relation_target[t_ind] : 
                            relation_target[t_ind].remove(0)
                        relation_target[t_ind].append(t_ind-e_ind)
                        if target_dim < len(relation_target[t_ind]) : 
                            target_dim = len(relation_target[t_ind])

                    if e_ind > t_ind : 
                        if 0 in relation_target[e_ind] : 
                            relation_target[e_ind].remove(0)
                        relation_target[e_ind].append(e_ind-t_ind)
                        if target_dim < len(relation_target[e_ind]) : 
                            target_dim = len(relation_target[e_ind])


        #print "=========================================="
    #print gold_all_relations

    #print relation_holder
    #print relation_target
        
    return relation_holder, relation_target, holder_dim, target_dim



def find_relations_lr(x, gold) : 
    
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
        #print i
        if gold[i].startswith("B_AGENT") and len(gold[i].split("_")) > 2 : 
            #print gold[i].split("_")
            rel = []
            for k in numpy.arange(1, len(gold[i].split("_REL"))) : 
                rel.append(int(gold[i].split("_REL")[k]))
                gold_holder_start[rel[-1]] = i
                    
            if i+1 >= len(x) : 
                for r in rel : 
                    gold_holder_end[r] = i
                
                    
            while i+1 < len(x) and gold[i+1].startswith("AGENT") :
                i+=1
                for r in rel : 
                    gold_holder_end[r] = i
                            
            for r in rel  :
                if r not in gold_holder_end.keys() : 
                    gold_holder_end[r] = gold_holder_start[r]
                        
        if gold[i].startswith("B_TARGET") and len(gold[i].split("_")) > 2 : 
            #print gold[i][j].split("_")
            #rel = int(gold[i][j].split("_")[-1][3:])
            #gold_target_start[rel] = i
            rel = []
            for k in numpy.arange(1, len(gold[i].split("_REL"))) : 
                rel.append(int(gold[i].split("_REL")[k]))
                gold_target_start[rel[-1]] = i

            if i+1 >= len(x) :
                for  r in rel : 
                    gold_target_end[r] = i
                break

            while i+1 < len(x) and gold[i+1].startswith("TARGET") :
                i+=1
                for  r in rel : 
                    gold_target_end[r] = i

            for r in rel : 
                if r not in gold_target_end.keys() : 
                    gold_target_end[r] = gold_target_start[r]

        if gold[i].startswith("B_DSE") and len(gold[i].split("_")) > 2: 
            #print gold[i][j].split("_")
            #rel = int(gold[i][j].split("_")[-1][3:])
            #gold_expr_start[rel] = i
            
            rel = []
            for k in numpy.arange(1, len(gold[i].split("_REL"))) : 
                rel.append(int(gold[i].split("_REL")[k]))
                gold_expr_start[rel[-1]] = i

                    
            if i+1 >= len(x) : 
                for r in rel : 
                    gold_expr_end[r] = i
                break

            while i+1 < len(x) and gold[i+1].startswith("DSE") :
                i+=1
                for r in rel : 
                    gold_expr_end[r] = i

            for r in rel : 
                if r not in gold_expr_end.keys() : 
                    gold_expr_end[r] = gold_expr_start[r]

        #######Now all the respective annotations are done!!!!
        
        #print i

        #print gold_holder_start
        #print gold_holder_end
        #print gold_target_start
        #print gold_target_end

    #print gold_holder_start
    #print gold_holder_end
    #print gold_target_start
    #print gold_target_end
    #print gold_expr_start
    #print gold_expr_end


    relation_left = [[0] for i in range(len(x))]
    relation_right = [[0] for i in range(len(x))]

    left_dim = 1
    right_dim = 1

    #print len(x)
    
    for key in gold_expr_start.keys() : 
        #print key
        start_e = gold_expr_start[key]
        #print gold_expr_start.keys()
        #print gold_expr_end.keys()
        end_e = gold_expr_end[key]
        if key in gold_holder_start.keys() : 
            start_h = gold_holder_start[key]
            end_h = gold_holder_end[key]
            liste = numpy.arange(start_e, end_e+1)
            listh = numpy.arange(start_h, end_h+1)
                        
            for e_ind in liste : 
                for h_ind in listh : 
                    if e_ind < h_ind : 
                        if 0 in relation_left[h_ind] : 
                            relation_left[h_ind].remove(0)
                        relation_left[h_ind].append(h_ind-e_ind)
                        if left_dim < len(relation_left[h_ind]) : 
                            left_dim = len(relation_left[h_ind])
                    if e_ind > h_ind : 
                        if 0 in relation_right[h_ind] : 
                            relation_right[h_ind].remove(0)
                        relation_right[h_ind].append(e_ind-h_ind)
                        if right_dim < len(relation_right[h_ind]) : 
                            right_dim = len(relation_right[h_ind])
            
        if key in gold_target_start.keys() : 
            start_t = gold_target_start[key]
            end_t = gold_target_end[key]
            liste = numpy.arange(start_e, end_e+1)
            listt = numpy.arange(start_t, end_t+1)
            
            for e_ind in liste : 
                for t_ind in listt : 
                    if e_ind < t_ind : 
                        if 0 in relation_left[t_ind] : 
                            relation_left[t_ind].remove(0)
                        relation_left[t_ind].append(t_ind-e_ind)
                        if left_dim < len(relation_left[t_ind]) : 
                            left_dim = len(relation_left[t_ind])

                    if e_ind > t_ind : 
                        if 0 in relation_right[t_ind] : 
                            relation_right[t_ind].remove(0)
                        relation_right[t_ind].append(e_ind-t_ind)
                        if right_dim < len(relation_right[t_ind]) : 
                            right_dim = len(relation_right[t_ind])


        #print "=========================================="
    #print gold_all_relations

    #print relation_holder
    #print relation_target
        
    return relation_left, relation_right, left_dim, right_dim



def find_relations_lrE(x, gold) : 
    
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
        #print i
        if gold[i].startswith("B_AGENT") and len(gold[i].split("_")) > 2 : 
            #print gold[i].split("_")
            rel = []
            for k in numpy.arange(1, len(gold[i].split("_REL"))) : 
                rel.append(int(gold[i].split("_REL")[k]))
                gold_holder_start[rel[-1]] = i
                    
            if i+1 >= len(x) : 
                for r in rel : 
                    gold_holder_end[r] = i
                
                    
            while i+1 < len(x) and gold[i+1].startswith("AGENT") :
                i+=1
                for r in rel : 
                    gold_holder_end[r] = i
                            
            for r in rel  :
                if r not in gold_holder_end.keys() : 
                    gold_holder_end[r] = gold_holder_start[r]
                        
        if gold[i].startswith("B_TARGET") and len(gold[i].split("_")) > 2 : 
            #print gold[i][j].split("_")
            #rel = int(gold[i][j].split("_")[-1][3:])
            #gold_target_start[rel] = i
            rel = []
            for k in numpy.arange(1, len(gold[i].split("_REL"))) : 
                rel.append(int(gold[i].split("_REL")[k]))
                gold_target_start[rel[-1]] = i

            if i+1 >= len(x) :
                for  r in rel : 
                    gold_target_end[r] = i
                break

            while i+1 < len(x) and gold[i+1].startswith("TARGET") :
                i+=1
                for  r in rel : 
                    gold_target_end[r] = i

            for r in rel : 
                if r not in gold_target_end.keys() : 
                    gold_target_end[r] = gold_target_start[r]

        if gold[i].startswith("B_DSE") and len(gold[i].split("_")) > 2: 
            #print gold[i][j].split("_")
            #rel = int(gold[i][j].split("_")[-1][3:])
            #gold_expr_start[rel] = i
            
            rel = []
            for k in numpy.arange(1, len(gold[i].split("_REL"))) : 
                rel.append(int(gold[i].split("_REL")[k]))
                gold_expr_start[rel[-1]] = i

                    
            if i+1 >= len(x) : 
                for r in rel : 
                    gold_expr_end[r] = i
                break

            while i+1 < len(x) and gold[i+1].startswith("DSE") :
                i+=1
                for r in rel : 
                    gold_expr_end[r] = i

            for r in rel : 
                if r not in gold_expr_end.keys() : 
                    gold_expr_end[r] = gold_expr_start[r]

        #######Now all the respective annotations are done!!!!
        
        #print i

        #print gold_holder_start
        #print gold_holder_end
        #print gold_target_start
        #print gold_target_end

    #print gold_holder_start
    #print gold_holder_end
    #print gold_target_start
    #print gold_target_end
    #print gold_expr_start
    #print gold_expr_end


    relation_left = [[0] for i in range(len(x))]
    relation_right = [[0] for i in range(len(x))]

    left_dim = 1
    right_dim = 1

    #print len(x)
    
    for key in gold_expr_start.keys() : 
        #print key
        start_e = gold_expr_start[key]
        #print gold_expr_start.keys()
        #print gold_expr_end.keys()
        end_e = gold_expr_end[key]
        if key in gold_holder_start.keys() : 
            start_h = gold_holder_start[key]
            end_h = gold_holder_end[key]
            liste = numpy.arange(start_e, end_e+1)
            listh = numpy.arange(start_h, end_h+1)
                        
            for e_ind in liste : 
                for h_ind in listh : 
                    if e_ind < h_ind : 
                        if 0 in relation_right[e_ind] : 
                            relation_right[e_ind].remove(0)
                        relation_right[e_ind].append(h_ind-e_ind)
                        if right_dim < len(relation_right[e_ind]) : 
                            right_dim = len(relation_right[e_ind])
                    if e_ind > h_ind : 
                        if 0 in relation_left[e_ind] : 
                            relation_left[e_ind].remove(0)
                        relation_left[e_ind].append(e_ind-h_ind)
                        if left_dim < len(relation_left[e_ind]) : 
                            left_dim = len(relation_left[e_ind])
            
        if key in gold_target_start.keys() : 
            start_t = gold_target_start[key]
            end_t = gold_target_end[key]
            liste = numpy.arange(start_e, end_e+1)
            listt = numpy.arange(start_t, end_t+1)
            
            for e_ind in liste : 
                for t_ind in listt : 
                    if e_ind < t_ind : 
                        if 0 in relation_left[t_ind] : 
                            relation_left[t_ind].remove(0)
                        relation_left[t_ind].append(t_ind-e_ind)
                        if left_dim < len(relation_left[t_ind]) : 
                            left_dim = len(relation_left[t_ind])

                    if e_ind > t_ind : 
                        if 0 in relation_right[t_ind] : 
                            relation_right[t_ind].remove(0)
                        relation_right[t_ind].append(e_ind-t_ind)
                        if right_dim < len(relation_right[t_ind]) : 
                            right_dim = len(relation_right[t_ind])


        #print "=========================================="
    #print gold_all_relations

    #print relation_holder
    #print relation_target
        
    return relation_left, relation_right, left_dim, right_dim


def find_relations_lrET(x, gold) : 
    
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
        #print i
        if gold[i].startswith("B_AGENT") and len(gold[i].split("_")) > 2 : 
            #print gold[i].split("_")
            rel = []
            for k in numpy.arange(1, len(gold[i].split("_REL"))) : 
                rel.append(int(gold[i].split("_REL")[k]))
                gold_holder_start[rel[-1]] = i
                    
            if i+1 >= len(x) : 
                for r in rel : 
                    gold_holder_end[r] = i
                
                    
            while i+1 < len(x) and gold[i+1].startswith("AGENT") :
                i+=1
                for r in rel : 
                    gold_holder_end[r] = i
                            
            for r in rel  :
                if r not in gold_holder_end.keys() : 
                    gold_holder_end[r] = gold_holder_start[r]
                        
        if gold[i].startswith("B_TARGET") and len(gold[i].split("_")) > 2 : 
            #print gold[i][j].split("_")
            #rel = int(gold[i][j].split("_")[-1][3:])
            #gold_target_start[rel] = i
            rel = []
            for k in numpy.arange(1, len(gold[i].split("_REL"))) : 
                rel.append(int(gold[i].split("_REL")[k]))
                gold_target_start[rel[-1]] = i

            if i+1 >= len(x) :
                for  r in rel : 
                    gold_target_end[r] = i
                break

            while i+1 < len(x) and gold[i+1].startswith("TARGET") :
                i+=1
                for  r in rel : 
                    gold_target_end[r] = i

            for r in rel : 
                if r not in gold_target_end.keys() : 
                    gold_target_end[r] = gold_target_start[r]

        if gold[i].startswith("B_DSE") and len(gold[i].split("_")) > 2: 
            #print gold[i][j].split("_")
            #rel = int(gold[i][j].split("_")[-1][3:])
            #gold_expr_start[rel] = i
            
            rel = []
            for k in numpy.arange(1, len(gold[i].split("_REL"))) : 
                rel.append(int(gold[i].split("_REL")[k]))
                gold_expr_start[rel[-1]] = i

                    
            if i+1 >= len(x) : 
                for r in rel : 
                    gold_expr_end[r] = i
                break

            while i+1 < len(x) and gold[i+1].startswith("DSE") :
                i+=1
                for r in rel : 
                    gold_expr_end[r] = i

            for r in rel : 
                if r not in gold_expr_end.keys() : 
                    gold_expr_end[r] = gold_expr_start[r]

        #######Now all the respective annotations are done!!!!
        
        #print i

        #print gold_holder_start
        #print gold_holder_end
        #print gold_target_start
        #print gold_target_end

    #print gold_holder_start
    #print gold_holder_end
    #print gold_target_start
    #print gold_target_end
    #print gold_expr_start
    #print gold_expr_end


    relation_left_holder = [[0] for i in range(len(x))]
    relation_right_holder = [[0] for i in range(len(x))]
    
    relation_left_target = [[0] for i in range(len(x))]
    relation_right_target = [[0] for i in range(len(x))]



    left_dim_holder = 1
    right_dim_holder = 1

    left_dim_target = 1
    right_dim_target = 1

    #print len(x)
    
    for key in gold_expr_start.keys() : 
        #print key
        start_e = gold_expr_start[key]
        #print gold_expr_start.keys()
        #print gold_expr_end.keys()
        end_e = gold_expr_end[key]
        if key in gold_holder_start.keys() : 
            start_h = gold_holder_start[key]
            end_h = gold_holder_end[key]
            liste = numpy.arange(start_e, end_e+1)
            listh = numpy.arange(start_h, end_h+1)
                        
            for e_ind in liste : 
                for h_ind in listh : 
                    if e_ind < h_ind : 
                        if 0 in relation_right_holder[e_ind] : 
                            relation_right_holder[e_ind].remove(0)
                        relation_right_holder[e_ind].append(h_ind-e_ind)
                        if right_dim_holder < len(relation_right_holder[e_ind]) : 
                            right_dim_holder = len(relation_right_holder[e_ind])
                    if e_ind > h_ind : 
                        if 0 in relation_left_holder[e_ind] : 
                            relation_left_holder[e_ind].remove(0)
                        relation_left_holder[e_ind].append(e_ind-h_ind)
                        if left_dim_holder < len(relation_left_holder[e_ind]) : 
                            left_dim_holder = len(relation_left_holder[e_ind])
            
        if key in gold_target_start.keys() : 
            start_t = gold_target_start[key]
            end_t = gold_target_end[key]
            liste = numpy.arange(start_e, end_e+1)
            listt = numpy.arange(start_t, end_t+1)
            
            for e_ind in liste : 
                for t_ind in listt : 
                    if e_ind < t_ind : 
                        if 0 in relation_right_target[e_ind] : 
                            relation_right_target[e_ind].remove(0)
                        relation_right_target[e_ind].append(t_ind-e_ind)
                        if right_dim_target < len(relation_right_target[e_ind]) : 
                            right_dim_target = len(relation_right_target[e_ind])

                    if e_ind > t_ind : 
                        if 0 in relation_left_target[e_ind] : 
                            relation_left_target[e_ind].remove(0)
                        relation_left_target[e_ind].append(e_ind-t_ind)
                        if left_dim_target < len(relation_left_target[e_ind]) : 
                            left_dim_target = len(relation_left_target[e_ind])


        #print "=========================================="
    #print gold_all_relations

    #print relation_holder
    #print relation_target
        
    return relation_left_holder, relation_right_holder, left_dim_holder, right_dim_holder, relation_left_target, relation_right_target, left_dim_target, right_dim_target



def load_data(path="../data_MPQA/", n_words=40000, valid_portion=0.1, maxlen = None, sort_by_len=True,  cross_val = 0) :
    '''Loads the daatset
    :type path : String
    :param path : the path to the dataset (here MPQA)
    :type n_words : int
    :param n_words : the number of word to keep in the vocabulary. All the extra words are set to unknown.
    :type valid portion : float
    :param valid_portion : the proportion of the full train set used for the validation set (Not to be used!!)
    :type maxlen : None or positive int
    :param maxlen : the max sequence length we use in the train/valid set
    :type sort_by_len : bool
    :name sort_by_len : Sort by the sequence length for the train,valid and test set. This allows faster execution as it cause less padding per minibatch. Another mechanism must be used to shuffle the train set at each epoch.
    :type cross_val : int
    :param cross_val : the train and test partition to be extracted for this cross validation number
'''

    #the data required by the model is in a different format : so we span over all the words in the dse.txt and give indices from the dictionary to the words

    words_dict = dict()
    
    all_words = open(path+"all_ILP.txt")
    for word_t in all_words : 
        #print word_t.split("\t")[0]
        if len(word_t.split("\t")) > 1  :
            word = word_t.split("\t")[0]
            #print word
            if word in words_dict : 
                #increment the counter
                words_dict[word] = words_dict.get(word)+1
            else : 
                words_dict[word] = 1

    print len(words_dict.keys())
    
    sorted_words_dict = sorted(words_dict.items(), key=operator.itemgetter(1))

    index_words_dict = dict()
    words_index_dict = dict()
    
    count = 0

    #writing the dictionary to a file dict.txt
    f_dict = open(path+"dict.txt", 'w+')

    for key, value in sorted_words_dict : 
        count +=1
        #print str(key)+"\t"+str(value)
        index_words_dict[count] = key
        words_index_dict[key] = count
        f_dict.write(str(count)+"\t"+str(key)+"\n")

    f_dict.close()

    sentence_f = open(path+"sentenceid.txt")
    dse_f = open(path+"all_ILP.txt")

    count_filename = 0
    current_filename = ""
    number_lines = 0
    filename_id = dict()
    number_Lines = dict()

    for sentence_id in sentence_f :
        #print sentence_id
        if sentence_id.split(" ")[0] == count_filename : 
            number_lines = number_lines+1
            current_filename = sentence_id.split(" ")[2]
        else : 
            #store the entry in a dict
            filename_id[current_filename] = count_filename
            #print str(count_filename)+"\t"+filename_id.get(count_filename)
            number_Lines[count_filename] = number_lines 
            count_filename = sentence_id.split(" ")[0]
            current_filename = sentence_id.split(" ")[2]
            number_lines = 1

    #when reading the file ends, we need to store for the last filename
    filename_id[current_filename] = count_filename
    number_Lines[count_filename] = number_lines 

    '''for key in filename_id.iterkeys() : 
        print key
        print filename_id.get(key)
        #print number_Lines.get(str(0))'''

    start_count = 0 #start with this filename and the corresponding count of lines.
    num_lines = number_Lines.get(str(start_count))

    all_sentences = dict()
    all_sentences_tags = dict()
    all_sentences_word_tags = dict()
    
    #storing the words from a sentence and the tags separately. All the words in the sentence are separated by spaces and so is the case with the tags

    current_sentence = []
    current_sentence_tag = []
    current_sentence_word_tag = []
    current_sentence_deps = []
    current_count = 0
    for dse_lines in dse_f : 
        if len(dse_lines.split("\t")) == 1 : #endline character, store the sentences in the list corresponding to the file numbers
            current_count = current_count+1
#            print str(current_count) +"\t"+str(num_lines)+"\t"+str(start_count)
            if current_count <= num_lines : 
                #store in the list corresponding to the current filenumber
                if start_count in all_sentences :
                    all_sentences.get(start_count)[current_count] = current_sentence
                    all_sentences_tags.get(start_count)[current_count] = current_sentence_tag
                    all_sentences_word_tags.get(start_count)[current_count] = current_sentence_word_tag

                else : 
                    #start a new entry
                    entry = dict()
                    entry_tag = dict()
                    entry_word_tag = dict()
                    entry[current_count] = current_sentence
                    entry_tag[current_count] = current_sentence_tag
                    entry_word_tag[current_count] = current_sentence_word_tag
                    all_sentences[start_count] = entry
                    all_sentences_tags[start_count] = entry_tag 
                    all_sentences_word_tags[start_count] = entry_word_tag 
                    
            else  :
                start_count += 1
               # print start_count
                num_lines = number_Lines.get(str(start_count))
                current_count = 1
                
                #start a new entry
                entry = dict()
                entry_tag = dict()
                entry_word_tag = dict()
                entry[current_count] = current_sentence
                entry_tag[current_count] = current_sentence_tag 
                entry_word_tag[current_count] = current_sentence_word_tag 
                all_sentences[start_count] = entry
                all_sentences_tags[start_count] = entry_tag
                all_sentences_word_tags[start_count] = entry_word_tag
            current_sentence = []
            current_sentence_tag = []
            current_sentence_word_tag = []

        else : 
            #current_sentence += dse_lines.split("\t")[0] + " "
            #current_sentence_tag += dse_lines.split("\t")[2]+ " "
            current_sentence.append(words_index_dict.get(dse_lines.split("\t")[0]))
            if words_index_dict.get(dse_lines.split("\t")[0]) < 1 : 
                print "No index found"
            #print words_index_dict.get(dse_lines.split("\t")[0])
            temp_tag = dse_lines.split("\t")[2].strip()
            #print dse_lines.split("\t")
            current_sentence_word_tag.append(temp_tag)
            if temp_tag.startswith("B_DSE") : 
                current_sentence_tag.append(1)
            elif temp_tag.startswith("DSE") : 
                current_sentence_tag.append(2)
            elif temp_tag == "O" : 
                current_sentence_tag.append(0)
            elif temp_tag.startswith("B_AGENT"): 
                current_sentence_tag.append(3)
            elif temp_tag.startswith("AGENT"): 
                current_sentence_tag.append(4)
            elif temp_tag.startswith("B_TARGET") : 
                current_sentence_tag.append(5)
            elif temp_tag.startswith("TARGET") : 
                current_sentence_tag.append(6)
            else : 
                print "Annotation not found\t"+str(temp_tag)


    '''for key in all_sentences.iterkeys() : 
        print str(key)+"\t"+str(number_Lines.get(str(key)))+"\t"+str(filename_id.get(str(key)))
        for s_key in all_sentences.get(key).iterkeys() : 
            print str(s_key)+"\t"+str(all_sentences.get(key).get(s_key))'''


    #Return the training test and valiation set depending on the cross-validation number 
    
    f_train = open(path+"datasplit/filelist_train"+str(cross_val))
    train_ids = []
    test_ids = []
    val_ids = []

    for ids in f_train : 
        train_ids.append(ids)

    f_test = open(path+"datasplit/filelist_test"+str(cross_val))
    
    for ids in f_test : 
        test_ids.append(ids)

    f_allsent = open(path+"datasplit/doclist.mpqaOriginalSubset")
    
    for ids in f_allsent : 
        if ids not in train_ids and ids not in test_ids : 
            val_ids.append(ids)

    #print len(train_ids)
    #print len(test_ids)
    #print len(val_ids)

    #creating training data, test data and validation data
    
    #Training Data
    new_train_set_x = []
    new_train_set_y = []
    new_train_set_word_y = []
    for ids in train_ids : 
        #print (filename_id.get(ids))
        #print all_sentences.keys()
        #print all_sentences.get(int(filename_id.get(ids)))
        for key in all_sentences.get(int(filename_id.get(ids))).iterkeys() : 
            #all_sentences.get(int(filename_id.get(str(ids)))).get(key)
            new_train_set_x.append(all_sentences.get(int(filename_id.get(str(ids)))).get(key))
            new_train_set_y.append(all_sentences_tags.get(int(filename_id.get(str(ids)))).get(key))
            new_train_set_word_y.append(all_sentences_word_tags.get(int(filename_id.get(str(ids)))).get(key))

    train_set = (new_train_set_x, new_train_set_y, new_train_set_word_y)
    del new_train_set_x, new_train_set_y, new_train_set_word_y

    
    #Test Data
    new_test_set_x = []
    new_test_set_y = []
    new_test_set_word_y = []
    for ids in test_ids : 
        #print (filename_id.get(ids))
        #print all_sentences.keys()
        #print all_sentences.get(int(filename_id.get(ids)))
        for key in all_sentences.get(int(filename_id.get(ids))).iterkeys() : 
            #all_sentences.get(int(filename_id.get(str(ids)))).get(key)
            new_test_set_x.append(all_sentences.get(int(filename_id.get(str(ids)))).get(key))
            new_test_set_y.append(all_sentences_tags.get(int(filename_id.get(str(ids)))).get(key))
            new_test_set_word_y.append(all_sentences_word_tags.get(int(filename_id.get(str(ids)))).get(key))

    test_set = (new_test_set_x, new_test_set_y, new_test_set_word_y)
    del new_test_set_x, new_test_set_y, new_test_set_word_y
            
    #Validation Data
    new_valid_set_x = []
    new_valid_set_y = []
    new_valid_set_word_y = []
    for ids in val_ids : 
        #print (filename_id.get(ids))
        #print all_sentences.keys()
        #print all_sentences.get(int(filename_id.get(ids)))
        for key in all_sentences.get(int(filename_id.get(ids))).iterkeys() : 
            #all_sentences.get(int(filename_id.get(str(ids)))).get(key)
            new_valid_set_x.append(all_sentences.get(int(filename_id.get(str(ids)))).get(key))
            new_valid_set_y.append(all_sentences_tags.get(int(filename_id.get(str(ids)))).get(key))
            new_valid_set_word_y.append(all_sentences_word_tags.get(int(filename_id.get(str(ids)))).get(key))

    valid_set = (new_valid_set_x, new_valid_set_y, new_valid_set_word_y)
    del new_valid_set_x, new_valid_set_y, new_valid_set_word_y 

    def remove_unk(x) :
        #print [[1 if w >= n_words else w for w in sen] for sen in x]
        return [[1 if w >= n_words else w for w in sen] for sen in x]
        

    test_set_x, test_set_y, test_set_word_y = test_set
    valid_set_x, valid_set_y, valid_set_word_y = valid_set
    train_set_x, train_set_y, train_set_word_y = train_set

    #not calling remove_unk for now! probably need to change the format of the data
    train_set_x = remove_unk(train_set_x)
    valid_set_x = remove_unk(valid_set_x)
    test_set_x = remove_unk(test_set_x)

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len : 
        sorted_index = len_argsort(test_set_x)
        test_set_x = [test_set_x[i] for i in sorted_index]
        test_set_y = [test_set_y[i] for i in sorted_index]
        test_set_word_y = [test_set_word_y[i] for i in sorted_index]

        sorted_index = len_argsort(valid_set_x)
        valid_set_x = [valid_set_x[i] for i in sorted_index]
        valid_set_y = [valid_set_y[i] for i in sorted_index]
        valid_set_word_y = [valid_set_word_y[i] for i in sorted_index]

        sorted_index = len_argsort(train_set_x)
        train_set_x = [train_set_x[i] for i in sorted_index]
        train_set_y = [train_set_y[i] for i in sorted_index]
        train_set_word_y = [train_set_word_y[i] for i in sorted_index]
        #print train_set_word_y
        
    train = (train_set_x, train_set_y)
    valid = (valid_set_x, valid_set_y)
    test = (test_set_x, test_set_y)
    #print test[1]
    word_labels = (train_set_word_y, valid_set_word_y, test_set_word_y)
    return train, valid, test, word_labels

#load_data();
    
        
    
