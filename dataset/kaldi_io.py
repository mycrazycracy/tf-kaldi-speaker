#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2014-2016  Brno University of Technology (author: Karel Vesely)
# Licensed under the Apache License, Version 2.0 (the "License")

#
# Modified:
#     2018  Yi Liu

import numpy as np
import sys, os, re, gzip, struct
import random
from six.moves import range

#################################################
# Adding kaldi tools to shell path,

# Select kaldi,
if not 'KALDI_ROOT' in os.environ:
    # Default! To change run python with 'export KALDI_ROOT=/some_dir python'
    os.environ['KALDI_ROOT']='/mnt/matylda5/iveselyk/Tools/kaldi-trunk'

# Add kaldi tools to path,
os.environ['PATH'] = os.popen('echo $KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lm/:$KALDI_ROOT/src/sgmmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$KALDI_ROOT/src/nnetbin:$KALDI_ROOT/src/nnet2bin:$KALDI_ROOT/src/nnet3bin:$KALDI_ROOT/src/online2bin/:$KALDI_ROOT/src/ivectorbin/:$KALDI_ROOT/src/lmbin/').readline().strip() + ':' + os.environ['PATH']


#################################################
# Define all custom exceptions,
class UnsupportedDataType(Exception): pass
class UnknownVectorHeader(Exception): pass
class UnknownMatrixHeader(Exception): pass

class BadSampleSize(Exception): pass
class BadInputFormat(Exception): pass

class SubprocessFailed(Exception): pass


class FeatureReader(object):
    """Read kaldi features"""
    def __init__(self, data):
        """This is a modified version of read_mat_scp in kaldi_io.

        I wrote the class because we don't want to open and close file frequently.
        The number of file descriptors is limited (= the num of arks) so we can keep all the files open.
        Once the feature archive is opened, it just keeps the file descriptors until the class is closed.

        Args:
            data: The kaldi data directory.
        """
        self.fd = {}
        self.data = data
        self.dim = self.get_dim()
        self.utt2num_frames = {}
        # Load utt2num_frames that the object can find the length of the utterance quickly.
        assert os.path.exists(os.path.join(data, "utt2num_frames")), "[Error] Expect utt2num_frames exists in %s " % data
        with open(os.path.join(data, "utt2num_frames"), 'r') as f:
            for line in f.readlines():
                utt, length = line.strip().split(" ")
                self.utt2num_frames[utt] = int(length)

    def get_dim(self):
        with open(os.path.join(self.data, "feats.scp"), "r") as f:
            dim = self.read(f.readline().strip())[0].shape[1]
        return dim

    def close(self):
        for name in self.fd:
            self.fd[name].close()

    def read(self, file_or_fd, length=None, shuffle=False, start=None):
        """ [mat, start_point] = read(file_or_fd)
         Reads single kaldi matrix, supports ascii and binary.
         file_or_fd : file, gzipped file, pipe or opened file descriptor.

         In our case, file_or_fd can only be a filename with offset. We will save the fd after opening it.

         Note:
             It is really painful to load data from compressed archives. To speed up training, the archives should be
             prepared as uncompressed data. Directly exit if loading data from compressed data. If you really like to
             use that, modify by yourself.
             Maybe other high-performance library can be used to accelerate the loading. No time to try here.
        """
        utt, file_or_fd = file_or_fd.split(" ")
        (filename, offset) = file_or_fd.rsplit(":", 1)
        if filename not in self.fd:
            fd = open(filename, 'rb')
            assert fd is not None
            self.fd[filename] = fd
        # Move to the target position
        self.fd[filename].seek(int(offset))
        try:
            binary = self.fd[filename].read(2).decode()
            if binary == '\0B':
                mat = _read_mat_binary(self.fd[filename])
            else:
                pass
        except:
            raise IOError("Cannot read features from %s" % file_or_fd)

        if length is not None:
            if start is None:
                num_features = mat.shape[0]
                length = num_features if length > num_features else length
                start = random.randint(0, num_features - length) if shuffle else 0
                mat = mat[start:start + length, :]
            else:
                assert not shuffle, "The start point is specified, thus shuffling is invalid."
                mat = mat[start:start + length, :]
        return mat, start

    def read_segment(self, file_or_fd, length=None, shuffle=False, start=None):
        """ [mat, start_point] = read_segment(file_or_fd)
         Reads single kaldi matrix, supports ascii and binary.
         We can load a segment of the feature, rather than the entire recording.
         I hope the segment-wise loading is helpful in the long utterance case.
         file_or_fd : file, gzipped file, pipe or opened file descriptor.

         In our case, file_or_fd can only be a filename with offset. We will save the fd after opening it.
        """
        utt, file_or_fd = file_or_fd.split(" ")
        (filename, offset) = file_or_fd.rsplit(":", 1)
        if filename not in self.fd:
            fd = open(filename, 'rb')
            assert fd is not None
            self.fd[filename] = fd
        # Move to the target position
        self.fd[filename].seek(int(offset))
        try:
            binary = self.fd[filename].read(2).decode()
            if binary == '\0B':
                # Do we need to load the entire recording?
                if length is not None:
                    if start is None:
                        num_features = self.utt2num_frames[utt]
                        length = num_features if length > num_features else length
                        start = random.randint(0, num_features - length) if shuffle else 0
                        mat = _read_submat_binary(self.fd[filename], start, length)
                    else:
                        assert not shuffle, "The start point is specified, thus shuffling is invalid."
                        mat = _read_submat_binary(self.fd[filename], start, length)
                else:
                    mat = _read_mat_binary(self.fd[filename])
            else:
                raise IOError("Cannot read features from %s" % file_or_fd)
        except:
            raise IOError("Cannot read features from %s" % file_or_fd)
        return mat, start


class FeatureReaderV2(object):
    """Read kaldi features and alignments.

    This is used for multitask_v1 training.
    """

    def __init__(self, data_dir, ali_dir, left_context, right_context):
        """data_dir contains feats.scp, utt2num_frames, vad.scp.
        ali_dir contains pdf.scp (NOT ali.scp) which is confusing here.
        ali.scp consists of transition ids while pdf.scp consists of pdf ids.
        So ali.scp should be converted to pdf.scp before training.
        In Kaldi, ali-to-pdf and ali-to-post is used in the egs generation script.

        Args:
            data_dir: The kaldi data directory.
            ali_dir: The kaldi ali directory.
        """
        self.ali_fd = {}
        self.vad_fd = {}
        self.fd = {}
        self.left_context = left_context
        self.right_context = right_context

        self.data_dir = data_dir
        self.ali_dir = ali_dir

        # Load utt2num_frames that the object can find the length of the utterance quickly.
        self.utt2num_frames = {}
        assert os.path.exists(os.path.join(data_dir, "utt2num_frames")), "[Error] Expect utt2num_frames exists in %s " % data_dir
        with open(os.path.join(data_dir, "utt2num_frames"), 'r') as f:
            for line in f.readlines():
                utt, length = line.strip().split(" ")
                self.utt2num_frames[utt] = int(length)

        # We do not have offset here. So we have to record the offset in different files in order to seek them quickly.
        self.utt2feats_offset = {}
        assert os.path.exists(os.path.join(data_dir, "feats.scp")), "[ERROR] Expect feats.scp exists in %s" % data_dir
        with open(os.path.join(data_dir, "feats.scp")) as f:
            for line in f.readlines():
                utt, info = line.strip().split(" ")
                info = info.split(":")
                # info[0] is the filename, info[1] is the offset
                self.utt2feats_offset[utt] = [info[0], int(info[1])]

        self.utt2vad_offset = {}
        assert os.path.exists(os.path.join(data_dir, "vad.scp")), "[ERROR] Expect vad.scp exists in %s" % data_dir
        with open(os.path.join(data_dir, "vad.scp")) as f:
            for line in f.readlines():
                utt, info = line.strip().split(" ")
                info = info.split(":")
                self.utt2vad_offset[utt] = [info[0], int(info[1])]

        self.utt2ali_offset = {}
        assert os.path.exists(os.path.join(ali_dir, "pdf.scp")), "[ERROR] Expect pdf.scp exists in %s" % ali_dir
        with open(os.path.join(ali_dir, "pdf.scp")) as f:
            for line in f.readlines():
                utt, info = line.strip().split(" ")
                info = info.split(":")
                self.utt2ali_offset[utt] = [info[0], int(info[1])]

        self.dim = self.get_dim()

    def get_dim(self):
        with open(os.path.join(self.data_dir, "feats.scp"), "r") as f:
            dim = self.read_segment(f.readline().split(" ")[0])[0].shape[1]
        return dim

    def close(self):
        for name in self.fd:
            self.fd[name].close()
        for name in self.vad_fd:
            self.vad_fd[name].close()
        for name in self.ali_fd:
            self.ali_fd[name].close()

    def read_segment(self, filename, length=None, shuffle=False, start=None):
        """ [mat, vad, ali, start_point] = read_segment(file_or_fd)
         filename : The filename we want to load.

        In order to load vad.scp and pdf.scp as well as feats.scp, we need the name of the feature.
        Unlike FeatureReader, the filename should not contain offset.

        The feaure expansion is applied. The returned feature will be longer than the specified length.
        """
        utt = filename
        feats_filename, feats_offset = self.utt2feats_offset[utt]
        if feats_filename not in self.fd:
            fd = open(feats_filename, 'rb')
            assert fd is not None
            self.fd[feats_filename] = fd
        # Load the features
        self.fd[feats_filename].seek(feats_offset)
        try:
            binary = self.fd[feats_filename].read(2).decode()
            num_features = self.utt2num_frames[utt]
            if binary == '\0B':
                # Do we need to load the entire recording?
                if length is not None:
                    # The length is specified
                    if start is None:
                        # If the length is too long, clip it to #frames
                        length = num_features if length > num_features else length
                        if shuffle:
                            start = random.randint(0, num_features-1)
                            if start + length > num_features:
                                start = num_features - length
                            real_start = start - self.left_context
                            real_length = length + self.left_context + self.right_context
                        else:
                            # Load from the very beginning
                            start = 0
                            real_start = start - self.left_context
                            real_length = length + self.left_context + self.right_context
                    else:
                        assert not shuffle, "The start point is specified, thus shuffling is invalid."
                        if start + length > num_features:
                            # The length is too long that we should shorten it.
                            length = num_features - start
                        # The left_context is considered
                        real_start = start - self.left_context
                        real_length = length + self.left_context + self.right_context
                else:
                    # We want the entire utterance
                    start = 0
                    length = num_features
                    real_start = start - self.left_context
                    real_length = length + self.left_context + self.right_context

                # Load the feature using real_start and real_length
                # Note: The real_start can be < 0 and the real_length can be > num_features
                # Do feature expansion if that happens.
                tmp_start = max(real_start, 0)
                tmp_end = min(real_start + real_length, num_features)
                mat = _read_submat_binary(self.fd[feats_filename], tmp_start, tmp_end - tmp_start)
                if real_start < 0:
                    # Left expansion
                    left_mat = np.tile(mat[0, :], [-real_start, 1])
                    mat = np.concatenate([left_mat, mat], axis=0)

                if real_start + real_length > num_features:
                    # Right expansion
                    right_mat = np.tile(mat[-1, :], [real_start + real_length - num_features, 1])
                    mat = np.concatenate([mat, right_mat], axis=0)
                assert(mat.shape[0] == real_length)
            else:
                raise IOError("Cannot read features from %s" % feats_filename)
        except:
            raise IOError("Cannot read features from %s" % feats_filename)

        # start, length are got from the feature loading.from
        # Use them in the vad and alignment loading.
        vad_filename, vad_offset = self.utt2vad_offset[utt]
        if vad_filename not in self.vad_fd:
            vad_fd = open(vad_filename, 'rb')
            assert vad_fd is not None
            self.vad_fd[vad_filename] = vad_fd
        # Load the vad
        self.vad_fd[vad_filename].seek(vad_offset)
        try:
            binary = self.vad_fd[vad_filename].read(2).decode()
            if binary == '\0B':  # binary flag
                vad = _read_subvec_flt_binary(self.vad_fd[vad_filename], start, length)
            else:  # ascii,
                raise IOError("Cannot read vad from %s" % vad_filename)
        except:
            raise IOError("Cannot read vad from %s" % vad_filename)

        # Use start, length to load alignment
        ali_filename, ali_offset = self.utt2ali_offset[utt]
        if ali_filename not in self.ali_fd:
            ali_fd = open(ali_filename, 'rb')
            assert ali_fd is not None
            self.ali_fd[ali_filename] = ali_fd
        # Load the alignment
        self.ali_fd[ali_filename].seek(ali_offset)
        try:
            binary = self.ali_fd[ali_filename].read(2).decode()
            if binary == '\0B':  # binary flag
                ali = _read_subvec_int_binary(self.ali_fd[ali_filename], start, length)
            else:  # ascii,
                raise IOError("Cannot read ali from %s" % ali_filename)
        except:
            raise IOError("Cannot read ali from %s" % ali_filename)

        assert(mat.shape[0] == vad.shape[0] + self.left_context + self.right_context and
               mat.shape[0] == ali.shape[0] + self.left_context + self.right_context)
        return mat, vad, ali, start


#################################################
# Data-type independent helper functions,

def open_or_fd(file, mode='rb'):
    """ fd = open_or_fd(file)
     Open file, gzipped file, pipe, or forward the file-descriptor.
     Eventually seeks in the 'file' argument contains ':offset' suffix.
    """
    offset = None
    try:
        # strip 'ark:' prefix from r{x,w}filename (optional),
        if re.search('^(ark|scp)(,scp|,b|,t|,n?f|,n?p|,b?o|,n?s|,n?cs)*:', file):
            (prefix,file) = file.split(':',1)
        # separate offset from filename (optional),
        if re.search(':[0-9]+$', file):
            (file,offset) = file.rsplit(':',1)
        # input pipe?
        if file[-1] == '|':
            fd = popen(file[:-1], 'rb') # custom,
        # output pipe?
        elif file[0] == '|':
            fd = popen(file[1:], 'wb') # custom,
        # is it gzipped?
        elif file.split('.')[-1] == 'gz':
            fd = gzip.open(file, mode)
        # a normal file...
        else:
            fd = open(file, mode)
    except TypeError:
        # 'file' is opened file descriptor,
        fd = file
    # Eventually seek to offset,
    if offset != None: fd.seek(int(offset))
    return fd

# based on '/usr/local/lib/python3.4/os.py'
def popen(cmd, mode="rb"):
    if not isinstance(cmd, str):
        raise TypeError("invalid cmd type (%s, expected string)" % type(cmd))

    import subprocess, io, threading

    # cleanup function for subprocesses,
    def cleanup(proc, cmd):
        ret = proc.wait()
        if ret > 0:
            raise SubprocessFailed('cmd %s returned %d !' % (cmd,ret))
        return

    # text-mode,
    if mode == "r":
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        threading.Thread(target=cleanup,args=(proc,cmd)).start() # clean-up thread,
        return io.TextIOWrapper(proc.stdout)
    elif mode == "w":
        proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE)
        threading.Thread(target=cleanup,args=(proc,cmd)).start() # clean-up thread,
        return io.TextIOWrapper(proc.stdin)
    # binary,
    elif mode == "rb":
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        threading.Thread(target=cleanup,args=(proc,cmd)).start() # clean-up thread,
        return proc.stdout
    elif mode == "wb":
        proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE)
        threading.Thread(target=cleanup,args=(proc,cmd)).start() # clean-up thread,
        return proc.stdin
    # sanity,
    else:
        raise ValueError("invalid mode %s" % mode)


def read_key(fd):
    """ [key] = read_key(fd)
     Read the utterance-key from the opened ark/stream descriptor 'fd'.
    """
    key = ''
    while 1:
        char = fd.read(1).decode("latin1")
        if char == '' : break
        if char == ' ' : break
        key += char
    key = key.strip()
    if key == '': return None # end of file,
    assert(re.match('^\S+$',key) != None) # check format (no whitespace!)
    return key


#################################################
# Integer vectors (alignments, ...),

def read_ali_ark(file_or_fd):
    """ Alias to 'read_vec_int_ark()' """
    return read_vec_int_ark(file_or_fd)

def read_vec_int_ark(file_or_fd):
    """ generator(key,vec) = read_vec_int_ark(file_or_fd)
     Create generator of (key,vector<int>) tuples, which reads from the ark file/stream.
     file_or_fd : ark, gzipped ark, pipe or opened file descriptor.

     Read ark to a 'dictionary':
     d = { u:d for u,d in kaldi_io.read_vec_int_ark(file) }
    """
    fd = open_or_fd(file_or_fd)
    try:
        key = read_key(fd)
        while key:
            ali = read_vec_int(fd)
            yield key, ali
            key = read_key(fd)
    finally:
        if fd is not file_or_fd: fd.close()

def read_vec_int(file_or_fd):
    """ [int-vec] = read_vec_int(file_or_fd)
     Read kaldi integer vector, ascii or binary input,
    """
    fd = open_or_fd(file_or_fd)
    binary = fd.read(2).decode()
    if binary == '\0B': # binary flag
        assert(fd.read(1).decode() == '\4'); # int-size
        vec_size = np.frombuffer(fd.read(4), dtype='int32', count=1)[0] # vector dim
        # Elements from int32 vector are sored in tuples: (sizeof(int32), value),
        vec = np.frombuffer(fd.read(vec_size*5), dtype=[('size','int8'),('value','int32')], count=vec_size)
        assert(vec[0]['size'] == 4) # int32 size,
        ans = vec[:]['value'] # values are in 2nd column,
    else: # ascii,
        arr = (binary + fd.readline().decode()).strip().split()
        try:
            arr.remove('['); arr.remove(']') # optionally
        except ValueError:
            pass
        ans = np.array(arr, dtype=int)
    if fd is not file_or_fd : fd.close() # cleanup
    return ans

def _read_subvec_int_binary(fd, start, length):
    assert (fd.read(1).decode() == '\4')  # int-size
    vec_size = np.frombuffer(fd.read(4), dtype='int32', count=1)[0]  # vector dim
    assert start + length <= vec_size
    if start > 0:
        fd.seek(start * 5, 1)
    # Elements from int32 vector are sored in tuples: (sizeof(int32), value),
    vec = np.frombuffer(fd.read(length * 5), dtype=[('size', 'int8'), ('value', 'int32')], count=length)
    assert (vec[0]['size'] == 4)  # int32 size,
    ans = vec[:]['value']  # values are in 2nd column,
    return ans

# Writing,
def write_vec_int(file_or_fd, v, key=''):
    """ write_vec_int(f, v, key='')
     Write a binary kaldi integer vector to filename or stream.
     Arguments:
     file_or_fd : filename or opened file descriptor for writing,
     v : the vector to be stored,
     key (optional) : used for writing ark-file, the utterance-id gets written before the vector.

     Example of writing single vector:
     kaldi_io.write_vec_int(filename, vec)

     Example of writing arkfile:
     with open(ark_file,'w') as f:
       for key,vec in dict.iteritems():
         kaldi_io.write_vec_flt(f, vec, key=key)
    """
    fd = open_or_fd(file_or_fd, mode='wb')
    if sys.version_info[0] == 3: assert(fd.mode == 'wb')
    try:
        if key != '' : fd.write((key+' ').encode("latin1")) # ark-files have keys (utterance-id),
        fd.write('\0B'.encode()) # we write binary!
        # dim,
        fd.write('\4'.encode()) # int32 type,
        fd.write(struct.pack(np.dtype('int32').char, v.shape[0]))
        # data,
        for i in range(len(v)):
            fd.write('\4'.encode()) # int32 type,
            fd.write(struct.pack(np.dtype('int32').char, v[i])) # binary,
    finally:
        if fd is not file_or_fd : fd.close()


#################################################
# Float vectors (confidences, ivectors, ...),

# Reading,
def read_vec_flt_scp(file_or_fd):
    """ generator(key,mat) = read_vec_flt_scp(file_or_fd)
     Returns generator of (key,vector) tuples, read according to kaldi scp.
     file_or_fd : scp, gzipped scp, pipe or opened file descriptor.

     Iterate the scp:
     for key,vec in kaldi_io.read_vec_flt_scp(file):
       ...

     Read scp to a 'dictionary':
     d = { key:mat for key,mat in kaldi_io.read_mat_scp(file) }
    """
    fd = open_or_fd(file_or_fd)
    try:
        for line in fd:
            (key,rxfile) = line.decode().split(' ')
            vec = read_vec_flt(rxfile)
            yield key, vec
    finally:
        if fd is not file_or_fd : fd.close()

def read_vec_flt_ark(file_or_fd):
    """ generator(key,vec) = read_vec_flt_ark(file_or_fd)
     Create generator of (key,vector<float>) tuples, reading from an ark file/stream.
     file_or_fd : ark, gzipped ark, pipe or opened file descriptor.

     Read ark to a 'dictionary':
     d = { u:d for u,d in kaldi_io.read_vec_flt_ark(file) }
    """
    fd = open_or_fd(file_or_fd)
    try:
        key = read_key(fd)
        while key:
            ali = read_vec_flt(fd)
            yield key, ali
            key = read_key(fd)
    finally:
        if fd is not file_or_fd: fd.close()

def read_vec_flt(file_or_fd):
    """ [flt-vec] = read_vec_flt(file_or_fd)
     Read kaldi float vector, ascii or binary input,
    """
    fd = open_or_fd(file_or_fd)
    binary = fd.read(2).decode()
    if binary == '\0B': # binary flag
        # Data type,
        header = fd.read(3).decode()
        if header == 'FV ': sample_size = 4 # floats
        elif header == 'DV ': sample_size = 8 # doubles
        else: raise UnknownVectorHeader("The header contained '%s'" % header)
        assert(sample_size > 0)
        # Dimension,
        assert(fd.read(1).decode() == '\4'); # int-size
        vec_size = np.frombuffer(fd.read(4), dtype='int32', count=1)[0] # vector dim
        # Read whole vector,
        buf = fd.read(vec_size * sample_size)
        if sample_size == 4 : ans = np.frombuffer(buf, dtype='float32')
        elif sample_size == 8 : ans = np.frombuffer(buf, dtype='float64')
        else : raise BadSampleSize
        return ans
    else: # ascii,
        arr = (binary + fd.readline().decode()).strip().split()
        try:
            arr.remove('['); arr.remove(']') # optionally
        except ValueError:
            pass
        ans = np.array(arr, dtype=float)
    if fd is not file_or_fd : fd.close() # cleanup
    return ans

def _read_subvec_flt_binary(fd, start, length):
    # Data type,
    header = fd.read(3).decode()
    if header == 'FV ':
        sample_size = 4  # floats
    elif header == 'DV ':
        sample_size = 8  # doubles
    else:
        raise UnknownVectorHeader("The header contained '%s'" % header)
    assert (sample_size > 0)
    # Dimension,
    assert (fd.read(1).decode() == '\4')  # int-size
    vec_size = np.frombuffer(fd.read(4), dtype='int32', count=1)[0]  # vector dim
    assert start + length <= vec_size
    # seek from the current position
    if start > 0:
        fd.seek(start * sample_size, 1)
    buf = fd.read(length * sample_size)
    if sample_size == 4:
        ans = np.frombuffer(buf, dtype='float32')
    elif sample_size == 8:
        ans = np.frombuffer(buf, dtype='float64')
    else:
        raise BadSampleSize
    return ans

# Writing,
def write_vec_flt(file_or_fd, v, key=''):
    """ write_vec_flt(f, v, key='')
     Write a binary kaldi vector to filename or stream. Supports 32bit and 64bit floats.
     Arguments:
     file_or_fd : filename or opened file descriptor for writing,
     v : the vector to be stored,
     key (optional) : used for writing ark-file, the utterance-id gets written before the vector.

     Example of writing single vector:
     kaldi_io.write_vec_flt(filename, vec)

     Example of writing arkfile:
     with open(ark_file,'w') as f:
       for key,vec in dict.iteritems():
         kaldi_io.write_vec_flt(f, vec, key=key)
    """
    fd = open_or_fd(file_or_fd, mode='wb')
    if sys.version_info[0] == 3: assert(fd.mode == 'wb')
    try:
        if key != '' : fd.write((key+' ').encode("latin1")) # ark-files have keys (utterance-id),
        fd.write('\0B'.encode()) # we write binary!
        # Data-type,
        if v.dtype == 'float32': fd.write('FV '.encode())
        elif v.dtype == 'float64': fd.write('DV '.encode())
        else: raise UnsupportedDataType("'%s', please use 'float32' or 'float64'" % v.dtype)
        # Dim,
        fd.write('\04'.encode())
        fd.write(struct.pack(np.dtype('uint32').char, v.shape[0])) # dim
        # Data,
        fd.write(v.tobytes())
    finally:
        if fd is not file_or_fd : fd.close()


#################################################
# Float matrices (features, transformations, ...),

# Reading,
def read_mat_scp(file_or_fd):
    """ generator(key,mat) = read_mat_scp(file_or_fd)
     Returns generator of (key,matrix) tuples, read according to kaldi scp.
     file_or_fd : scp, gzipped scp, pipe or opened file descriptor.

     Iterate the scp:
     for key,mat in kaldi_io.read_mat_scp(file):
       ...

     Read scp to a 'dictionary':
     d = { key:mat for key,mat in kaldi_io.read_mat_scp(file) }
    """
    fd = open_or_fd(file_or_fd)
    try:
        for line in fd:
            (key,rxfile) = line.decode().split(' ')
            mat = read_mat(rxfile)
            yield key, mat
    finally:
        if fd is not file_or_fd : fd.close()

def read_mat_ark(file_or_fd):
    """ generator(key,mat) = read_mat_ark(file_or_fd)
     Returns generator of (key,matrix) tuples, read from ark file/stream.
     file_or_fd : scp, gzipped scp, pipe or opened file descriptor.

     Iterate the ark:
     for key,mat in kaldi_io.read_mat_ark(file):
       ...

     Read ark to a 'dictionary':
     d = { key:mat for key,mat in kaldi_io.read_mat_ark(file) }
    """
    fd = open_or_fd(file_or_fd)
    try:
        key = read_key(fd)
        while key:
            mat = read_mat(fd)
            yield key, mat
            key = read_key(fd)
    finally:
        if fd is not file_or_fd : fd.close()

def read_mat(file_or_fd):
    """ [mat] = read_mat(file_or_fd)
     Reads single kaldi matrix, supports ascii and binary.
     file_or_fd : file, gzipped file, pipe or opened file descriptor.
    """
    fd = open_or_fd(file_or_fd)
    try:
        binary = fd.read(2).decode()
        if binary == '\0B' :
            mat = _read_mat_binary(fd)
        else:
            assert(binary == ' [')
            mat = _read_mat_ascii(fd)
    finally:
        if fd is not file_or_fd: fd.close()
    return mat


def _read_mat_binary(fd):
    # Data type
    header = fd.read(3).decode()
    # 'CM', 'CM2', 'CM3' are possible values,
    if header.startswith('CM'): return _read_compressed_mat(fd, header)
    elif header == 'FM ': sample_size = 4 # floats
    elif header == 'DM ': sample_size = 8 # doubles
    else: raise UnknownMatrixHeader("The header contained '%s'" % header)
    assert(sample_size > 0)
    # Dimensions
    s1, rows, s2, cols = np.frombuffer(fd.read(10), dtype='int8,int32,int8,int32', count=1)[0]
    # Read whole matrix
    buf = fd.read(rows * cols * sample_size)
    if sample_size == 4 : vec = np.frombuffer(buf, dtype='float32')
    elif sample_size == 8 : vec = np.frombuffer(buf, dtype='float64')
    else : raise BadSampleSize
    mat = np.reshape(vec,(rows,cols))
    return mat


def _read_submat_binary(fd, start, length):
    # Data type
    header = fd.read(3).decode()
    # 'CM', 'CM2', 'CM3' are possible values,
    if header.startswith('CM'): return _read_compressed_submat(fd, header, start, length)
    else:
        raise ValueError("The features should be in the compressed format.")


def _read_mat_ascii(fd):
    rows = []
    while 1:
        line = fd.readline().decode()
        if (len(line) == 0) : raise BadInputFormat # eof, should not happen!
        if len(line.strip()) == 0 : continue # skip empty line
        arr = line.strip().split()
        if arr[-1] != ']':
            rows.append(np.array(arr,dtype='float32')) # not last line
        else:
            rows.append(np.array(arr[:-1],dtype='float32')) # last line
            mat = np.vstack(rows)
            return mat


def _read_compressed_mat(fd, format):
    """ Read a compressed matrix,
        see: https://github.com/kaldi-asr/kaldi/blob/master/src/matrix/compressed-matrix.h
        methods: CompressedMatrix::Read(...), CompressedMatrix::CopyToMat(...),
    """
    assert(format == 'CM ') # The formats CM2, CM3 are not supported...

    # Format of header 'struct',
    global_header = np.dtype([('minvalue','float32'),('range','float32'),('num_rows','int32'),('num_cols','int32')]) # member '.format' is not written,
    per_col_header = np.dtype([('percentile_0','uint16'),('percentile_25','uint16'),('percentile_75','uint16'),('percentile_100','uint16')])

    # Mapping for percentiles in col-headers,
    def uint16_to_float(value, min, range):
        return np.float32(min + range * 1.52590218966964e-05 * value)

    # Mapping for matrix elements,
    def uint8_to_float_v2(vec, p0, p25, p75, p100):
        # Split the vector by masks,
        mask_0_64 = (vec <= 64);
        mask_65_192 = np.all([vec>64, vec<=192], axis=0);
        mask_193_255 = (vec > 192);
        # Sanity check (useful but slow...),
        # assert(len(vec) == np.sum(np.hstack([mask_0_64,mask_65_192,mask_193_255])))
        # assert(len(vec) == np.sum(np.any([mask_0_64,mask_65_192,mask_193_255], axis=0)))
        # Build the float vector,
        ans = np.empty(len(vec), dtype='float32')
        ans[mask_0_64] = p0 + (p25 - p0) / 64. * vec[mask_0_64]
        ans[mask_65_192] = p25 + (p75 - p25) / 128. * (vec[mask_65_192] - 64)
        ans[mask_193_255] = p75 + (p100 - p75) / 63. * (vec[mask_193_255] - 192)
        return ans

    # Read global header,
    globmin, globrange, rows, cols = np.frombuffer(fd.read(16), dtype=global_header, count=1)[0]

    # The data is structed as [Colheader, ... , Colheader, Data, Data , .... ]
    #                         {           cols           }{     size         }
    col_headers = np.frombuffer(fd.read(cols*8), dtype=per_col_header, count=cols)
    data = np.reshape(np.frombuffer(fd.read(cols*rows), dtype='uint8', count=cols*rows), newshape=(cols,rows)) # stored as col-major,

    mat = np.empty((cols,rows), dtype='float32')
    for i, col_header in enumerate(col_headers):
        col_header_flt = [ uint16_to_float(percentile, globmin, globrange) for percentile in col_header ]
        mat[i] = uint8_to_float_v2(data[i], *col_header_flt)

    return mat.T # transpose! col-major -> row-major,


def _read_compressed_submat(fd, format, start, length):
    """ Read a compressed sub-matrix,
        see: https://github.com/kaldi-asr/kaldi/blob/master/src/matrix/compressed-matrix.h
        methods: CompressedMatrix::Read(...), CompressedMatrix::CopyToMat(...),
    """
    assert(format == 'CM ') # The formats CM2, CM3 are not supported...

    # Format of header 'struct',
    global_header = np.dtype([('minvalue','float32'),('range','float32'),('num_rows','int32'),('num_cols','int32')]) # member '.format' is not written,
    per_col_header = np.dtype([('percentile_0','uint16'),('percentile_25','uint16'),('percentile_75','uint16'),('percentile_100','uint16')])

    # Mapping for percentiles in col-headers,
    def uint16_to_float(value, min, range):
        return np.float32(min + range * 1.52590218966964e-05 * value)

    # Mapping for matrix elements,
    def uint8_to_float_v2(vec, p0, p25, p75, p100):
        # Split the vector by masks,
        mask_0_64 = (vec <= 64);
        mask_65_192 = np.all([vec>64, vec<=192], axis=0);
        mask_193_255 = (vec > 192);
        # Sanity check (useful but slow...),
        # assert(len(vec) == np.sum(np.hstack([mask_0_64,mask_65_192,mask_193_255])))
        # assert(len(vec) == np.sum(np.any([mask_0_64,mask_65_192,mask_193_255], axis=0)))
        # Build the float vector,
        ans = np.empty(len(vec), dtype='float32')
        ans[mask_0_64] = p0 + (p25 - p0) / 64. * vec[mask_0_64]
        ans[mask_65_192] = p25 + (p75 - p25) / 128. * (vec[mask_65_192] - 64)
        ans[mask_193_255] = p75 + (p100 - p75) / 63. * (vec[mask_193_255] - 192)
        return ans

    # Read global header,
    globmin, globrange, rows, cols = np.frombuffer(fd.read(16), dtype=global_header, count=1)[0]

    assert rows >= (start + length), "The number of frames is not enough for length %d" % length
    sub_rows = length
    mat = np.zeros((cols, sub_rows), dtype='float32')

    # The data is structed as [Colheader, ... , Colheader, Data, Data , .... ]
    #                         {           cols           }{     size         }
    col_headers = np.frombuffer(fd.read(cols*8), dtype=per_col_header, count=cols)
    col_left = 0
    for i, col_header in enumerate(col_headers):
        col_header_flt = [uint16_to_float(percentile, globmin, globrange) for percentile in col_header]
        # Read data in col-major.
        # It is not necessary to load all the data
        # Seek to the start point from the current position.
        fd.seek(col_left + start, 1)
        data = np.frombuffer(fd.read(length), dtype='uint8', count=length)
        mat[i] = uint8_to_float_v2(data, *col_header_flt)
        col_left = rows - (start + length)
    fd.seek(col_left, 1)

    return mat.T # transpose! col-major -> row-major,


# Writing,
def write_mat(file_or_fd, m, key=''):
    """ write_mat(f, m, key='')
    Write a binary kaldi matrix to filename or stream. Supports 32bit and 64bit floats.
    Arguments:
     file_or_fd : filename of opened file descriptor for writing,
     m : the matrix to be stored,
     key (optional) : used for writing ark-file, the utterance-id gets written before the matrix.

     Example of writing single matrix:
     kaldi_io.write_mat(filename, mat)

     Example of writing arkfile:
     with open(ark_file,'w') as f:
       for key,mat in dict.iteritems():
         kaldi_io.write_mat(f, mat, key=key)
    """
    fd = open_or_fd(file_or_fd, mode='wb')
    if sys.version_info[0] == 3: assert(fd.mode == 'wb')
    try:
        if key != '' : fd.write((key+' ').encode("latin1")) # ark-files have keys (utterance-id),
        fd.write('\0B'.encode()) # we write binary!
        # Data-type,
        if m.dtype == 'float32': fd.write('FM '.encode())
        elif m.dtype == 'float64': fd.write('DM '.encode())
        else: raise UnsupportedDataType("'%s', please use 'float32' or 'float64'" % m.dtype)
        # Dims,
        fd.write('\04'.encode())
        fd.write(struct.pack(np.dtype('uint32').char, m.shape[0])) # rows
        fd.write('\04'.encode())
        fd.write(struct.pack(np.dtype('uint32').char, m.shape[1])) # cols
        # Data,
        fd.write(m.tobytes())
    finally:
        if fd is not file_or_fd : fd.close()


#################################################
# 'Posterior' kaldi type (posteriors, confusion network, nnet1 training targets, ...)
# Corresponds to: vector<vector<tuple<int,float> > >
# - outer vector: time axis
# - inner vector: records at the time
# - tuple: int = index, float = value
#

def read_cnet_ark(file_or_fd):
    """ Alias of function 'read_post_ark()', 'cnet' = confusion network """
    return read_post_ark(file_or_fd)

def read_post_ark(file_or_fd):
    """ generator(key,vec<vec<int,float>>) = read_post_ark(file)
     Returns generator of (key,posterior) tuples, read from ark file.
     file_or_fd : ark, gzipped ark, pipe or opened file descriptor.

     Iterate the ark:
     for key,post in kaldi_io.read_post_ark(file):
       ...

     Read ark to a 'dictionary':
     d = { key:post for key,post in kaldi_io.read_post_ark(file) }
    """
    fd = open_or_fd(file_or_fd)
    try:
        key = read_key(fd)
        while key:
            post = read_post(fd)
            yield key, post
            key = read_key(fd)
    finally:
        if fd is not file_or_fd: fd.close()

def read_post(file_or_fd):
    """ [post] = read_post(file_or_fd)
     Reads single kaldi 'Posterior' in binary format.

     The 'Posterior' is C++ type 'vector<vector<tuple<int,float> > >',
     the outer-vector is usually time axis, inner-vector are the records
     at given time,  and the tuple is composed of an 'index' (integer)
     and a 'float-value'. The 'float-value' can represent a probability
     or any other numeric value.

     Returns vector of vectors of tuples.
    """
    fd = open_or_fd(file_or_fd)
    ans=[]
    binary = fd.read(2).decode(); assert(binary == '\0B'); # binary flag
    assert(fd.read(1).decode() == '\4'); # int-size
    outer_vec_size = np.frombuffer(fd.read(4), dtype='int32', count=1)[0] # number of frames (or bins)

    # Loop over 'outer-vector',
    for i in range(outer_vec_size):
        assert(fd.read(1).decode() == '\4'); # int-size
        inner_vec_size = np.frombuffer(fd.read(4), dtype='int32', count=1)[0] # number of records for frame (or bin)
        data = np.frombuffer(fd.read(inner_vec_size*10), dtype=[('size_idx','int8'),('idx','int32'),('size_post','int8'),('post','float32')], count=inner_vec_size)
        assert(data[0]['size_idx'] == 4)
        assert(data[0]['size_post'] == 4)
        ans.append(data[['idx','post']].tolist())

    if fd is not file_or_fd: fd.close()
    return ans


#################################################
# Kaldi Confusion Network bin begin/end times,
# (kaldi stores CNs time info separately from the Posterior).
#

def read_cntime_ark(file_or_fd):
    """ generator(key,vec<tuple<float,float>>) = read_cntime_ark(file_or_fd)
     Returns generator of (key,cntime) tuples, read from ark file.
     file_or_fd : file, gzipped file, pipe or opened file descriptor.

     Iterate the ark:
     for key,time in kaldi_io.read_cntime_ark(file):
       ...

     Read ark to a 'dictionary':
     d = { key:time for key,time in kaldi_io.read_post_ark(file) }
    """
    fd = open_or_fd(file_or_fd)
    try:
        key = read_key(fd)
        while key:
            cntime = read_cntime(fd)
            yield key, cntime
            key = read_key(fd)
    finally:
        if fd is not file_or_fd : fd.close()

def read_cntime(file_or_fd):
    """ [cntime] = read_cntime(file_or_fd)
     Reads single kaldi 'Confusion Network time info', in binary format:
     C++ type: vector<tuple<float,float> >.
     (begin/end times of bins at the confusion network).

     Binary layout is '<num-bins> <beg1> <end1> <beg2> <end2> ...'

     file_or_fd : file, gzipped file, pipe or opened file descriptor.

     Returns vector of tuples.
    """
    fd = open_or_fd(file_or_fd)
    binary = fd.read(2).decode(); assert(binary == '\0B'); # assuming it's binary

    assert(fd.read(1).decode() == '\4'); # int-size
    vec_size = np.frombuffer(fd.read(4), dtype='int32', count=1)[0] # number of frames (or bins)

    data = np.frombuffer(fd.read(vec_size*10), dtype=[('size_beg','int8'),('t_beg','float32'),('size_end','int8'),('t_end','float32')], count=vec_size)
    assert(data[0]['size_beg'] == 4)
    assert(data[0]['size_end'] == 4)
    ans = data[['t_beg','t_end']].tolist() # Return vector of tuples (t_beg,t_end),

    if fd is not file_or_fd : fd.close()
    return ans


#################################################
# Segments related,
#

# Segments as 'Bool vectors' can be handy,
# - for 'superposing' the segmentations,
# - for frame-selection in Speaker-ID experiments,
def read_segments_as_bool_vec(segments_file):
    """ [ bool_vec ] = read_segments_as_bool_vec(segments_file)
     using kaldi 'segments' file for 1 wav, format : '<utt> <rec> <t-beg> <t-end>'
     - t-beg, t-end is in seconds,
     - assumed 100 frames/second,
    """
    segs = np.loadtxt(segments_file, dtype='object,object,f,f', ndmin=1)
    # Sanity checks,
    assert(len(segs) > 0) # empty segmentation is an error,
    assert(len(np.unique([rec[1] for rec in segs ])) == 1) # segments with only 1 wav-file,
    # Convert time to frame-indexes,
    start = np.rint([100 * rec[2] for rec in segs]).astype(int)
    end = np.rint([100 * rec[3] for rec in segs]).astype(int)
    # Taken from 'read_lab_to_bool_vec', htk.py,
    frms = np.repeat(np.r_[np.tile([False,True], len(end)), False],
                     np.r_[np.c_[start - np.r_[0, end[:-1]], end-start].flat, 0])
    assert np.sum(end-start) == np.sum(frms)
    return frms


if __name__ == "__main__":
    def read(file_or_fd, length=None, shuffle=False):
        """ [mat] = read_mat(file_or_fd)
         Reads single kaldi matrix, supports ascii and binary.
         file_or_fd : file, gzipped file, pipe or opened file descriptor.

         In our case, file_or_fd can only be a filename with offset. We will save the fd after opening it.
        """
        (filename, offset) = file_or_fd.rsplit(":", 1)
        fd = open(filename, 'rb')
        fd.seek(int(offset))

        binary = fd.read(2).decode()
        if binary == '\0B':
            mat, time1, time2, time3 = read_mat_binary(fd)
        else:
            pass

        if length is not None:
            num_features = mat.shape[0]
            length = num_features if length > num_features else length
            start = random.randint(0, num_features - length) if shuffle else 0
            mat = mat[start:start+length, :]
        fd.close()
        return mat, time1, time2, time3

    def read_mat_binary(fd):
        # Data type
        import time
        ts = time.time()
        header = fd.read(3).decode()
        # 'CM', 'CM2', 'CM3' are possible values,
        if header.startswith('CM'):
            return read_compressed_mat(fd, header)
        elif header == 'FM ':
            sample_size = 4  # floats
        elif header == 'DM ':
            sample_size = 8  # doubles
        else:
            raise UnknownMatrixHeader("The header contained '%s'" % header)
        assert (sample_size > 0)
        # Dimensions
        s1, rows, s2, cols = np.frombuffer(fd.read(10), dtype='int8,int32,int8,int32', count=1)[0]
        t1 = time.time() - ts
        # Read whole matrix
        ts = time.time()
        buf = fd.read(rows * cols * sample_size)
        t2 = time.time() - ts
        ts = time.time()
        if sample_size == 4:
            vec = np.frombuffer(buf, dtype='float32')
        elif sample_size == 8:
            vec = np.frombuffer(buf, dtype='float64')
        else:
            raise BadSampleSize
        mat = np.reshape(vec, (rows, cols))
        t3 = time.time() - ts
        return mat, t1, t2, t3

    def read_compressed_mat(fd, format):
        """ Read a compressed matrix,
            see: https://github.com/kaldi-asr/kaldi/blob/master/src/matrix/compressed-matrix.h
            methods: CompressedMatrix::Read(...), CompressedMatrix::CopyToMat(...),
        """
        assert (format == 'CM ')  # The formats CM2, CM3 are not supported...

        # Format of header 'struct',
        global_header = np.dtype([('minvalue', 'float32'), ('range', 'float32'), ('num_rows', 'int32'),
                                  ('num_cols', 'int32')])  # member '.format' is not written,
        per_col_header = np.dtype([('percentile_0', 'uint16'), ('percentile_25', 'uint16'), ('percentile_75', 'uint16'),
                                   ('percentile_100', 'uint16')])

        # Mapping for percentiles in col-headers,
        def uint16_to_float(value, min, range):
            return np.float32(min + range * 1.52590218966964e-05 * value)

        # Mapping for matrix elements,
        def uint8_to_float_v2(vec, p0, p25, p75, p100):
            # Split the vector by masks,
            mask_0_64 = (vec <= 64)
            mask_65_192 = np.all([vec > 64, vec <= 192], axis=0)
            mask_193_255 = (vec > 192)
            # Sanity check (useful but slow...),
            # assert(len(vec) == np.sum(np.hstack([mask_0_64,mask_65_192,mask_193_255])))
            # assert(len(vec) == np.sum(np.any([mask_0_64,mask_65_192,mask_193_255], axis=0)))
            # Build the float vector,
            ans = np.empty(len(vec), dtype='float32')
            ans[mask_0_64] = p0 + (p25 - p0) / 64. * vec[mask_0_64]
            ans[mask_65_192] = p25 + (p75 - p25) / 128. * (vec[mask_65_192] - 64)
            ans[mask_193_255] = p75 + (p100 - p75) / 63. * (vec[mask_193_255] - 192)
            return ans

        import time
        ts = time.time()
        # Read global header,
        globmin, globrange, rows, cols = np.frombuffer(fd.read(16), dtype=global_header, count=1)[0]
        t1 = time.time() - ts
        # The data is structed as [Colheader, ... , Colheader, Data, Data , .... ]
        #                         {           cols           }{     size         }
        ts = time.time()
        col_headers = np.frombuffer(fd.read(cols * 8), dtype=per_col_header, count=cols)
        data = np.reshape(np.frombuffer(fd.read(cols * rows), dtype='uint8', count=cols * rows),
                          newshape=(cols, rows))  # stored as col-major,
        t2 = time.time() - ts

        ts = time.time()
        mat = np.empty((cols, rows), dtype='float32')
        for i, col_header in enumerate(col_headers):
            col_header_flt = [uint16_to_float(percentile, globmin, globrange) for percentile in col_header]
            mat[i] = uint8_to_float_v2(data[i], *col_header_flt)
        t3 = time.time() - ts

        return mat.T, t1, t2, t3  # transpose! col-major -> row-major,

    # data = "/home/dawna/mgb3/transcription/exp-yl695/Snst/xvector/cpdaic_1.0_50/data/voxceleb_train_combined_no_sil"
    data = "/scratch/yl695/voxceleb/data/voxceleb_train_combined_no_sil"
    feats_scp = []
    with open(os.path.join(data, "feats.scp"), "r") as f:
        for line in f.readlines():
            utt, scp = line.strip().split(" ")
            feats_scp.append(scp)
    import random
    import time
    ts = time.time()
    time1 = 0
    time2 = 0
    time3 = 0
    for _ in range(2):
        num_samples = 640
        batch_length = random.randint(200, 400)
        selected = random.sample(feats_scp, num_samples)
        for utt in selected:
            _, t1, t2, t3 = read(utt, batch_length, shuffle=True)
            time1 += t1
            time2 += t2
            time3 += t3
    te = time.time() - ts
    print("Total time: %f s, time 1: %f s, time 2: %f s, time 3: %f s" % (te, time1, time2, time3))
