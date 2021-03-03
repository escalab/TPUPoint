"""
AUTOADJUST
crated by ESCAL Lab at the University of California Riverside
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from google.cloud import storage  # pip install google-cloud-storage
import csv
from time import time
from math import log

# from common import tpu_profiler_hook
# from tensorflow.contrib.tpu.python.tpu import tpu_context
# from tensorflow.python.estimator import model_fn as model_fn_lib
# from tensorflow.core.protobuf import rewriter_config_pb2

# from tensorflow.python.data.ops import dataset_ops

import os
import sys
import copy
import multiprocessing
import itertools
import resource
import pandas as pd # pip install pandas --user
import numpy as np
import weakref
# from common import tpu_profiler_hook # Just test and then use from source when needed
import tensorflow as tf


# #######################################################################################################

def GetSingleValFromTensorInt(tensor):
	tmp = str(tensor.op)
	# json_obj = json.JSONDecoder(tmp)
	tmp = tmp.split('\n')
	possible_ret = []
	for i in tmp:
		if("_val:" in i):
			ret_ = i.split(': ')
			ret_ = [str(_) for _ in ret_]
			possible_ret.append( ret_ )
	# (TODO)need to handle cases for diferent dtypes i.e. float )
	try:
		ret = int( possible_ret[0][1] )
	except:
		ret = -1
	return ret


def GetSingleBoolFromTensorInt(tensor):
	tmp = str(tensor.op)
	# json_obj = json.JSONDecoder(tmp)
	tmp = tmp.split('\n')
	possible_ret = []
	for i in tmp:
		if("_val:" in i):
			ret_ = i.split(': ')
			ret_ = [str(_) for _ in ret_]
			possible_ret.append( ret_ )

	# (TODO)need to handle cases for diferent dtypes i.e. float )
	return bool( possible_ret[0][1] )			


def GetPrefetchSizes(dataset_list):
	prefetch_sizes_ret = []
	for i in range(len(dataset_list)):
		dataset_layer = dataset_list[i]
		if 'PrefetchDataset' in str(dataset_layer):
			prefetch_sizes_ret.append( GetSingleValFromTensorInt(dataset_layer._buffer_size) )
	return prefetch_sizes_ret


def GetMapSizes(dataset_list):
	map_parallel_ret = []
	for i in range(len(dataset_list)):
		dataset_layer = dataset_list[i]
		if 'dataset_ops.MapDataset' in str(dataset_layer.__class__):
			map_parallel_ret.append( 1 ) # If it is a Map, no parallel calls were made
		elif 'dataset_ops.ParallelMapDataset' in str(dataset_layer.__class__):
			map_parallel_ret.append( GetSingleValFromTensorInt( dataset_layer._num_parallel_calls ) )
	return map_parallel_ret


def GetBatchSizes(dataset_list):
	batch_ret = []
	for i in range(len(dataset_list)):
		dataset_layer = dataset_list[i]
		if 'dataset_ops.BatchDataset' in str(dataset_layer.__class__):				
			batch_ret.append( GetSingleValFromTensorInt( dataset_layer._batch_size ) )
	return batch_ret


def GetMapAndBatchSizes(dataset_list):
	map_and_batch_ret = []
	for i in range(len(dataset_list)):
		dataset_layer = dataset_list[i]
		if('_MapAndBatchDataset' in str(dataset_layer)):
			_num_parallel_calls_t = dataset_layer._num_parallel_calls_t # is the num_parallel_calls * batch_size
			_batch_size = GetSingleValFromTensorInt( dataset_layer._batch_size_t )
			# _use_inter_op_parallelism = dataset_layer._use_inter_op_parallelism
			# _batch_size_t = dataset_layer._batch_size_t # Is a Tensor
			ret_num_calls = GetSingleValFromTensorInt(_num_parallel_calls_t) # Get it as int
			original_size = int(ret_num_calls / _batch_size)
			# self.HelperPrinter("\tGetMapAndBatchSizes totall parallel calls: " + str(ret_num_calls))
			# self.HelperPrinter("\tGetMapAndBatchSizes batch size: " + str(_batch_size))
			map_and_batch_ret.append(original_size)
	return map_and_batch_ret


def ChangePrefetch(self, dataset_list, params):
	# if(self.prefetch_modified  == 0):
	# 	tmp_dtype = dataset_layer._buffer_size.dtype
	# 	dataset_layer._buffer_size = tf.convert_to_tensor( int(3) , dtype=tmp_dtype)
	# 	self.prefetch_modified += 1
	prefetch_modified = 0

	prefetch_index = []
	for i in range(len(dataset_list)):
		dataset_layer = dataset_list[i]
		if 'PrefetchDataset' in str(dataset_layer):
			prefetch_index += [i]

	prefetch_buffer_size = params["prefetch_buffer_size"] # Should be a list with all available prefetch types there

	# def ModifyPrefetch(dataset_list, index, prefetch_buffer_size_entry):
	# 	# if( type(prefetch_buffer_size_entry) == type(list()) ): # multiple prefetches inside
	# 	# 	if( self.prefetch_modified < len(prefetch_buffer_size_entry) ):
	# 	# 		dataset_layer = dataset_list[index]
	# 	# 		tmp_dtype = dataset_layer._buffer_size.dtype
	# 	# 		dataset_layer._buffer_size = tf.convert_to_tensor( int( prefetch_buffer_size_entry[self.prefetch_modified] ) , dtype=tmp_dtype)
	# 	# 		return dataset_layer
	# 	if("Autotune" in str(prefetch_buffer_size_entry)): # replace the layer all together with one using AUTOTUNE
	# 		return dataset_list[ max(0, index-1)].prefetch( tf.contrib.data.AUTOTUNE )
	# 	else:
	# 		dataset_layer = dataset_list[index]
	# 		tmp_dtype = dataset_layer._buffer_size.dtype
	# 		dataset_layer._buffer_size = tf.convert_to_tensor( int( prefetch_buffer_size_entry ) , dtype=tmp_dtype)
	# 		return dataset_layer

	# No prefetch found and no prefetch_buffer_size generated
	if(len(prefetch_index) == 0):
		if(prefetch_buffer_size == []): 
			# Do Nothing
			dataset_list = FinalizeDataList(dataset_list)
			return dataset_list
		else:
			# prefetch_buffer_size_ =  prefetch_buffer_size[prefetch_modified]
			prefetch_buffer_size_ =  IntOrAutotune( prefetch_buffer_size[prefetch_modified] )
			# if("Autotune" in str(prefetch_buffer_size_)):
			# 	prefetch_buffer_size_ = tf.contrib.data.AUTOTUNE

			prefetch_op = dataset_list[-1].prefetch( prefetch_buffer_size_ )
			dataset_list.append(prefetch_op)
			dataset_list = FinalizeDataList(dataset_list)
			prefetch_modified += 1
			return dataset_list

				


	# Prefetches exist, replace them with our adjustments
	# self.dataset_v1_list holds the v1 wrappers
	for i in range(len(dataset_list)):
		dataset_layer = dataset_list[i]

		if 'PrefetchDataset' in str(dataset_layer):
			prefetch_buffer_size_ =  prefetch_buffer_size[prefetch_modified]

			if("Autotune" in str(prefetch_buffer_size_)):
				prefetch_buffer_size_ = tf.contrib.data.AUTOTUNE
				# Chreate a new prefetch op to replace the old one
				# IF v1 then use it's index in the list to create the prefetch op, append it into the list

				# if((self.dataset_v1_list != [])):
				# 	prefetch_op = self.dataset_v1_list[max(0, i-1)].prefetch( prefetch_buffer_size_ )
				# 	prefetch_op_v1 = prefetch_op
				# 	prefetch_op = prefetch_op._dataset
				# 	self.dataset_v1_list[i] = prefetch_op_v1
				# 	dataset_list[i] = prefetch_op
				# else:
				prefetch_op = dataset_list[max(0, i-1)].prefetch( prefetch_buffer_size_ )
				dataset_list[i] = prefetch_op


			else:
				tmp_dtype = dataset_layer._buffer_size.dtype
				dataset_layer._buffer_size = tf.convert_to_tensor( int( prefetch_buffer_size_ ) , dtype=tmp_dtype)

			prefetch_modified += 1

	dataset_list = FinalizeDataList(dataset_list)
	return dataset_list


def GetDatasetList(dataset):
	tmp_dataset_ = dataset
	ret = []

	# inputs_list = tmp_dataset_._inputs()
	# while(True):
	# 	try: 
	# 		if('_dataset' in dir(tmp_dataset_)):
	# 			ret.append(tmp_dataset_._dataset)
	# 		else:
	# 			ret.append(tmp_dataset_)
	# 		tmp_dataset_ = inputs_list[0]
	# 		inputs_list = tmp_dataset_._inputs()
	# 	except:
	# 		break

	while(True):
		try:
			if('_dataset' in dir(tmp_dataset_)):
				tmp_dataset_ = tmp_dataset_._dataset
			ret.append(tmp_dataset_)
			tmp_dataset_ = tmp_dataset_._input_dataset
		except:
			break

	ret.reverse()
	return ret


def GetDatasetV1AdapterList(dataset):
	tmp_dataset_ = dataset
	ret = []
	while(True):
		try:
			if('_dataset' in dir(tmp_dataset_)):
				ret.append(tmp_dataset_)
				tmp_dataset_ = tmp_dataset_._dataset # dataset op
			tmp_dataset_ = tmp_dataset_._input_dataset
		except:
			break
	ret.reverse()
	return ret

def IntOrAutotune(val):
	if("Auto" in str(val)):
		return tf.contrib.data.AUTOTUNE
	else:
		return int(val)


def ChangeDtypes(tmp_dataset_list):
	# tmp_dataset_list = dataset_list
	def ChangeDtypes_fn(*args):
		args = list(args)
		for i in range(len(args)):

			# tf.float32 --> tf.bfloat32
			if(args[i].dtype == tf.float32):
				args[i] = tf.cast(args[i], dtype=tf.bfloat16)
				# self.auto_params['change_dtypes'] += '  tf.bfloat32'
		args = tuple(args)
		return args

	index = 1
	for dataset_op in tmp_dataset_list[1:]:
		types_ = list(dataset_op.output_types) if isinstance(dataset_op.output_types, type(tuple())) else [dataset_op.output_types]
		if(tf.float32 in types_):
			index = tmp_dataset_list.index(dataset_op) + 1
			tmp_dataset_list.insert(index,   tmp_dataset_list[max(index-1, 0)].map(ChangeDtypes_fn,  num_parallel_calls=multiprocessing.cpu_count() ) )
			# for j in range(len(tmp_dataset_list)-1):
			# 	tmp_dataset_list[j+1]._input_dataset = tmp_dataset_list[j]
			break
	tmp_dataset_list = FinalizeDataList2(tmp_dataset_list)
	return tmp_dataset_list


def ChangeMapParallel(self, dataset_list, auto_params):
	tmp_dataset_list = dataset_list
	indices = []
	for dataset_op in tmp_dataset_list:
		if( ('dataset_ops.MapDataset' in str(dataset_op.__class__))  or ('dataset_ops.ParallelMapDataset' in str(dataset_op.__class__)) ):
			indices.append( tmp_dataset_list.index(dataset_op) )


	for index in indices:
		map_op = tmp_dataset_list[index]
		index_index = indices.index(index)
		map_op_fn = map_op._map_func._func
		new_call = auto_params['map_parallel'][index_index]

		# PrintStr("MAP OP DIR: " + str(dir(map_op)) )
		# prtstr = "\033[92m test_input_fn: \033[0m ChangeMapParallel new_call: "
		# prtstr += str(new_call) + " @ dataset[" + str(index) + "]: " + str(tmp_dataset_list[index])
		# prtstr += "\n  pmap_op from dataset["+str(index-1)+"]: " + str(tmp_dataset_list[index-1])
		# tf.logging.info(prtstr)

		try:

			# if("Autotune" in str( new_call  )):
			# 	calls = tf.contrib.data.AUTOTUNE
			# else:
			# 	calls = int( new_call )
			calls = IntOrAutotune( new_call )
			# PrintStr("LIST BEFORE CHANGE ChangeMapParallel")
			# PrintDataList(tmp_dataset_list)

			pmap_op = tmp_dataset_list[index-1].map( map_op_fn, num_parallel_calls=calls )
			if('_dataset' in dir(pmap_op)):
				pmap_op = pmap_op._dataset
			tmp_dataset_list[index] = pmap_op

			# PrintStr("LIST AFTER CHANGE ChangeMapParallel")
			# PrintDataList(tmp_dataset_list)
			# FinalizeDataList(tmp_dataset_list)


			# if("Autotune" in str( new_call  )):
			# 	pmap_op = tmp_dataset_list[index-1].map( map_op_fn, num_parallel_calls=tf.contrib.data.AUTOTUNE )
			# 	# prtstr = "\033[92m test_input_fn: \033[0m ChangeMapParallel tf.contrib.data.AUTOTUNE MADE"
			# 	# tf.logging.info(prtstr)
			# else:
			# 	calls = int( new_call )
			# 	pmap_op = tmp_dataset_list[index-1].map( map_op_fn, num_parallel_calls=calls )
			# 	# prtstr = "\033[92m test_input_fn: \033[0m ChangeMapParallel "+str(calls)+" MADE"
			# 	# tf.logging.info(prtstr)

			# tmp_dataset_list[index] = pmap_op
			# for j in range(len(tmp_dataset_list)-1):
			# 	tmp_dataset_list[j+1]._input_dataset = dataset_list[j]
			# prtstr = "\033[92m test_input_fn: \033[0m ChangeMapParallel SUCCESS"
			# tf.logging.info(prtstr)
		except:
			prtstr = "\033[91m test_input_fn: \033[0m ChangeMapParallel FAILED FAILED"
			tf.logging.info(prtstr)
			pass

	# tmp_dataset_list = FinalizeDataList(tmp_dataset_list)
	return tmp_dataset_list


def ChangeVectorizeMap(self, data_list, params):
	map_index = []
	batch_index = []
	batch_size = params["batch_size"]

	for dataset_op in data_list:
		# if('MapDataset' in str(dataset_op.__class__) ):
		if(('dataset_ops.MapDataset' in str(dataset_op.__class__)) or  ('dataset_ops.ParallelMapDataset' in str(dataset_op.__class__)) ):
			map_index.append( data_list.index(dataset_op) )
		if('dataset_ops.BatchDataset' in str(dataset_op.__class__) ):
			batch_index.append( data_list.index(dataset_op) )

	"""
	1) No map, no batch
		- Do Nothing
	2) map, no batch
		- Do Nothing
	3) No map, batch
		- Do Nothing
	4) map, batch
		- Check that the batch comes before or after the map
		- If so, move it and reconnect the pipeline
	"""
	if(batch_index == [] and map_index == []):
		pass
	elif(batch_index == [] and map_index != []):
		pass
	elif(batch_index != [] and map_index == []):
		pass
	elif(batch_index != [] and map_index != []):

		map_i = map_index[0]
		batch_i = batch_index[0]

		if(batch_i > map_i):
			prtstr = "\033[92m AutoadjustClass & ChangeVectorizeMap: \033[0m map_index ->" + str(map_index)
			tf.logging.info(prtstr)
			prtstr = "\033[92m AutoadjustClass & ChangeVectorizeMap: \033[0m batch_index ->" + str(batch_index)
			tf.logging.info(prtstr)


			old_batch = data_list.pop(batch_i)
			# old_batch_size = old_batch._batch_size
			# tmp_dtype = old_batch._batch_size.dtype

			data_list.insert(map_i, old_batch)
			# data_list.insert(map_i, data_list[map_i-1].batch( tf.convert_to_tensor(batch_size, dtype=tmp_dtype) ) )
			# data_list[map_i]._batch_size = old_batch_size

			# for map_i in map_index:
			# 	if(map_i+1 in batch_index): # if batch occurs before the map, move it
			# 		data_list.insert(map_i, data_list.pop(map_i+1))
			# 	else: # Else, add one with whatever the batch size currently is
			# 		prtstr = "\033[92m AutoadjustClass & ChangeVectorizeMap: \033[0m" + str(batch_size)
			# 		tf.logging.info(prtstr)
			# 		data_list.insert(map_i, data_list[map_i-1].batch(batch_size) )


	data_list = FinalizeDataList(data_list)

	prtstr = "\033[92m AutoadjustClass & ChangeVectorizeMap: \033[0m FinalizedDataList() worked!" 
	tf.logging.info(prtstr)

	return data_list


def ChangeVectorizeMap2(data_list, params):
	map_index = []
	batch_index = []
	batch_size = params["batch_size"]

	for dataset_op in data_list:
		# if('MapDataset' in str(dataset_op.__class__) ):
		if(('dataset_ops.MapDataset' in str(dataset_op.__class__)) or  ('dataset_ops.ParallelMapDataset' in str(dataset_op.__class__)) ):
			map_index.append( data_list.index(dataset_op) )
		if('dataset_ops.BatchDataset' in str(dataset_op.__class__) ):
			batch_index.append( data_list.index(dataset_op) )

	"""
	Batch & Map Relationships
	0) No map no batch || batch no map || map but no batch
		- do nothing
	1) Batch before the map
		- Do nothing 
	2) Map before the batch
		- Remove the batch()
		- Apply a map_and_batch() in place of map()
	"""
	if(batch_index != [] and map_index != []):
		# prtstr = "\033[92m AutoadjustClass & ChangeVectorizeMap: \033[0m map_index ->" + str(map_index)
		# tf.logging.info(prtstr)
		# prtstr = "\033[92m AutoadjustClass & ChangeVectorizeMap: \033[0m batch_index ->" + str(batch_index)
		# tf.logging.info(prtstr)
		map_i = map_index[0]
		batch_i = batch_index[0]


		if(batch_i > map_i):
			# prtstr = "\033[92m AutoadjustClass & ChangeVectorizeMap: \033[0m # 2) Map before the batch"
			# tf.logging.info(prtstr)

			# batch_op = data_list.pop(batch_i)
			batch_op = data_list[batch_i]

			# if((self.dataset_v1_list  != [])):
			# 	self.dataset_v1_list.pop(batch_i)

			batch_op_drop = GetSingleBoolFromTensorInt( batch_op._drop_remainder )

			map_op = data_list[map_i]
			map_op_fn = map_op._map_func._func

			# if((self.dataset_v1_list  != [])):
			# 	previous_op = self.dataset_v1_list[map_i-1]
			# else:
			previous_op = data_list[map_i-1]



			try:
				map_op_calls = GetSingleValFromTensorInt( map_op._num_parallel_calls )
				new_map_batch_op = previous_op.apply(
						tf.contrib.data.map_and_batch(
							map_op_fn,
							batch_size=batch_size,
							num_parallel_batches=map_op_calls,
							drop_remainder=batch_op_drop
					)
				)
			except:
				new_map_batch_op = previous_op.apply(
						tf.contrib.data.map_and_batch(
							map_op_fn,
							batch_size=batch_size,
							drop_remainder=batch_op_drop,
							num_parallel_calls=multiprocessing.cpu_count(),
					)
				)

			if('_dataset' in dir(new_map_batch_op)): # Get the real dataset op and not the wraper
				new_map_batch_op = new_map_batch_op._dataset


			data_list[map_i] = new_map_batch_op
			data_list.pop(batch_i)


	data_list = FinalizeDataList(data_list)

	return data_list


def ChangeCacheAfterMap(data_list):
	map_index = []
	cache_index = []

	for dataset_op in data_list:
		if('dataset_ops.MapDataset' in str(dataset_op.__class__) ):
			map_index.append( data_list.index(dataset_op) )
		if('dataset_ops.CacheDataset' in str(dataset_op.__class__) ):
			cache_index.append( data_list.index(dataset_op) )

	for map_i in map_index:
		if not(map_i+1 in cache_index): # if batch occurs before the map, move it
			data_list.insert(map_i, data_list[map_i-1].cache() )

	for j in range(len(data_list)-1):
		data_list[j+1]._input_dataset = data_list[j]

	return data_list				


def ChangeBatchDataset(dataset_list, auto_params):
	tmp_dataset_list = dataset_list
	indices = []
	for dataset_op in tmp_dataset_list:
		if('dataset_ops.BatchDataset' in str(dataset_op.__class__) ):
			indices.append( tmp_dataset_list.index(dataset_op) )



	for index in indices:
		index_index = indices.index(index)
		new_size = auto_params['batch_dataset'][index_index]

		try:
			# print("ChangeBatchDataset "+str(self.auto_params['batch_dataset'])+":")
			# if("Autotune" in str( new_call  )):
			# 	print("\tChangeBatchDataset AUTO")
			# 	# batch_op = tmp_dataset_list[index-1].batch( tf.contrib.data.AUTOTUNE )
			# 	batch_op = tmp_dataset_list[index].batch( tf.contrib.data.AUTOTUNE )
			# 	print("\t\tChangeBatchDataset AUTO WORKS")
			# else:
			# print("\tChangeBatchDataset "+str(new_size))
			# new_size = int( new_size )
			new_size = IntOrAutotune( new_size )
			# batch_op = tmp_dataset_list[index-1].batch( new_size )
			tmp_dtype = tmp_dataset_list[index]._batch_size.dtype
			tmp_dataset_list[index]._batch_size = tf.convert_to_tensor(new_size, dtype=tmp_dtype)

			# print("\tChangeBatchDataset GET HERE 1")
			# tmp_dataset_list[index] = batch_op
			# for j in range(len(tmp_dataset_list)-1):
			# 	tmp_dataset_list[j+1]._input_dataset = dataset_list[j]
		except:
			print("\t\tChangeBatchDataset FAILED FAILED")
			pass

	tmp_dataset_list = FinalizeDataList(tmp_dataset_list)
	return tmp_dataset_list


def ChangeMapAndBatch(self, dataset_list):
	tmp_dataset_list = dataset_list
	indices = []
	for dataset_op in tmp_dataset_list:
		# print("ChangeBatch: " + str(dataset_op.__class__) )
		if('_MapAndBatchDataset' in str(dataset_op) ):
			indices.append( tmp_dataset_list.index(dataset_op) )
	for op_index in indices:
		op = dataset_list[op_index]
		index_index = indices.index(op_index)
		# new_parallel_calls = int( self.auto_params['map_and_batch'][index_index] )
		new_parallel_calls = IntOrAutotune( self.auto_params['map_and_batch'][index_index] )
		#
		# _use_inter_op_parallelism = op._use_inter_op_parallelism
		_num_parallel_calls_t = op._num_parallel_calls_t # is the num_parallel_calls * batch_size
		tmp_dtype = _num_parallel_calls_t.dtype
		#
		_batch_size_t = op._batch_size_t
		_batch_size = GetSingleValFromTensorInt( _batch_size_t )
		# _batch_size = op._batch_size # Use this instead of _batch_size_t as will be -1 if tf.contrib.data.AUTOTUNE is used

		if(new_parallel_calls > 0 ):
			new_num_parallel_calls_t = int(new_parallel_calls * _batch_size)
			op._num_parallel_calls_t = tf.convert_to_tensor(new_num_parallel_calls_t, dtype=tmp_dtype)
			dataset_list[op_index] = op

	dataset_list = FinalizeDataList(dataset_list)
	return dataset_list


def FinalizeDataList(data_list):
	for j in range(len(data_list)-1):
		data_list[j+1]._input_dataset = data_list[j]
	return data_list	


def FinalizeDataList2(data_list):
	for j in range(len(data_list)-1):
		if('ParallelMapDataset' in str(data_list[j+1]) ):
			map_op = data_list[j+1]

			map_op_fn = map_op._map_func._func
			num_calls = GetSingleValFromTensorInt(map_op._num_parallel_calls)
			# print("\t Changing: " + str(data_list[j+1]) + " w/ num_calls: " + str(num_calls) )
			new_map_op = data_list[ j ].map( map_op_fn , num_parallel_calls=num_calls )
			data_list[j+1] = new_map_op

		elif('MapDataset' in str(data_list[j+1]) ):
			# print("\t Changing: " + str(data_list[j+1]) )
			map_op = data_list[j+1]

			map_op_fn = map_op._map_func._func
			new_map_op = data_list[ j ].map( map_op_fn )
			data_list[j+1] = new_map_op

			# pmap_op = data_list[ index-1 ].map( map_op_fn, num_parallel_calls=multiprocessing.cpu_count() )
			# data_list.insert( map_i, data_list.pop(map_i+1))

		else:
			data_list[j+1]._input_dataset = data_list[j]
	data_list = FinalizeDataList(data_list)
	return data_list




def CleanAttribute(argument, attribute, previous_op=None):
	"""
	- element v.s. _tensors
	- dataset v.s. _input_dataset(s)
	- map_func v.s. map_func
	- batch_size v.s. Tensor(batch_size)
	"""		
	if( isinstance(attribute, tf.Tensor) ):
		if( attribute.dtype.is_bool  ):
			# attribute = tf.convert_to_tensor(  GetSingleBoolFromTensorInt(attribute) , dtype=attribute.dtype)
			attribute = GetSingleBoolFromTensorInt(attribute)
		elif( attribute.dtype.is_integer ):
			# attribute = tf.convert_to_tensor( GetSingleValFromTensorInt(attribute) , dtype=attribute.dtype)
			attribute = GetSingleValFromTensorInt(attribute) 
		# else:
		# 	attribute = tf.identity(attribute)


	if("element" in argument):
		# return tf.identity( attribute[0] )

		# attribute = attribute[0] 
		# PrintStr("attribute: " + str(attribute))
		# PrintStr("dir(attribute): " + str(dir(attribute)))
		# PrintStr(str(attribute._op))
		# help(attribute._value_index)

		return  attribute

	for previous_dataset_arg in ["dataset", "input_dataset"]:
		if((previous_dataset_arg in argument) and (previous_op != None)):
			return previous_op

	# for size in ["count", "size", "calls"]:
	# 	if(previous_dataset_arg in argument):
	# 		PrintStr(str(argument) + ": " + str(attribute))
			
	# 		# return GetSingleValFromTensorInt(attribute)
	# 		quit()


	if("map_func" in argument):
		return attribute._func

	return attribute

def GetOriginalAttribute(argument_name, attributes):
	if(argument_name == "element"):
		argument_name = "tensors"
	if( "dataset" == argument_name and "_dataset" in attributes):
		return "_dataset" , attributes["_dataset"]

	found = []
	for attributes_key in attributes:
		if( (argument_name) in attributes_key  ):
			found.append( attributes_key )

	if(len(found) == 0): # argument_name not found in attribute
		return None, None

	found = found[0]
	return found, attributes[found]


def CreateNewDatasetOp(original_op, original_attributes, arguments, previous_op=None):
	op_class = original_op.__class__
	ret_op = None




	for argument in arguments:
		arguments[argument] = CleanAttribute(argument=argument, attribute=arguments[argument], previous_op=previous_op)

	# PrintStr("argument")
	# for new_argument in arguments:
	# 	# PrintStr("\t" + str(new_argument) + " ==> " + str(arguments[new_argument]) + " TYPE: " + str(type(arguments[new_argument])))
	# 	PrintStr("\t" + str(new_argument) + " ==> " + str(arguments[new_argument])  )

	# if("TensorDataset" in str(op_class)):
	# 	def ArtificialDataset(num_samples=300):
	# 		def generator(num_samples):
	# 			import time
	# 			# Opening the file
	# 			time.sleep(0.03)
	# 			for sample_idx in range(num_samples):
	# 				# Reading data (line, record) from the file
	# 				time.sleep(0.015)
	# 				yield (sample_idx,)
	# 		ret = tf.data.Dataset.from_generator(
	# 			generator,
	# 			output_types=tf.int64,
	# 			output_shapes=(1,),
	# 			args=(num_samples,)
	# 		)
	# 		return ret
	# 	ret_op = ArtificialDataset()
	# else:
	# 	ret_op = op_class(**arguments)

	ret_op = op_class(**arguments)
	# quit()
	return ret_op


def ReSetDatasetOp(original, previous_op=None): 
	# new_op = super(original, )__init
	"""
	# # RESETTING THE _VARIANT_TRACKER
	weak_self = weakref.proxy(original)
	original._variant_tracker =  original._track_trackable(
		tf.python.data.ops.dataset_ops._VariantTracker(
			original._variant_tensor,
			# _trace_variant_creation only works when executing eagerly, so we
			# don't want to run it immediately. We also want the _VariantTracker
			# to have a weak reference to the Dataset to avoid creating
			# reference cycles and making work for the garbage collector.
			lambda: weak_self._trace_variant_creation()()
		),  # pylint: disable=unnecessary-lambda,protected-access
		name="_variant_tracker",
		overwrite=True,
	)
	"""

	a = original.__class__
	attributes = original.__dict__
	argument_names = list( original.__init__.__func__.func_code.co_varnames )
	try: argument_names.remove('self') # do not need self argument for initialization
	except: pass
	try: argument_names.remove('variant_tensor') # ignore varaint_tensor argument
	except: pass


	# PrintStr("\n\n" + str(a))
	# PrintStr("attributes")
	# for attributes_key in attributes:
	# 	PrintStr( "\t" + str(attributes_key) + ": " + str(attributes[attributes_key]) )

	new_arguments = {}

	# PrintStr("found attributes")
	for argument_name in argument_names:
		name, found = GetOriginalAttribute(argument_name, attributes)
		# PrintStr( "\t" + str(argument_name) + ": " + str(name))
		if(found != None):
			new_arguments[argument_name] = found


	ret_op = CreateNewDatasetOp(
		original_op=original, 
		original_attributes=attributes, 
		arguments=new_arguments, 
		previous_op=previous_op,
	)
	return ret_op


def FinalizeDataListV1(data_list):
	data_list_ops = data_list
	data_list_v1 = [None for _ in data_list]

	for data_op_index in range(len(data_list_ops)):
		if(data_op_index == 0):
			data_list_ops[data_op_index] = ReSetDatasetOp(data_list_ops[data_op_index])
			# data_list_ops[data_op_index] = tf.identity( data_list_ops[data_op_index] )
			# pass
		else:
			data_list_ops[data_op_index] = ReSetDatasetOp(data_list_ops[data_op_index] , data_list_ops[data_op_index-1])
	return data_list_ops


	# for data_op_index in range(len(data_list_ops)):
	# 	if(data_op_index == 0):
	# 		data_list_v1[data_op_index] =  tf.python.data.ops.dataset_ops.DatasetV1Adapter( data_list_ops[data_op_index] )
	# 	else:
	# 		data_list_ops[data_op_index]._input_dataset = data_list_v1[data_op_index-1]
	# 		data_list_v1[data_op_index] =  tf.python.data.ops.dataset_ops.DatasetV1Adapter( data_list_ops[data_op_index] )
	# return data_list_v1





def CheckingV1Linking(data_list_v1, data_list_ops):
	# data_list_v1_inputs = []
	PrintStr("CHECKING THE INPUTS")
	checks = []
	for i in range(len(data_list_v1)):
		# data_list_v1_inputs.append( data_list_v1[i]._inputs() )
		try:
			data_list_v1_input = data_list_v1[i+1]._inputs()[0]
			data_list_v1_next = data_list_v1[i]
			# print(str(data_list_v1_input) + " == " + str(data_list_v1_next))
			# PrintStr( data_list_v1_input  ==  data_list_v1_next)
			checks.append( data_list_v1_input  ==  data_list_v1_next )
		except:
			pass

	# PrintStr( data_list_ops[-1]  ==  data_list_v1[-1]._dataset )
	checks.append( data_list_ops[-1]  ==  data_list_v1[-1]._dataset )
	if(False in checks):
		PrintStr("\tfailed")
	else:
		PrintStr("\tpassed")


def FinalizeDataListV2(data_list_v1, data_list_ops):
	PrintStr("LEN data_list_v1: " + str(len(data_list_v1)) )
	PrintStr("LEN data_list_ops: " + str(len(data_list_ops)) )

	if((data_list_v1 == None) or (data_list_v1 == [])):
		return data_list_v1, FinalizeDataList(data_list_ops)

	data_list_ops = FinalizeDataList(data_list_ops)
	new_data_list_v1 = []
	for i in range(len(data_list_ops)):
		new_data_list_v1.append( tf.python.data.ops.dataset_ops.DatasetV1Adapter( data_list_ops[i] ) )
	return new_data_list_v1, data_list_ops
	# class DatasetV1Adapter


def PrintStr(lines):
	prtstr = "\033[92m TPUPoint: \033[0m" + " " + str(lines)
	print(prtstr)
	tf.logging.info(prtstr)
	

def PrintDataList(data_list):
	for i in range(len(data_list)):
		# prtstr = "\033[92m test_input_fn: \033[0m" + "i["+str(i)+"]: " + str(data_list[i])
		PrintStr( "i["+str(i)+"]: " + str(data_list[i]) )
		if("TensorDataset" in str(data_list[i])):
			PrintStr("\t _tensors: " + str(data_list[i]._tensors) )


def RecPrintDataset(tmp_dataset_, depth=0):
	PrintStr("["+str(depth)+"]: " + str(tmp_dataset_))
	try:
		if('_dataset' in dir(tmp_dataset_)):		
			tmp_dataset_ = tmp_dataset_._dataset
			PrintStr("["+str(depth)+"]: ._dataset: " + str(tmp_dataset_))		
		tmp_dataset_ = tmp_dataset_._input_dataset
		RecPrintDataset(tmp_dataset_, depth+1)
	except:
		return


class AutoadjustClass(object):
	"""
	Args:
		classifier : user's original TPUEstimator
		valrange   : epsilon range of values to test for to the power of 2
		trainteststeps : Number of steps to run the training test with
	"""
	def __init__(self, 
		input_fn,
		model_fn=None, 
		classifier=None, 
		user_train_batch_size=8,
		valrange=20, 
		num_train_steps=1, 
		printing=False, 
		csvresults=True, 
		num_pipeline_tests=6):

		# self.tpu_name=tpu_name
		# self.tpu_zone=tpu_zone
		# self.gcp_project=gcp_project
		# self.save_checkpoints_steps=save_checkpoints_steps
		# self.log_step_count_steps=log_step_count_steps
		# self.iterations_per_loop=iterations_per_loop
		# self.num_cores=num_cores

		# self.user_config=config

		self.valrange = valrange
		self.num_train_steps = num_train_steps
		self.printing = printing
		self.csvresults = csvresults
		self.num_pipeline_tests = num_pipeline_tests + 1



		self.user_classifier = classifier
		# self.user_model_fn=self.user_classifier._model_fn
		if(self.user_classifier != None):
			if model_fn!=None:
				self.user_model_fn=model_fn 
			else:
				self.user_model_fn=self.user_classifier._model_fn
			self.user_modeldir=self.user_classifier._model_dir
			self.user_config=copy.deepcopy(classifier._config)
			self.user_params=copy.deepcopy(classifier._params)
			self.user_use_tpu=True
			self.user_train_batch_size=self.user_classifier._ctx._train_batch_size
			self.user_eval_batch_size=self.user_classifier._ctx._eval_batch_size
			self.user_predict_batch_size=self.user_classifier._ctx._predict_batch_size
			self.user_batch_axis=None
			self.user_eval_on_tpu=self.user_classifier._ctx._eval_on_tpu
			self.user_export_to_tpu=self.user_classifier._export_to_tpu
			self.user_warm_start_from=None
			self.user_log_every_n_steps=self.user_classifier._log_every_n_steps
			self.user_prefetch_buffer_size=None
		else:
			self.user_model_fn=None
			self.user_modeldir=None
			self.user_config=None
			self.user_params=None
			self.user_use_tpu=False
			self.user_train_batch_size=user_train_batch_size
			self.user_eval_batch_size=None
			self.user_predict_batch_size=None
			self.user_batch_axis=None
			self.user_eval_on_tpu=False
			self.user_export_to_tpu=False
			self.user_warm_start_from=None
			self.user_log_every_n_steps=None
			self.user_prefetch_buffer_size=None
		self.user_exec_time="USER ORIGINAL PARAMS"

		try:
			self.user_input_fn= copy.copy(input_fn)
		except:
			self.user_input_fn= input_fn
		# self.user_model_input_types= self.GetModelInputTypes()
		
		self.auto_params = {}
		self.auto_tests_csv_path = "TPUPoint_autotest_results.csv"

		


		self.keywords = ["model_fn",
						"model_dir",
						"config",
						"params",
						"use_tpu",
						"train_batch_size",
						"eval_batch_size",
						"predict_batch_size",
						"batch_axis",
						"eval_on_tpu",
						"export_to_tpu",
						"warm_start_from",
						"log_every_n_steps",
						"prefetch_buffer_size",
						"change_dtypes",
						"map_parallel",
						"vectorize_map",
						"batch_dataset",
						"map_and_batch"
						]
		self.WriteCSV( self.keywords + ["exec_time" , "mem_usage", "User_Default_Test"] )

		self.SetParams()

		# This will hold the same information & format as the CSV for the results of the testing
		self.train_results=[]
		self.train_params_results={}
		self.train_test_counts = 0
		self.train_test_totals = 0
		self.train_test_model_fn=self.test_model_fn_fn
		self.modified_input_fn=self.GetModifiedDataset

		self.ckpt= "" #"gs://abe_ucr_bucket2/resnet/model.ckpt-95076"
		self.Setckpt() # TODO, should evaltually be manualy entered

		self.TestingEarlyStop = False

  ### GLOBAL FUNCTIONS USED ###

	def HelperPrinter(self, string, status=0):
		if(self.printing):
			if(status <= 1): # Green
				tf.logging.info("\033[92m TPUPoint: \033[0m" + string)
				print("\033[92m TPUPoint: \033[0m" + string)
			elif(status == 2): # Yellow Warning 
				tf.logging.info("\033[93m TPUPoint: \033[0m" + string)
				print("\033[93m TPUPoint: \033[0m" + string)
			elif(status == 3): # Red Fail
				tf.logging.info("\033[91m TPUPoint: \033[0m" + string)
				print("\033[91m TPUPoint: \033[0m" + string)
			else: # Green 
				tf.logging.info("\033[92m TPUPoint: \033[0m" + string)
				print("\033[92m TPUPoint: \033[0m" + string)

	def GetDatasetList(self, dataset):
		return GetDatasetList(dataset)

	def PrintDataList(self, data_list):
		
		PrintDataList(data_list)

  ### MODIFIED INPUT FN CLASS ###

	class test_input_fn(object):
		def __init__(self, ret_input_fn, auto_params, user_test=False):
			self.ret_input_fn=ret_input_fn
			self.auto_params=auto_params
			self.user_test = user_test
			self.prefetch_modified = 0
			self.map_modified = 0
			self.dataset_v1_list = None


		def GetParams(self):
			return self.auto_params

		def ModifyInputFn(self, input_fn, params):
			tmp_dataset = input_fn(params)


			tmp_dataset_list = GetDatasetList(tmp_dataset)
			# self.dataset_v1_list = GetDatasetV1AdapterList(tmp_dataset)


			# Modify the dataset batch API, this does not relate to the batch size of the datset itself, only the API
			if('False' not in str(self.auto_params['batch_dataset'])):
				tmp_dataset_list = ChangeBatchDataset(tmp_dataset_list, self.auto_params)
			elif(": [" not in self.auto_params['batch_dataset'] ): # first time writting this in the 
				self.auto_params['batch_dataset'] += ": " + str( GetBatchSizes(tmp_dataset_list) )



			# Go up the dataset layers and find each of the Dataset class layers
			if('False' not in str(self.auto_params['prefetch_buffer_size'])):
				tmp_dataset_list = ChangePrefetch(self, tmp_dataset_list,  self.auto_params)
				# PrintDataList(tmp_dataset_list)


			elif(": [" not in self.auto_params['prefetch_buffer_size'] ): # first time writting this in the 
				self.auto_params['prefetch_buffer_size'] += ": " + str( GetPrefetchSizes(tmp_dataset_list) )



			# Making any non-parallel map calls to map parallel
			if('False' not in str(self.auto_params['map_parallel'])):
				# PrintStr("RIGHT BEFORE ChangeMapParallel")
				# PrintDataList(tmp_dataset_list)
				tmp_dataset_list = ChangeMapParallel(self, tmp_dataset_list, self.auto_params)
				# PrintStr("RIGHT AFTER ChangeMapParallel")
				# PrintDataList(tmp_dataset_list)
			elif(": [" not in self.auto_params['map_parallel'] ):
				self.auto_params['map_parallel'] += ": " + str( GetMapSizes(tmp_dataset_list) )



			# Add a batch op before map
			if('False' not in self.auto_params['vectorize_map'] ):
				tmp_dataset_list = ChangeVectorizeMap2(tmp_dataset_list, params) 


			# inserting a map op to reduce float32-->bfloat16 for faster infreed processing
			if('False' not in str(self.auto_params['change_dtypes'])):
				tmp_dataset_list = ChangeDtypes(tmp_dataset_list)
				# PrintDataList(tmp_dataset_list)



			# Change the number of parallel calls for a map_and_batch operation
			if('False' not in str(self.auto_params['map_and_batch'])):
				tmp_dataset_list = ChangeMapAndBatch(self, tmp_dataset_list) # Not implemented Yet
			elif(": [" not in self.auto_params['map_and_batch']):
				self.auto_params['map_and_batch'] += ": " + str(GetMapAndBatchSizes(tmp_dataset_list))

			# #############################

			# Add a cache op after map operation
			# tmp_dataset_list = ChangeCacheAfterMap(tmp_dataset_list)
			# PrintDataList(tmp_dataset_list)

			# self.dataset_v1_list, tmp_dataset_list = FinalizeDataListV1(self.dataset_v1_list, tmp_dataset_list)
			# if(self.dataset_v1_list != []):
			# 	PrintStr("RETURNING DATASET_V1")
			# 	tmp_dataset_v1 = self.dataset_v1_list[-1]
			# 	return tmp_dataset_v1
			# else:
			# 	tmp_dataset = tmp_dataset_list[-1] 
			# 	return tmp_dataset


			# PrintStr("RIGHT AFTER FinalizeDataList")
			# PrintDataList(tmp_dataset_list)
			# tmp_dataset_list = FinalizeDataList(tmp_dataset_list)
			# tmp_dataset = tmp_dataset_list[-1]
			
			# try:
			tmp_dataset_list = FinalizeDataListV1(tmp_dataset_list)
			tmp_dataset = tmp_dataset_list[-1]
			tmp_dataset =  tf.python.data.ops.dataset_ops.DatasetV1Adapter( tmp_dataset )
			# except:
			# 	tmp_dataset_list = FinalizeDataList(tmp_dataset_list)
			# 	tmp_dataset = tmp_dataset_list[-1]
			# 	try: 
			# 		tmp_dataset =  tf.python.data.ops.dataset_ops.DatasetV1Adapter( tmp_dataset )
			# 	except:
			# 		pass
			return tmp_dataset


		def __call__(self, params):
			if(self.user_test):
				return self.ret_input_fn(params)
			return self.ModifyInputFn(self.ret_input_fn, params)

		def GetModifiedDataset(self, params):
			# used_params = self.self.auto_params
			# used_params.update(params)
			return self.ModifyInputFn(self.ret_input_fn, params)

		def GetUserDatset(self, params):

			return self.ret_input_fn(params)


	# # (TODO) Not being used currently, make sure safe to delete
	class test_model_fn(object):

		def __init__(self, model_fn, input_types):
			self.user_model_fn=model_fn
			self.user_input_types=input_types

		def __call__(self, features, labels, mode, params):
			if(features.dtype != self.user_input_types[0]):
				features = tf.cast(features, dtype=self.user_input_types[0])
			if(labels.dtype != self.user_input_types[1]):
				labels = tf.cast(labels, dtype=self.user_input_types[1])

			# help(self.model_fn)
			# help(self.user_model_fn)
			# ret = self.user_model_fn(features, labels, mode, params) 
			# return ret
			return self.user_model_fn(features, labels, mode, params) 

	def test_model_fn_fn(self, features, labels, mode, params):
		# feature & labels dont match
		if((features.dtype != self.user_model_input_types[0]) and (labels.dtype != self.user_model_input_types[1])):
			return self.user_model_fn( tf.cast(features, dtype=self.user_model_input_types[0]), tf.cast(labels, dtype=self.user_model_input_types[1]), mode, params)

		# feature dont match
		elif((features.dtype != self.user_model_input_types[0]) and (labels.dtype == self.user_model_input_types[1])):
			return self.user_model_fn( tf.cast(features, dtype=self.user_model_input_types[0]) , labels, mode, params)

		# labels dont match
		elif((features.dtype == self.user_model_input_types[0]) and (labels.dtype != self.user_model_input_types[1])):
			return self.user_model_fn( features, tf.cast(labels, dtype=self.user_model_input_types[1]), mode, params)

		# feature & lable match
		elif((features.dtype == self.user_model_input_types[0]) and (labels.dtype != self.user_model_input_types[1])):
			return self.user_model_fn(features, labels, mode, params)

		return self.user_model_fn(features, labels, mode, params)

		# if(features.dtype != self.user_model_input_types[0]):
		# 	features = tf.cast(features, dtype=self.user_model_input_types[0])
		# if(labels.dtype != self.user_model_input_types[1]):
		# 	labels = tf.cast(labels, dtype=self.user_model_input_types[1])

		# ret = self.user_model_fn(features, labels, mode, params)
		# return ret



	def WriteCSV(self, lis):
		if(self.csvresults):
			with open(self.auto_tests_csv_path, 'a') as csvfile:
				event_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
				event_writer.writerow(lis)

	def GetUserParams(self):
		if(self.user_classifier==None):
			return
		self.user_train_batch_size = self.user_classifier._ctx._train_batch_size
		self.user_eval_batch_size = self.user_classifier._ctx._eval_batch_size
		self.user_predict_batch_size = self.user_classifier._ctx._predict_batch_size
		self.user_modeldir = self.user_classifier._model_dir

	def Setckpt(self, path=""):
		if(self.user_classifier == None):
			return
		tmp_warm_start_from = tf.train.latest_checkpoint(self.user_modeldir)
		# base_step_number =  tf.train.latest_checkpoint("gs://abe_ucr_bucket2/resnet/") # (TODO) change to what user_modeldir is
		# base_step_number = int( base_step_number.split(".ckpt-")[1] )

		if("gs://" not in path):
			# self.HelperPrinter("invalid ckpt path given")
			self.ckpt = tmp_warm_start_from
		else:
			self.ckpt = path
		return

	def SetParams(self):

		def ProduceParamsFromValue(original_val, valrange=self.valrange, added_option=[], minimum=8):
			ret = []
			if(original_val==None):
				return ret
			if(original_val==-1): # using tf.contrib.data.AUTOTUNE will return prefetch buffer size -1
				original_val = 8
			# if( original_val < 8):
			# 	original_val = 8
			base = int(log(original_val, 2))
			# base = max(minimum,2**(base-valrange))
			base = max(minimum,2**(base-1))

			min_range = max(minimum , original_val-(base*valrange) )
			max_range = original_val+(base*valrange)

			for i in range( min_range  , max_range , base):
				ret.append(i)
			
			if(original_val not in ret):
				ret.append( original_val )

			ret += added_option
			ret.sort()
			orig_index = ret.index(original_val)
			# ret_min = max(0,orig_index-valrange)
			# ret_max = min( int(len(ret)) , orig_index+valrange )
			# ret = ret[ret_min:ret_max]

			return ret

		def ProduceParamsFromList(original_list, valrange=self.valrange, added_option=[], minimum=8):
			ranges = []
			for original_val in original_list:
				ranges_ret = ProduceParamsFromValue(original_val=original_val , added_option=added_option, valrange=valrange, minimum=minimum) 
				ranges.append( ranges_ret ) 


			

			if(original_list == []):
				ranges = [added_option]

			ret = list(itertools.product(*ranges))
			ret = [list(_) for _ in ret] 

			# Gard against too many values to test
			valrange = 5 # MaxValRange
			test_length = len(ret) * self.num_pipeline_tests
			max_test_length = self.num_pipeline_tests * valrange
			if( max_test_length < test_length):
				self.HelperPrinter("Reducing number of tests", 1)
				tmp_ret = []
				step_size = int(len(ret) / valrange)
				for i in range(0,len(ret), step_size):
					tmp_ret.append(ret[i])

				# Make sure last largest test is added
				if(ret[-1] not in tmp_ret): 
					tmp_ret.append(ret[-1])
				ret = tmp_ret

			return ret

		params = {}
		params["batch_size"] = self.user_train_batch_size
		params = self.GetQuickParams(params)
		tmp_dataset = self.user_input_fn(params)
		tmp_dataset_list = GetDatasetList(tmp_dataset)

		prefetch_sizes = GetPrefetchSizes(tmp_dataset_list)
		self.HelperPrinter("original prefetch_sizes: " + str(prefetch_sizes))
		prefetch_sizes = ProduceParamsFromList(prefetch_sizes, added_option=['Autotune'])
		# prefetch_sizes = prefetch_sizes[int(len(prefetch_sizes)/2) : ]
		# prefetch_sizes = ProduceParamsFromList(prefetch_sizes)
		self.HelperPrinter(" test prefetch_sizes: " + str(prefetch_sizes))
		
		map_sizes = GetMapSizes(tmp_dataset_list)
		self.HelperPrinter("original map_sizes: " + str(map_sizes))
		# map_sizes = ProduceParamsFromList(map_sizes, added_option=['Autotune'])
		map_sizes = ProduceParamsFromList(map_sizes)
		self.HelperPrinter(" test map_sizes: " + str(map_sizes))


		# batch_sizes = GetBatchSizes(tmp_dataset_list)
		# self.HelperPrinter("original batch_sizes: " + str(batch_sizes))
		# batch_sizes = ProduceParamsFromList(batch_sizes, added_option=[batch_sizes[0]*2,batch_sizes[0]*3])
		# batch_sizes = ProduceParamsFromList(batch_sizes)
		# self.HelperPrinter(" test batch_sizes: " + str(batch_sizes))

		map_and_batch_sizes = GetMapAndBatchSizes(tmp_dataset_list)
		self.HelperPrinter("original map_and_batch_sizes: " + str(map_and_batch_sizes)) # (TODO) Check for tf.contrib.data.AUTOTUNE
		map_and_batch_sizes = ProduceParamsFromList(map_and_batch_sizes)
		self.HelperPrinter(" test map_and_batch_sizes: " + str(map_and_batch_sizes))


		# No longer modifying the batch size
		# self.auto_params["train_batch_size"] = ProduceParamsFromValue(self.user_train_batch_size)
		# self.auto_params["eval_batch_size"] = ProduceParamsFromValue(self.user_eval_batch_size)
		# self.auto_params["predict_batch_size"] = ProduceParamsFromValue(self.user_predict_batch_size)


		self.auto_params["prefetch_buffer_size"] =  ['False'] + prefetch_sizes #if prefetch_sizes!=[] else []
		self.auto_params["map_parallel"] = ['False'] + map_sizes #if map_sizes!=[] else []
		self.auto_params["vectorize_map"] = ['True', 'False'] # ['False','True'] # ['False'] # ['True'] # 
		self.auto_params["map_and_batch"] = ['False'] + map_and_batch_sizes #if map_and_batch_sizes!=[] else []
		# self.auto_params["change_dtypes"] = ['False'] #['False','True']
		# self.auto_params["batch_dataset"] = ['False'] #  batch_sizes + ['False']  # 

		p1 = max(1, len(self.auto_params["prefetch_buffer_size"]))
		p2 = max(1, len(self.auto_params["map_parallel"]))
		p4 = max(1, len(self.auto_params["vectorize_map"]))
		p6 = max(1, len(self.auto_params["map_and_batch"]))
		# p3 = max(1, len(self.auto_params["change_dtypes"]))
		# p5 = max(1, len(self.auto_params["batch_dataset"]))
		num_tests = self.num_pipeline_tests

		tests =  (p1*p2*p4*p6*num_tests) + num_tests
		self.HelperPrinter(" TestingDatapipeline Estimated tests = " + str(tests))

		p1 = len(self.auto_params["prefetch_buffer_size"])
		p2 = len(self.auto_params["map_parallel"])
		p4 = len(self.auto_params["vectorize_map"])
		p6 = len(self.auto_params["map_and_batch"])
		# p3 = len(self.auto_params["change_dtypes"])
		# p5 = len(self.auto_params["batch_dataset"])

		tests =  ((p1+p2+p4+p6)*num_tests) + num_tests
		self.HelperPrinter(" TestingDatapipeline2 Estimated tests = " + str(tests))

	def PrintOriginalSizes(self, given_input_fn):


		params = {}
		params["batch_size"] = self.user_train_batch_size
		params = self.GetQuickParams(params)
		tmp_dataset = given_input_fn(params)
		tmp_dataset_list = GetDatasetList(tmp_dataset)
		tmp_dataset_list_v1 = GetDatasetV1AdapterList(tmp_dataset)
		PrintStr("FROM PrintOriginalSizes")
		PrintDataList(tmp_dataset_list)
		# CheckingV1Linking(tmp_dataset_list_v1, tmp_dataset_list)
		# PrintStr("FROM RecPrintDataset")
		# RecPrintDataset(tmp_dataset)


		prefetch_sizes = GetPrefetchSizes(tmp_dataset_list)
		self.HelperPrinter(" prefetch_sizes: " + str(prefetch_sizes))
		
		map_sizes = GetMapSizes(tmp_dataset_list)
		self.HelperPrinter(" map_sizes: " + str(map_sizes))


		batch_sizes = GetBatchSizes(tmp_dataset_list)
		self.HelperPrinter(" batch_sizes: " + str(batch_sizes))

		map_and_batch_sizes = GetMapAndBatchSizes(tmp_dataset_list)
		self.HelperPrinter(" map_and_batch_sizes: " + str(map_and_batch_sizes)) # (TODO) Check for tf.contrib.data.AUTOTUNE


	def GetAutoParams(self, 
		auto_model_fn=None, 
		auto_model_dir=None, 
		auto_config=None, 
		auto_params=None, 
		auto_use_tpu=None, 
		auto_train_batch_size=None, 
		auto_eval_batch_size=None, 
		auto_predict_batch_size=None, 
		auto_batch_axis=None, 
		auto_eval_on_tpu=None, 
		auto_export_to_tpu=None, 
		auto_warm_start_from=None, 
		auto_log_every_n_steps=None, 
		auto_prefetch_buffer_size=None,
		auto_change_dtypes=None,
		auto_map_parallel=None,
		auto_vectorize_map=None,
		auto_batch_dataset=None,
		auto_map_and_batch=None):



		# self.user_model_fn=model_fn
		# self.user_modeldir=self.user_classifier._model_dir
		# self.user_config=copy.deepcopy(classifier._config)
		# self.user_params=copy.deepcopy(classifier._params)
		# self.user_use_tpu=True
		# self.user_train_batch_size=self.user_classifier._ctx._train_batch_size
		# self.user_eval_batch_size=self.user_classifier._ctx._eval_batch_size
		# self.user_predict_batch_size=self.user_classifier._ctx._predict_batch_size
		# self.user_batch_axis=None
		# self.user_eval_on_tpu=self.user_classifier._ctx._eval_on_tpu
		# self.user_export_to_tpu=self.user_classifier._export_to_tpu
		# self.user_warm_start_from=None
		# self.user_log_every_n_steps=self.user_classifier._log_every_n_steps
		# self.user_prefetch_buffer_size=None


		if(auto_model_fn==None):
			# auto_model_fn=self.user_classifier._model_fn
			auto_model_fn=self.train_test_model_fn 
		if(auto_model_dir==None):
			auto_model_dir=self.user_modeldir
			# auto_model_dir=self.user_classifier._model_dir

		if(auto_config==None):
			auto_config=self.user_config
			# auto_config=self.user_classifier._config
			# auto_config=copy.deepcopy(self.user_config)
			# auto_config= tf.contrib.tpu.RunConfig(cluster=


		if(auto_params==None):
			auto_params=self.user_params
		if(auto_use_tpu==None):
			auto_use_tpu=self.user_use_tpu
			# auto_use_tpu=True
			# auto_use_tpu=self.user_classifier._ctx._use_tpu


		if(auto_train_batch_size==None):
			auto_train_batch_size=self.user_train_batch_size
			# auto_train_batch_size=self.user_classifier._ctx._train_batch_size
			# auto_train_batch_size=None
		if(auto_eval_batch_size==None):
			auto_eval_batch_size=self.user_eval_batch_size
			# auto_eval_batch_size=self.user_classifier._ctx._eval_batch_size
		if(auto_predict_batch_size==None):
			auto_predict_batch_size=self.user_predict_batch_size
			# auto_predict_batch_size=self.user_classifier._ctx._predict_batch_size

		# if(auto_batch_axis==None):
		# 	auto_batch_axis=self.user_classifier._batch_axis
		if(auto_eval_on_tpu==None):
			auto_eval_on_tpu=self.user_eval_on_tpu
			# auto_eval_on_tpu=self.user_classifier._ctx._eval_on_tpu
		if(auto_export_to_tpu==None):
			auto_export_to_tpu=self.user_export_to_tpu
			# auto_export_to_tpu=self.user_classifier._export_to_tpu
			# auto_export_to_tpu=True

		if(auto_log_every_n_steps==None):
			auto_log_every_n_steps=self.user_log_every_n_steps
			# auto_log_every_n_steps=self.user_classifier._log_every_n_steps
			# auto_config.log_step_count_steps=auto_log_every_n_steps
			# auto_config._log_every_n_steps=auto_log_every_n_steps

		if(auto_prefetch_buffer_size==None):
			auto_prefetch_buffer_size="False"
		if(auto_change_dtypes==None):
			auto_change_dtypes = "False"
		if(auto_map_parallel==None):
			auto_map_parallel = "False"
		if(auto_vectorize_map==None):
			auto_vectorize_map = "False"
		if(auto_batch_dataset==None):
			auto_batch_dataset = "False"
		if(auto_map_and_batch==None):
			auto_map_and_batch = "False"

		ret = {"model_fn" : auto_model_fn,
				"model_dir" : auto_model_dir,
				"config" : auto_config,
				"params" : auto_params,
				"use_tpu" : auto_use_tpu,
				"train_batch_size" : auto_train_batch_size,
				"eval_batch_size" : auto_eval_batch_size,
				"predict_batch_size" : auto_predict_batch_size,
				"batch_axis" : auto_batch_axis,
				"eval_on_tpu" : auto_eval_on_tpu,
				"export_to_tpu" : auto_export_to_tpu,
				"warm_start_from" : auto_warm_start_from,
				"log_every_n_steps" : auto_log_every_n_steps,
				"prefetch_buffer_size" : auto_prefetch_buffer_size, #}
				"change_dtypes" : auto_change_dtypes,#}
				"map_parallel" : auto_map_parallel,#}
				"vectorize_map" : auto_vectorize_map,
				"batch_dataset" : auto_batch_dataset,
				"map_and_batch" : auto_map_and_batch}

		ret = self.GetQuickParams(ret)
		return ret


	def GetTrainParams(self):

		ret = []

		
		# for vectorize_map in self.auto_params["vectorize_map"]:
		# 	ret.append( self.GetAutoParams( 
		# 			auto_model_fn=self.train_test_model_fn,
		# 			auto_vectorize_map=vectorize_map
		# 		)
		# 	)
		# self.train_test_totals = len(ret)
		# return ret


		for prefetch_buffer_size in self.auto_params["prefetch_buffer_size"] :
			for map_parallel in  self.auto_params["map_parallel"] :
				# for change_dtypes in self.auto_params["change_dtypes"] :
				for vectorize_map in self.auto_params["vectorize_map"]:
					# for batch_dataset in self.auto_params["batch_dataset"]:
					for map_and_batch in self.auto_params["map_and_batch"]:
						ret.append( self.GetAutoParams( 
								auto_model_fn=self.train_test_model_fn,
								auto_prefetch_buffer_size=prefetch_buffer_size,
								auto_map_parallel=map_parallel,
								# auto_change_dtypes=change_dtypes,
								auto_vectorize_map=vectorize_map,
								# auto_batch_dataset=batch_dataset,
								auto_map_and_batch=map_and_batch
							)
						)



		self.train_test_totals = len(ret)
		return ret

	# THIS IS WHAT EXECUTES THE TRAIN RUNNING & STORES THE DATA
	def TestingTrain(self, classifier, input_fn, used_params, users_test=False, best_params=False, steps=None):
		"""
		TestingTrain
		Args:
			classifer = tf.contrib.tpu.TPUEstimator
			input_fn = input function possibly of Dataset instance
			user_params = the autogenerated parameters used to test this
		Returns:
			Nothing
		"""


		if(users_test):
			try:
				# self.HelperPrinter("CLEARING DEFAULT GRAPH START")
				# tf.reset_default_graph()
				# self.HelperPrinter("CLEARING DEFAULT GRAPH END")

				self.HelperPrinter(" starting test: user baseline test")

				# use_input_fn = self.test_input_fn(input_fn, used_params, user_test=True)

				# Run once as to avoid capturing the compilation time
				# classifier.train( input_fn=use_input_fn, steps=self.num_train_steps)

				start_stamp = time()
				# classifier.train( input_fn=use_input_fn, steps=self.num_train_steps)
				if(steps==None):
					classifier.train( input_fn=input_fn, steps=self.num_train_steps)
				else:
					classifier.train( input_fn=input_fn, steps=steps)
				# classifier.train( input_fn=use_input_fn, steps=10)
				elapsed_time = time() - start_stamp


				# train(self, input_fn, 
				# 	hooks=None, 
				# 	steps=None, 
				# 	max_steps=None, 
				# 	saving_listeners=None) 
				# p = multiprocessing.Process(target=classifier.train , args=(use_input_fn, None, self.num_train_steps, None, None))
				# self.HelperPrinter("GOT HERE multiprocessing ")
				# p.start()
				# self.HelperPrinter("Gi")
				# p.join()
				# elapsed_time = time() - start_stamp




				user_original_params=[]
				user_original_params.append(str(self.user_model_fn)) # "model_fn"
				user_original_params.append(str(self.user_modeldir)) # "model_dir"
				user_original_params.append(str(self.user_config)) # "config"
				user_original_params.append(str(self.user_params)) # "params"
				user_original_params.append(str(self.user_use_tpu)) # "use_tpu"
				user_original_params.append(str(self.user_train_batch_size)) # "train_batch_size"
				user_original_params.append(str(self.user_eval_batch_size)) # "eval_batch_size"
				user_original_params.append(str(self.user_predict_batch_size)) # "predict_batch_size"
				user_original_params.append(str(None)) # "batch_axis"
				user_original_params.append(str(self.user_eval_on_tpu)) # "eval_on_tpu"
				user_original_params.append(str(self.user_export_to_tpu)) # "export_to_tpu"
				user_original_params.append(str(self.user_warm_start_from)) # "warm_start_from"
				user_original_params.append(str(self.user_log_every_n_steps)) # "log_every_n_steps"
				user_original_params.append(str(self.user_prefetch_buffer_size)) # "prefetch_buffer_size"
				user_original_params.append(str(False)) # "change_dtypes"
				user_original_params.append(str(False)) # "map_parallel",
				user_original_params.append(str(False)) # ""vectorize_map"",
				user_original_params.append(str(elapsed_time)) # "exec_time"

						
				mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
				user_original_params.append(str(mem_usage)) # "mem_usage"


				user_original_params.append("THIS IS THE USER\'S TEST ")

				self.WriteCSV(user_original_params)


				tf.reset_default_graph()			

				return
			except:
				self.HelperPrinter(" ERROR: User Test FAILED")

				user_original_params=[]
				user_original_params.append(str(self.user_model_fn)) # "model_fn"
				user_original_params.append(str(self.user_modeldir)) # "model_dir"
				user_original_params.append(str(self.user_config)) # "config"
				user_original_params.append(str(self.user_params)) # "params"
				user_original_params.append(str(self.user_use_tpu)) # "use_tpu"
				user_original_params.append(str(self.user_train_batch_size)) # "train_batch_size"
				user_original_params.append(str(self.user_eval_batch_size)) # "eval_batch_size"
				user_original_params.append(str(self.user_predict_batch_size)) # "predict_batch_size"
				user_original_params.append(str(None)) # "batch_axis"
				user_original_params.append(str(self.user_eval_on_tpu)) # "eval_on_tpu"
				user_original_params.append(str(self.user_export_to_tpu)) # "export_to_tpu"
				user_original_params.append(str(self.user_warm_start_from)) # "warm_start_from"
				user_original_params.append(str(self.user_log_every_n_steps)) # "log_every_n_steps"
				user_original_params.append(str(self.user_prefetch_buffer_size)) # "prefetch_buffer_size"
				user_original_params.append(str(False)) # "change_dtypes"
				user_original_params.append(str(False)) # "map_parallel",
				user_original_params.append(str(False)) # ""vectorize_map"",
				user_original_params.append(str("ERROR: FAILED TEST")) # "exec_time"

				

				mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
				user_original_params.append(str(mem_usage)) # "mem_usage"


				
				user_original_params.append("THIS IS THE USER\'S TEST ")
				self.WriteCSV(user_original_params)
				return

		
		classifier._log_every_n_steps = used_params["log_every_n_steps"] # keeps an error from occuring

		# try: ##############################################
		if(not(best_params)):
			self.HelperPrinter("starting test: " + str(self.train_test_counts) + "/" + str(self.train_test_totals) )
		else:
			self.HelperPrinter("starting best params test: " + str(self.train_test_counts - self.train_test_totals + 1) )

		use_input_fn = self.test_input_fn(input_fn, used_params)


		# self.HelperPrinter("Retrieved test_input_fn(). Starting timer")

		start_stamp = time()
		# classifier.train( input_fn=input_fn, steps=self.num_train_steps)
		if(steps==None):
			classifier.train( input_fn=use_input_fn, steps=self.num_train_steps)
		else:
			classifier.train( input_fn=use_input_fn, steps=steps)
		elapsed_time = time() - start_stamp

		# ["batch_size", "eval_size", "predict_size","exec_time"]
		results = []
		used_params_orig = used_params
		used_params = use_input_fn.GetParams()
		for keyword in self.keywords:
			results.append( str(used_params[keyword]) )
		results.append(str(elapsed_time))
		if(best_params):
			results.append("BEST PARAMS TEST")


		self.WriteCSV(results)
		self.train_results.append(results)
		self.train_params_results[elapsed_time] = used_params_orig

		# if(self.train_test_counts==0):
		# 	old_prefetch_tensor, new_prefetch_tensor = use_input_fn.GetDatasetUsed()
		# 	self.HelperPrinter(" old_prefetch_tensor = " + str(old_prefetch_tensor) )
		# 	self.HelperPrinter(" new_prefetch_tensor = " + str(new_prefetch_tensor) )
		# 	self.HelperPrinter(" old == new = " + str(old_prefetch_tensor == new_prefetch_tensor))
		# 	self.HelperPrinter( " type(use_input_fn.user_prefetch_buffer_size) =" + str(type(use_input_fn.GetDatasetUsed())) )
		# 	self.HelperPrinter( " use_input_fn.user_prefetch_buffer_size =" + str((use_input_fn.GetDatasetUsed())) )
		# 	with tf.Session() as sess:
		# 		self.HelperPrinter(  " use_input_fn.GetDatasetUsed().eval() = " + str(use_input_fn.GetDatasetUsed().eval()) )

		# self.HelperPrinter( " NUM PREFETCHES COUNTED = " + str(use_input_fn.prefetch_modified) )


		self.train_test_counts += 1

		# except: ##############################################
		# 	self.HelperPrinter("Failed to run test: " + str(self.train_test_counts))
		# 	results = []
		# 	for keyword in self.keywords:
		# 		results.append( str(used_params[keyword]) )
		# 	results.append(str("ERROR: FAILED TEST"))
		# 	self.WriteCSV(results)
		# 	self.train_test_counts += 1

	def TestingDatapipeline(self, test_baseline=True):
		auto_gen_params_ = self.GetTrainParams()
		iterator_list = []
		auto_gen_params_list = []

		auto_gen_params = self.GetAutoParams()
		auto_gen_params["batch_size"] = self.user_train_batch_size
		auto_gen_params["User_Default_Test"] = "THIS IS THE USER\'S TEST "
		use_input_fn = self.test_input_fn( self.user_input_fn, auto_gen_params, user_test=True)

		num_tests = self.num_pipeline_tests

		# # # Make the user baseline tests
		# for i in range(1):
		# 	auto_gen_params = self.GetAutoParams() # user_original_params # 
		# 	auto_gen_params["batch_size"] = self.user_train_batch_size
		# 	auto_gen_params["User_Default_Test"] = "THIS IS THE USER\'S TEST "
		# 	use_input_fn = self.test_input_fn( self.user_input_fn, auto_gen_params, user_test=True)

		# 	for i in range(num_tests):
		# 		auto_gen_params_list.append( use_input_fn.GetParams() )
		# 	tmp_dataset = use_input_fn.GetModifiedDataset(auto_gen_params)
		# 	iterator = tmp_dataset.make_one_shot_iterator().get_next()
		# 	for i in range(num_tests):
		# 		iterator_list.append(iterator)

		# # # Make tests parameters
		for auto_gen_params in auto_gen_params_:
			auto_gen_params["batch_size"] = self.user_train_batch_size
			use_input_fn = self.test_input_fn( self.user_input_fn, auto_gen_params)
			for i in range(num_tests):
				auto_gen_params_list.append( use_input_fn.GetParams() )
			tmp_dataset = use_input_fn.GetModifiedDataset(auto_gen_params)
			# iterator = tmp_dataset.make_one_shot_iterator().get_next()
			iterator = tmp_dataset.make_initializable_iterator()
			for i in range(num_tests):
				iterator_list.append(iterator)


		# Make the user baseline tests
		num_tests = 20
		if(test_baseline):
			auto_gen_params = self.GetAutoParams() # user_original_params # 
			auto_gen_params["batch_size"] = self.user_train_batch_size
			auto_gen_params["User_Default_Test"] = "THIS IS THE USER\'S TEST "
			use_input_fn = self.test_input_fn( self.user_input_fn, auto_gen_params, user_test=True)
			for i in range(num_tests):
				auto_gen_params_list.append( use_input_fn.GetParams() )
			tmp_dataset = use_input_fn.GetModifiedDataset(auto_gen_params)
			# iterator = tmp_dataset.make_one_shot_iterator().get_next()
			iterator = tmp_dataset.make_initializable_iterator()
			for i in range(num_tests):
				iterator_list.append(iterator)




		min_exec_time = None
		min_exec_params = None

		iterator_list_len = len(iterator_list)
		# self.HelperPrinter(" ACTUAL TEST LENGTH: " + str(len(iterator_list)))
		# iterator_list = iterator_list[80:]
		# auto_gen_params_list = auto_gen_params_list[80:]
		# self.HelperPrinter(" NEW TEST LENGTH: " + str(len(iterator_list)))
		# return

		for i in range(iterator_list_len):
			iterator = iterator_list[i]
			next_element = iterator.get_next()
			used_params = auto_gen_params_list[i]


			with tf.Session() as sess:

				# line = " PARAMS " + str(used_params)
				# self.HelperPrinter(line, 1)
				try:				
					try:
						sess.run(tf.tables_initializer())
					except:
						pass
					sess.run(iterator.initializer)

					start_stamp = time()
					# sess.run(iterator)
					# retx, rety = sess.run(next_element)
					ret = sess.run(next_element)
					elapsed_time = time() - start_stamp
					mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

					if((min_exec_time == None) or (min_exec_time > elapsed_time)):
						min_exec_time = elapsed_time
						min_exec_params = auto_gen_params_list[i]

					results = []

					for keyword in self.keywords:
						results.append( str(used_params[keyword]) )
					results.append(str(elapsed_time))
					results.append(str(mem_usage))

					if "User_Default_Test" in used_params:
						results.append( str(used_params["User_Default_Test"]) )


					self.WriteCSV(results)
					line = "RAN TEST: " + str(i) +"/"+ str(iterator_list_len) + " TIME: " + str(elapsed_time) + " MEM: " + str(mem_usage)
					self.HelperPrinter(line)
					# line = "\t\tOUTPUT " + str(ret)
					# self.HelperPrinter(line)
					# line = "\t\t X(FEATURES): " + str(len(retx)) + " Y(LABELS): " + str(len(rety))
					# self.HelperPrinter(line)
					# quit()

					self.train_params_results[elapsed_time] = used_params

				except:
					self.HelperPrinter(" FAILED ON: " + str(used_params))
					# quit()
			# tf.reset_default_graph()
		#self.train_params_results = min_exec_params

		# self.HelperPrinter("MIN["+str(min_exec_time)+"]: " + str(self.train_params_results))

	def TestingDatapipeline2(self, test_baseline=True):
		pass
		# (TODO) Make a function to retrieve the best value for each param
		testing_start_time = time()

		def GetBestParamFromTests(param_list, param_name):
			if(self.TestingEarlyStop):
				return None
			best_params = 'False'
			param_times = []
			num_tests = self.num_pipeline_tests

			if(param_list == []):
				return None


			for param in param_list:
				ret_params = self.GetAutoParams(**{param_name : param, 'auto_model_fn': self.train_test_model_fn})
				ret_params["batch_size"] = self.user_train_batch_size

				# # Create Test Pipeline
				use_input_fn = self.test_input_fn( self.user_input_fn, ret_params)
				tmp_dataset = use_input_fn.GetModifiedDataset(ret_params)
				used_params = use_input_fn.GetParams()

				iterator = tmp_dataset.make_initializable_iterator()
				next_element = iterator.get_next()
				times = []
				test_count = 0


				line = "STARTING "+param_name+" TESTING: " + str(param_list.index(param)) + "/" + str(len(param_list))
				self.HelperPrinter( line , 1)

				# # Run Test Pipeline
				with tf.Session() as sess:
					for i in range(num_tests):
						try:
							try:
								sess.run(tf.tables_initializer())
							except:
								pass
							sess.run(iterator.initializer)
							start_stamp = time()
							ret = sess.run(next_element)
							elapsed_time = time() - start_stamp
							mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
							# Avoid the first run as it has cost of starting up
							if(test_count != 0): 
								times.append(elapsed_time)

							line = "RAN TEST: " + str(test_count) +"/"+ str(num_tests) + " TIME: " + str(elapsed_time) + " MEM: " + str(mem_usage)
							test_count += 1
							self.HelperPrinter(line)

							# Write out to CSV
							results = []
							for keyword in self.keywords:
								results.append( str(used_params[keyword]) )
							results.append(str(elapsed_time))
							results.append(str(mem_usage))
							self.WriteCSV(results)
						except:
							line = "FAILED TEST: " + str(test_count) +"/"+ str(num_tests)
							self.HelperPrinter(line, 1)
							# self.HelperPrinter(" FAILED ON: " + str(used_params))
							times.append(None)
							test_count += 1

					times = [_ for _ in times if _ != None]
					if(len(times) != 0):
						avg_time = sum(times)/len(times)
					else:
						avg_time = sys.maxint
					param_times.append(  avg_time )

			self.HelperPrinter(" "+param_name+": " + str(param_list) , 1)
			self.HelperPrinter(" TIMES: " + str(param_times) , 1)
			min_time = min(param_times)
			min_index = param_times.index(min_time)
			best_params = param_list[min_index]
			return best_params


		best_vectorize_map = GetBestParamFromTests(self.auto_params["vectorize_map"], 'auto_vectorize_map') 
		best_prefetch = GetBestParamFromTests(self.auto_params["prefetch_buffer_size"], 'auto_prefetch_buffer_size') 
		best_map_parallel = GetBestParamFromTests(self.auto_params["map_parallel"], 'auto_map_parallel') 
		best_map_and_batch = GetBestParamFromTests(self.auto_params["map_and_batch"], 'auto_map_and_batch') 

		# (TODO) Need more fixing to ensure safe run
		# best_change_dtypes = GetBestParam(self.auto_params["change_dtypes"], 'auto_change_dtypes') 
		# best_batch_dataset = GetBestParam(self.auto_params["batch_dataset"], 'auto_batch_dataset') 



		# self.HelperPrinter("best_prefetch: " + str(best_prefetch))
		# self.HelperPrinter("best_map_parallel: " + str(best_map_parallel))
		# self.HelperPrinter("best_map_and_batch: " + str(best_map_and_batch))

		elapsed_testing_time = time() - testing_start_time
		self.HelperPrinter("TestingDatapipeline2 TESTIMG TIME: " + str(elapsed_testing_time), 1)

		# Storing the single best found
		best_params = self.GetAutoParams( 
			auto_model_fn=self.train_test_model_fn,
			# auto_vectorize_map=best_vectorize_map,
			auto_prefetch_buffer_size=best_prefetch,
			auto_map_parallel=best_map_parallel,
			auto_map_and_batch=best_map_and_batch
		)
		self.train_params_results[-1] = best_params

		# Store Results on CSV as the smallest value
		self.HelperPrinter("TestingDatapipeline2: Storing Best Param in " + self.auto_tests_csv_path, 1)
		results = []
		for keyword in self.keywords:
				results.append( str(best_params[keyword]) )
		results.append(str(-1)) # elapsed_time
		results.append(str(-1)) # mem_usage
		results.append("COMBINATION OF ALL INDIVIDUALY TESTED PARAMETERS") # user test
		self.WriteCSV(results)


	def TestingDatapipeline3(self, test_baseline=True):
		"""
		Hill Climbing search for the best parameters 
		a.k.a. parameters that take the least amount of time
		"""
		start_TestingDatapipeline3 = time()

		params = {}
		params["batch_size"] = self.user_train_batch_size
		params = self.GetQuickParams(params)
		tmp_dataset = self.user_input_fn(params)
		tmp_dataset_list = GetDatasetList(tmp_dataset)

		# Starting State
		prefetch_sizes = GetPrefetchSizes(tmp_dataset_list)
		map_sizes = GetMapSizes(tmp_dataset_list)
		map_and_batch_sizes = GetMapAndBatchSizes(tmp_dataset_list)		


		# Limit 
		# Range: Autotune/-1  --- 1 --- CPU_COUTN
		CPU_COUNT = multiprocessing.cpu_count()

		def Neighbors(currentParams):
			# self.HelperPrinter("===========Neighbors===================")
			ret_lis = []

			paramOptions = {}

			for paramName in currentParams:
				paramVal = currentParams[paramName]
				if(paramVal == None):
					pass
				elif(len(paramVal) == 0):
					pass

				# elif( 'False' in str(paramVal) ):
				# 	self.HelperPrinter("PARAM["+paramName+"]: " + str(paramVal), 3 )
				# 	valueRange = ['True']
				# 	self.HelperPrinter("\t valRange " + str(valueRange) )
				# 	paramOptions[paramName] = [ valueRange ]
				# elif('True' in str(paramVal) ):
				# 	self.HelperPrinter("PARAM["+paramName+"]: " + str(paramVal), 3 )
				# 	valueRange = ['False']
				# 	self.HelperPrinter("\t valRange " + str(valueRange) )
				# 	paramOptions[paramName] = [ valueRange ]

				elif( 'True' in str(paramVal) or 'False' in str(paramVal)):
					self.HelperPrinter("PARAM["+paramName+"]: " + str(paramVal), 3 )					
					paramOptions[paramName] = [ paramVal ]


				else:
					self.HelperPrinter("PARAM["+paramName+"]: " + str(paramVal), 3 )
					
					valueRange = []
					for n in paramVal:
						newRange = []
						if(isinstance(n,int)):
							range_list = range(n-1, n+2)
							for n_ in range_list:
								if(n_ <= 0):
									newRange.append('Autotune')
								elif(n_ <= CPU_COUNT):
									newRange.append(n_)
							newRange = list(set(newRange))
						else:
							newRange.append(n)
						valueRange.append(newRange)
					self.HelperPrinter("\t valRange " + str(valueRange) )

					productRange = [list(n) for n in itertools.product(*valueRange) if list(n) != paramVal]
					self.HelperPrinter("\t productRange " + str(productRange) )
					
					paramOptions[paramName] = productRange


			# for paramName in paramOptions:
			# 	self.HelperPrinter("paramOptions["+paramName+"]: " + str(len(paramOptions[paramName])) , 3 )

			paramNeighbors = list(dict(zip(paramOptions.keys(), values)) for values in itertools.product(*paramOptions.values()))
			# self.HelperPrinter("NumNeighbors: " + str(len(paramNeighbors)) , 3 )
			# self.HelperPrinter("NumNeighbor[0]: " + str((paramNeighbors[0])) , 3 )
			# self.HelperPrinter("======================================")
			return paramNeighbors

		def EvalParam(currentParams, baseline_test=False):
			# self.HelperPrinter("===========EvalParam===================")

			ret_params = copy.copy(currentParams)
			ret_params['auto_model_fn'] = self.train_test_model_fn
			ret_params = self.GetAutoParams(**ret_params)
			ret_params["batch_size"] = self.user_train_batch_size
			# self.HelperPrinter("params: " + str(ret_params) )

			# # Create Test Pipeline
			use_input_fn = self.test_input_fn( self.user_input_fn, ret_params)
			tmp_dataset = use_input_fn.GetModifiedDataset(ret_params)
			used_params = use_input_fn.GetParams()

			iterator = tmp_dataset.make_initializable_iterator()
			next_element = iterator.get_next()
			times = []
			test_count = 0
			num_tests = self.num_pipeline_tests


			with tf.Session() as sess:
				for i in range(num_tests):
					try:
						try:
							sess.run(tf.tables_initializer())
						except:
							pass
						sess.run(iterator.initializer)
						start_stamp = time()
						ret = sess.run(next_element)
						elapsed_time = time() - start_stamp
						mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
						
						# Avoid the first run as it can contain cost of starting up pipeline
						if(test_count != 0): 
							times.append(elapsed_time)

						line = "\t RAN TEST: " + str(test_count) +"/"+ str(num_tests) + " TIME: " + str(elapsed_time) + " MEM: " + str(mem_usage)
						test_count += 1
						self.HelperPrinter(line)

						# # Write out to CSV
						results = []
						for keyword in self.keywords:
							results.append( str(used_params[keyword]) )
						results.append(str(elapsed_time))
						results.append(str(mem_usage))
						if(baseline_test):
							results.append("THIS IS THE USER\'S TEST ")
						self.WriteCSV(results)
						
					except:
						line = "FAILED TEST: " + str(test_count) +"/"+ str(num_tests)
						self.HelperPrinter(line, 3)
						self.HelperPrinter(" FAILED ON: " + str(used_params), 3)
						# times.append(None)
						test_count += 1



			# self.HelperPrinter("======================================")
			if(len(times) == 0):
				return float('inf')

			avg_time = sum(times) / len(times)
			return avg_time


		# (DONE) Search Space with bounds
		# When Conversion is reached
		# self.TestingEarlyStop

		currentParams = {
			'auto_vectorize_map': ['False'],
			'auto_prefetch_buffer_size': prefetch_sizes if prefetch_sizes != [] else None ,
			'auto_map_parallel': map_sizes if map_sizes != [] else None,
			'auto_map_and_batch': map_and_batch_sizes if map_and_batch_sizes != [] else None ,
		}
		
		auto_vectorize_map_false = copy.copy( currentParams )
		auto_vectorize_map_true =  copy.copy(  currentParams )
		auto_vectorize_map_true['auto_vectorize_map']  = ['True']

		auto_vectorize_map_false_time = EvalParam(auto_vectorize_map_false)
		auto_vectorize_map_true_time = EvalParam(auto_vectorize_map_true)
		if( auto_vectorize_map_true_time < auto_vectorize_map_false_time):
			currentParams['auto_vectorize_map'] = ['True']


		currentAvgTime = EvalParam(currentParams, baseline_test=True)

		neighborSearchCount = 0
		while(True):

			self.HelperPrinter("Neighbor Search Iteration["+str(neighborSearchCount)+"]")

			nextAvgTime = float('inf')
			nextParams = None

			neighborParams = Neighbors(currentParams)
			totalNeighbors = len(neighborParams)
			for i, nextNeighbor in enumerate(neighborParams):
				if( self.TestingEarlyStop ):
					if( nextAvgTime >=  currentAvgTime):
						break
					currentParams = nextParams
					nextAvgTime = nextAvgTime
					break

				self.HelperPrinter("Neighbor " + str(i) + "/" + str(totalNeighbors))
				nextNeighborAvgTime = EvalParam(nextNeighbor)

				if(nextNeighborAvgTime < nextAvgTime):
					nextAvgTime = nextNeighborAvgTime
					nextParams = nextNeighbor
			
			if( nextAvgTime >=  currentAvgTime):
				# No neighbor with smaller time
				break

			currentParams = nextParams
			nextAvgTime = nextAvgTime

			neighborSearchCount += 1



		self.train_params_results[-1] = self.GetAutoParams(**currentParams)

		self.HelperPrinter("BEST PARAM: " + str(currentParams), 3)

		results = []
		currentParams = self.train_params_results[-1]
		for keyword in self.keywords:
				results.append( str(currentParams[keyword]) )
		results.append(str(-1)) # elapsed_time
		results.append(str(-1)) # mem_usage
		results.append("BEST PARAM FOUND") # user test
		self.WriteCSV(results)

		dur_TestingDatapipeline3 = time() - start_TestingDatapipeline3
		self.HelperPrinter("TIME OF TestingDatapipeline3: " + str(dur_TestingDatapipeline3), 3)


	def TestingDatapipeline4(self, test_baseline=True):
		"""
		Hill Climbing search for the worst parameters 
		a.k.a. parameters that take the least amount of time
		"""
		start_TestingDatapipeline3 = time()

		params = {}
		params["batch_size"] = self.user_train_batch_size
		params = self.GetQuickParams(params)
		tmp_dataset = self.user_input_fn(params)
		tmp_dataset_list = GetDatasetList(tmp_dataset)

		# Starting State
		prefetch_sizes = GetPrefetchSizes(tmp_dataset_list)
		map_sizes = GetMapSizes(tmp_dataset_list)
		map_and_batch_sizes = GetMapAndBatchSizes(tmp_dataset_list)		


		# Limit 
		# Range: Autotune/-1  --- 1 --- CPU_COUTN
		CPU_COUNT = multiprocessing.cpu_count()

		def Neighbors(currentParams):
			# self.HelperPrinter("===========Neighbors===================")
			ret_lis = []

			paramOptions = {}

			for paramName in currentParams:
				paramVal = currentParams[paramName]
				if(paramVal == None):
					pass
				elif(len(paramVal) == 0):
					pass

				# elif( 'False' in str(paramVal) ):
				# 	self.HelperPrinter("PARAM["+paramName+"]: " + str(paramVal), 3 )
				# 	valueRange = ['True']
				# 	self.HelperPrinter("\t valRange " + str(valueRange) )
				# 	paramOptions[paramName] = [ valueRange ]
				# elif('True' in str(paramVal) ):
				# 	self.HelperPrinter("PARAM["+paramName+"]: " + str(paramVal), 3 )
				# 	valueRange = ['False']
				# 	self.HelperPrinter("\t valRange " + str(valueRange) )
				# 	paramOptions[paramName] = [ valueRange ]

				elif( 'True' in str(paramVal) or 'False' in str(paramVal)):
					self.HelperPrinter("PARAM["+paramName+"]: " + str(paramVal), 3 )					
					paramOptions[paramName] = [ paramVal ]


				else:
					self.HelperPrinter("PARAM["+paramName+"]: " + str(paramVal), 3 )
					
					valueRange = []
					for n in paramVal:
						newRange = []
						if(isinstance(n,int)):
							range_list = range(n-1, n+2)
							for n_ in range_list:
								if(n_ <= 0):
									newRange.append('Autotune')
								elif(n_ <= CPU_COUNT):
									newRange.append(n_)
							newRange = list(set(newRange))
						else:
							newRange.append(n)
						valueRange.append(newRange)
					self.HelperPrinter("\t valRange " + str(valueRange) )

					productRange = [list(n) for n in itertools.product(*valueRange) if list(n) != paramVal]
					self.HelperPrinter("\t productRange " + str(productRange) )
					
					paramOptions[paramName] = productRange


			# for paramName in paramOptions:
			# 	self.HelperPrinter("paramOptions["+paramName+"]: " + str(len(paramOptions[paramName])) , 3 )

			paramNeighbors = list(dict(zip(paramOptions.keys(), values)) for values in itertools.product(*paramOptions.values()))
			# self.HelperPrinter("NumNeighbors: " + str(len(paramNeighbors)) , 3 )
			# self.HelperPrinter("NumNeighbor[0]: " + str((paramNeighbors[0])) , 3 )
			# self.HelperPrinter("======================================")
			return paramNeighbors

		def EvalParam(currentParams, baseline_test=False):
			# self.HelperPrinter("===========EvalParam===================")

			ret_params = copy.copy(currentParams)
			ret_params['auto_model_fn'] = self.train_test_model_fn
			ret_params = self.GetAutoParams(**ret_params)
			ret_params["batch_size"] = self.user_train_batch_size
			# self.HelperPrinter("params: " + str(ret_params) )

			# # Create Test Pipeline
			use_input_fn = self.test_input_fn( self.user_input_fn, ret_params)
			tmp_dataset = use_input_fn.GetModifiedDataset(ret_params)
			used_params = use_input_fn.GetParams()

			iterator = tmp_dataset.make_initializable_iterator()
			next_element = iterator.get_next()
			times = []
			test_count = 0
			num_tests = self.num_pipeline_tests


			with tf.Session() as sess:
				for i in range(num_tests):
					try:
						try:
							sess.run(tf.tables_initializer())
						except:
							pass
						sess.run(iterator.initializer)
						start_stamp = time()
						ret = sess.run(next_element)
						elapsed_time = time() - start_stamp
						mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
						
						# Avoid the first run as it can contain cost of starting up pipeline
						if(test_count != 0): 
							times.append(elapsed_time)

						line = "\t RAN TEST: " + str(test_count) +"/"+ str(num_tests) + " TIME: " + str(elapsed_time) + " MEM: " + str(mem_usage)
						test_count += 1
						self.HelperPrinter(line)

						# # Write out to CSV
						results = []
						for keyword in self.keywords:
							results.append( str(used_params[keyword]) )
						results.append(str(elapsed_time))
						results.append(str(mem_usage))
						if(baseline_test):
							results.append("THIS IS THE USER\'S TEST ")
						self.WriteCSV(results)
						
					except:
						line = "FAILED TEST: " + str(test_count) +"/"+ str(num_tests)
						self.HelperPrinter(line, 3)
						self.HelperPrinter(" FAILED ON: " + str(used_params), 3)
						# times.append(None)
						test_count += 1



			# self.HelperPrinter("======================================")
			if(len(times) == 0):
				return float('inf')

			avg_time = sum(times) / len(times)
			return avg_time


		# (DONE) Search Space with bounds
		# When Conversion is reached
		# self.TestingEarlyStop

		currentParams = {
			'auto_vectorize_map': ['False'],
			'auto_prefetch_buffer_size': prefetch_sizes if prefetch_sizes != [] else None ,
			'auto_map_parallel': map_sizes if map_sizes != [] else None,
			'auto_map_and_batch': map_and_batch_sizes if map_and_batch_sizes != [] else None ,
		}
		
		auto_vectorize_map_false = copy.copy( currentParams )
		auto_vectorize_map_true =  copy.copy(  currentParams )
		auto_vectorize_map_true['auto_vectorize_map']  = ['True']

		auto_vectorize_map_false_time = EvalParam(auto_vectorize_map_false)
		auto_vectorize_map_true_time = EvalParam(auto_vectorize_map_true)
		if( auto_vectorize_map_true_time > auto_vectorize_map_false_time):
			currentParams['auto_vectorize_map'] = ['True']


		currentAvgTime = EvalParam(currentParams, baseline_test=True)

		neighborSearchCount = 0
		while(True):

			self.HelperPrinter("Neighbor Search Iteration["+str(neighborSearchCount)+"]")

			nextAvgTime = float('-inf')
			nextParams = None

			neighborParams = Neighbors(currentParams)
			totalNeighbors = len(neighborParams)
			for i, nextNeighbor in enumerate(neighborParams):
				if( self.TestingEarlyStop ):
					if( nextAvgTime <  currentAvgTime):
						break
					currentParams = nextParams
					nextAvgTime = nextAvgTime
					break

				self.HelperPrinter("Neighbor " + str(i) + "/" + str(totalNeighbors))
				nextNeighborAvgTime = EvalParam(nextNeighbor)

				if(nextNeighborAvgTime > nextAvgTime):
					nextAvgTime = nextNeighborAvgTime
					nextParams = nextNeighbor
			
			if( nextAvgTime <  currentAvgTime):
				# No neighbor with larger time
				break

			currentParams = nextParams
			nextAvgTime = nextAvgTime

			neighborSearchCount += 1



		self.train_params_results[float('inf')] = self.GetAutoParams(**currentParams)

		self.HelperPrinter("WORST PARAM: " + str(currentParams), 3)

		results = []
		currentParams = self.train_params_results[float('inf')]
		for keyword in self.keywords:
				results.append( str(currentParams[keyword]) )
		results.append(str('inf')) # elapsed_time
		results.append(str('inf')) # mem_usage
		results.append("WORST PARAM FOUND") # user test
		self.WriteCSV(results)
		
		dur_TestingDatapipeline3 = time() - start_TestingDatapipeline3
		self.HelperPrinter("TIME OF TestingDatapipeline4: " + str(dur_TestingDatapipeline3), 3)






	def StoreData(self, file_path=None):
		"""
		Uploads data to Google Storage path provided for the model
		"""
		try:
			if(file_path==None):
				self.upload_blob(source_file_name=self.auto_tests_csv_path)
			else:
				self.upload_blob(source_file_name=file_path)
		except:
			pass

	def upload_blob(self, bucket_name=None, source_file_name=None, destination_blob_name=None):
		"""Uploads a file to the bucket."""
		# bucket_name = "your-bucket-name"
		# source_file_name = "local/path/to/file"
		# destination_blob_name = "storage-object-name"

		if(bucket_name==None):
			bucket_name=self.user_modeldir.split("/")[2]
		if(source_file_name==None):
			source_file_name=self.auto_tests_csv_path
		if(destination_blob_name==None):
			destination_blob_name=  os.path.join( (self.user_modeldir.split(bucket_name)[1])[1:] ,  self.auto_tests_csv_path )


		try:
			storage_client = storage.Client()
			bucket = storage_client.bucket(bucket_name)
			blob = bucket.blob(destination_blob_name)
			blob.upload_from_filename(source_file_name)
		except:
			self.HelperPrinter(" Faile to upload file")

	    # self.HelperPrinter( "File {} uploaded to {}.".format( source_file_name, destination_blob_name ) )

	def GetBestParams(self):
		min_params = None
		if(self.train_params_results == {}):
			# No testing was done
			min_params = self.GetAutoParams()
		else: 
			# Testing was done, find best results, keys are time or -1 for best establised
			param_key = min(self.train_params_results.keys())
			min_params = self.train_params_results[param_key]

		self.HelperPrinter("MIN EXEC PARAMS:", 1)
		for key in min_params:
			val = min_params[key]
			self.HelperPrinter( "\t" + str(key) + ": " + str(val) )
		return min_params

	def GetWorstParams(self):
		min_params = None
		if(self.train_params_results == {}):
			# No testing was done
			min_params = self.GetAutoParams()
		else: 
			# Testing was done, find best results, keys are time or -1 for best establised
			param_key = max(self.train_params_results.keys())
			min_params = self.train_params_results[param_key]

		self.HelperPrinter("MAX EXEC PARAMS:", 1)
		for key in min_params:
			val = min_params[key]
			self.HelperPrinter( "\t" + str(key) + ": " + str(val) )
		return min_params

	def GetBestFromCSV(self, filename=None):
		if(filename==None):
			filename = self.auto_tests_csv_path

		def ProcessValues(value):
			ret_value = None
			if(isinstance(value,np.ndarray)):
				value = list(value)

			# self.HelperPrinter("\tDAYM: " + str(value) , 1)
			if(isinstance(value[0], str) ):
				try:
					ret_value = eval(value[0]) # Turn back into values
					if(isinstance(ret_value, bool)):
						ret_value = str(ret_value)
				except:
					pass
					if(sum([1 if 'False' in str(_) else 0 for _ in value]) ):
						ret_value = 'False'
					elif(sum([1 if 'True' in str(_) else 0 for _ in value]) ):
						ret_value = 'True'
				# if('[' == value[0][0]): # Turn string of list back into list
				# 	ret_value = value[0].strip('][').split(', ') 
				# 	tmp_ret = []
				# 	for val in ret_value:
				# 		try:
				# 			tmp_ret.append( float(val) )
				# 		except:
				# 			tmp_ret.append( val.strip('\'') )
				# 	# ret_value = [int(_) for _ in ret_value]
				# 	ret_value = tmp_ret
				# else: # Turn int back into intager
				# 	try:
				# 		ret_value = float(value[0])
				# 	except: # Is string but not a value, so turn into a string
				# 		pass
				# 		ret_value = eval(value[0])
			else:
				ret_value = str(value) 
			return ret_value

		df = pd.read_csv(filename)
		exec_min = df['exec_time'].min()
		min_params = df[df["exec_time"] == exec_min]
		
		ret_params = {}

		self.HelperPrinter("MIN EXEC PARAMS: \n\n", 1)
		for key in min_params:
			val = min_params[key].values
			val = ProcessValues(val)
			val_type = type(val)
			self.HelperPrinter( "\t" + str(key) + ": " + str(val) )
			# For debugging purposes
			# self.HelperPrinter( "\t" + str(key) + ": " + str(val) + " oring: " + str(min_params[key].values) )
			# self.HelperPrinter( "\t" + str(key) + ": " + str(val) + " oring: " + str(min_params[key].values) + " type: " + str(val_type))
			ret_params[key] = val

		# self.HelperPrinter("MIN EXEC PARAMS: \n\n" + str(ret_params) +"\n\n", 1)
		# self.HelperPrinter("type(min_params): " + str(type(ret_params)))

		ret_params = self.GetQuickParams(ret_params)
		return ret_params

	def GetModifiedDataset(self, params):
		# if(os.path.exists(self.auto_tests_csv_path)):
		# 	self.HelperPrinter("FILE EXISTS", 3)
		# 	auto_params = self.GetBestFromCSV()
		# else:
		# 	auto_params = self.GetBestParams()
		auto_params = self.GetBestParams()
		ret = self.test_input_fn(ret_input_fn=self.user_input_fn, auto_params=auto_params)
		return ret(params)

	def GetWorstModifiedDataset(self, params):
		# if(os.path.exists(self.auto_tests_csv_path)):
		# 	self.HelperPrinter("FILE EXISTS", 3)
		# 	auto_params = self.GetBestFromCSV()
		# else:
		# 	auto_params = self.GetBestParams()
		auto_params = self.GetWorstParams()
		ret = self.test_input_fn(ret_input_fn=self.user_input_fn, auto_params=auto_params)
		return ret(params)


	def GetQuickParams(self, params):
		# Add the user's baseline parameters
		if(isinstance(self.user_params, dict)):
			params.update(self.user_params)
		elif(isinstance(self.user_params, tf.contrib.training.HParams)):
			params.update(self.user_params.values())

		# Merge both dictionary and attribute needs
		class params_class(dict):
			def __init__(self, dictionary):
				self.dictionary = dictionary
				dict.__init__(self, self.dictionary)
			def __getitem__(self, key):
				return self.dictionary[key]
			def __getattr__(self, key):
				return self.dictionary[key]
			def __setitem__(self, key, value):
				self.dictionary[key] = value
			def __str__(self):
				return str(self.dictionary)
			def __repr__(self):
				return self.dictionary
			# def __iter__(self):
			# 	return iter(self.dictionary.keys())

		params = params_class(params)
		return params

	def GetModelInputTypes(self):
		# params = self.user_params
		params = {}
		# params["batch_size"] = 8**4 # random number chosen. Does not matter, return is deleted
		params["batch_size"] = self.user_train_batch_size # random number chosen. Does not matter, return is deleted
		params = self.GetQuickParams(params)

		test_ret = self.user_input_fn(params)
		# self.HelperPrinter( "str(test_ret)= " + str(test_ret))
		# self.HelperPrinter( "str(type(test_ret))= " + str(type(test_ret)))
		ret = None
		if type(test_ret) == type(tuple()):
			ret = list(test_ret)
			for i in range(len(ret)):
				if not isinstance(ret[i], tf.DType):
					ret[i] = ret[i].dtype
			ret = tuple(ret)
		else:
			ret = test_ret.output_types
		# ret = test_ret.output_types if type(test_ret) != type(tuple()) else test_ret
		del test_ret
		del params
		return ret

	def PrintOriginalDataset(self):
			
		auto_params = self.GetAutoParams() # No modifications to the paramaters
		tmp_dataset = self.test_input_fn(ret_input_fn=self.user_input_fn, auto_params=auto_params)
		params = {"batch_size": self.user_train_batch_size }
		params = self.GetQuickParams(params)
		tmp_dataset = tmp_dataset(params)
		tmp_dataset = GetDatasetList(tmp_dataset)
		self.PrintDataList(tmp_dataset)
