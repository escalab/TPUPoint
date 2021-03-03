"""
TPUPoint Optimization Example
crated by ESCAL Lab at the University of California Riverside
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import app
from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import
import time
import os
import sys

import tensorflow as tf
from tensorflow.contrib.tpu import TPUPoint



def mapped_function(*args):
    # Do some pre-processing
    tf.py_func(lambda: time.sleep(0.03), [], ())
    return args

def ArtificialDataset(num_samples=300):
	def generator(num_samples):
		# Opening the file
		time.sleep(0.03)
		for sample_idx in range(num_samples):
			# Reading data (line, record) from the file
			time.sleep(0.015)
			yield (sample_idx,)
	ret = tf.data.Dataset.from_generator(
		generator,
		output_types=tf.int64,
		output_shapes=(1,),
		args=(num_samples,)
	)
	return ret

def ArtificialPipelineBaseline(params):
	batch_size = params["batch_size"]
	dataset = ArtificialDataset()
	dataset = dataset.map(mapped_function)
	dataset = dataset.batch(batch_size)
	dataset = dataset.prefetch(3)
	dataset = dataset.repeat()
	return dataset


def ArtificialPipelineTFOptions(params):
	batch_size = params["batch_size"]
	dataset = ArtificialDataset()
	dataset = dataset.map(mapped_function)
	dataset = dataset.batch(batch_size)
	dataset = dataset.prefetch(3)
	dataset = dataset.repeat()
	return dataset

	options = tf.data.Options()
	options.experimental_optimization.apply_default_optimizations = True
	dataset = dataset.with_options(options)

	return dataset


def benchmarkPipeline(dataset, num_epochs=1, num_steps_per_epoch=10, batch_size=8):
	params = {"batch_size":batch_size}
	tmp_dataset = dataset(params)

	iterator = tmp_dataset.make_initializable_iterator()
	next_element = iterator.get_next()
	with tf.Session() as sess:
		sess.run(iterator.initializer)

		start_stamp = time.time()
		for epoch_num in range(num_epochs):
			print("\tEpoch: " + str(epoch_num))
			for step in range(num_steps_per_epoch):
				ret = sess.run(next_element)
				print("\t\tStep: " + str(step) )
				# time.sleep(0.01) # preforme training step

		elapsed_time = time.time() - start_stamp
	line = "Execution time: " + str(elapsed_time)
	print(line)
	return elapsed_time

def main():
	tpupoint = TPUPoint()
	
	baseline_fn = ArtificialPipelineBaseline
	tensorflow_fn = ArtificialPipelineTFOptions

	tpupoint.optimize_input_fn(baseline_fn, blocking=True)
	tpupoint_fn = tpupoint.GetModifiedDataset()

	time1 = benchmarkPipeline(baseline_fn)
	time2 = benchmarkPipeline(tpupoint_fn)
	time3 = benchmarkPipeline(tensorflow_fn)

	print("baseline_time: " + str(time1))
	print("tpupoint_time: " + str(time2))
	print("tensorflow_time: " + str(time3))
	print("\n\n")
	print("TPUPoint Speed up: " + str(time1/time2) )
	print("TF Options Speed up: " + str(time1/time3) )




if __name__ == '__main__':
	main()