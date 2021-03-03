import tensorflow as tf 

def SimilarityHelper(dir_path, name):
	import tensorflow as tf
	from tensorflow.contrib.tpu import SummarizationClass
	test_func = tf.contrib.tpu.profiler.Summarization.main

	test_func(logdir=dir_path, output_prefix=name)