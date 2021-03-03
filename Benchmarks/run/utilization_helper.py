import tensorflow as tf 
import os
import json
import csv

# gcs_file = ["TESTING/TEST1/TPUv2/BERT_MRPC/optimize/plugins/profile/2020-08-05_05:30:49/"]
# bucket_name = "abe_ucr_bucket3"

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

tf.flags.DEFINE_string("dir", None, "GCS bucket where profiles are. Written as 'gs://bucket_name'")
tf.flags.DEFINE_string("name", "test", "name for test")
FLAGS = tf.flags.FLAGS

def FindJSONKeyValue(key, data):
  ret = []
  if(isinstance(data,list)):
    for data_ in data:
      ret += FindJSONKeyValue(key, data_)
  elif(isinstance(data,dict)):
    for k, v in data.iteritems():
      if key in str(k):
        ret.append(v)
      else:
        if(isinstance(v,list) or isinstance(v,dict)):
          ret += FindJSONKeyValue(key, v)
  return ret


def FindOverviewFiles(bucket_name, gcs_file):
	blobList = []
	if(isinstance(gcs_file,list)):
		for i in range(len(gcs_file)):
			gcs_file_ = os.path.join( "gs://" , bucket_name, gcs_file[i] )
			file_list = tf.gfile.ListDirectory(gcs_file_)
			file_list = [file for file in file_list if "overview" in file]
			file_list = [os.path.join(gcs_file_ , file) for file in file_list]
			blobList += file_list
	else:
		gcs_file_ =  os.path.join( "gs://" , bucket_name, gcs_file )
		file_list = tf.gfile.ListDirectory(gcs_file_)
		file_list = [file for file in file_list if "overview" in file]
		file_list = [os.path.join(gcs_file_ , file) for file in file_list]
		blobList += file_list
	return blobList


def UtilizationHelper(dir_path, name):
	FLAGS_NAME = name 
	FLAGS_DIR = dir_path
	tf.flags.mark_flag_as_required("dir")

	# gs://bucket_name/path --> 'gs:','','bucket_name','path'
	bucket_name = str(FLAGS_DIR).split('/')[2]
	gcs_file = os.path.join( *str(FLAGS_DIR).split('/')[3:] )

	gcs_file_profiels = os.path.join( FLAGS_DIR, "plugins", "profile" )
	gcs_file_base = os.path.join( gcs_file, "plugins", "profile" )
	profile_list = tf.gfile.ListDirectory(gcs_file_profiels)
	profile_list = [os.path.join(gcs_file_base, profile) for profile in profile_list]

	blobList = FindOverviewFiles(bucket_name, profile_list)

	heder = [FLAGS_NAME]
	compute_percent_list = []
	infeed_percent_list = []
	flop_list = []
	memory_list = []

	for blob in blobList:
		file = tf.gfile.GFile(blob, 'rb')
		overview_data = json.load(file)
		
		compute_ms_average = FindJSONKeyValue("compute_ms_average", overview_data)
		compute_ms_average = float(compute_ms_average[0])
		
		infeed_ms_average = FindJSONKeyValue("infeed_ms_average", overview_data)
		infeed_ms_average = float(infeed_ms_average[0])
		
		steptime_ms_average = FindJSONKeyValue("steptime_ms_average", overview_data)
		steptime_ms_average = float(steptime_ms_average[0])

		flop_rate_utilization_relative_to_roofline = FindJSONKeyValue("flop_rate_utilization_relative_to_roofline", overview_data)
		flop_rate_utilization_relative_to_roofline = float( (flop_rate_utilization_relative_to_roofline[0]).replace('%','') )
		flop_rate_utilization_relative_to_roofline = flop_rate_utilization_relative_to_roofline / 100.0

		memory_bw_utilization_relative_to_hw_limit = FindJSONKeyValue("memory_bw_utilization_relative_to_hw_limit", overview_data)
		memory_bw_utilization_relative_to_hw_limit = float( (memory_bw_utilization_relative_to_hw_limit[0]).replace('%','') )
		memory_bw_utilization_relative_to_hw_limit = memory_bw_utilization_relative_to_hw_limit / 100.0

		file.close()
		print("SUCCESSFULLY to open: " + str(blob))
		print("\tcompute_ms_average: "+str(compute_ms_average))
		print("\tinfeed_ms_average: "+str(infeed_ms_average))
		print("\tsteptime_ms_average: "+str(steptime_ms_average))
		compute_percent = (compute_ms_average/steptime_ms_average) if steptime_ms_average > 0 else 0.0 
		infeed_percent = (infeed_ms_average/steptime_ms_average) if steptime_ms_average > 0 else 0.0
		print("\tcompute %: "+str( compute_percent ))
		print("\tinfeed %: "+str( infeed_percent ))
		print("\tflop_rate_utilization_relative_to_roofline %: "+str( flop_rate_utilization_relative_to_roofline ))
		print("\tmemory_bw_utilization_relative_to_hw_limit %: "+str( memory_bw_utilization_relative_to_hw_limit ))
		

		heder.append( os.path.basename(blob) )
		compute_percent_list.append(compute_percent)
		infeed_percent_list.append(infeed_percent)
		flop_list.append( flop_rate_utilization_relative_to_roofline )
		memory_list.append( memory_bw_utilization_relative_to_hw_limit )

	csv_file = open("TPUPoint_"+FLAGS_NAME+"_utilization.csv","a+")
	event_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
	event_writer.writerow( heder )
	event_writer.writerow( [FLAGS_NAME + "_compute%"] + compute_percent_list )
	event_writer.writerow( [FLAGS_NAME + "_infeed%"] + infeed_percent_list )
	event_writer.writerow( [FLAGS_NAME + "_flop%"] + flop_list )
	event_writer.writerow( [FLAGS_NAME + "_mem%"] + memory_list )

	DrawUtilization(compute_percent_list, infeed_percent_list, flop_list, memory_list, name=FLAGS_NAME, image_name="TPUPointUtilizationImage_"+FLAGS_NAME+".png")

def DrawUtilization(compute_percent_list, infeed_percent_list, flop_list, memory_list,name="", image_name="TPUPointUtilizationImage.png"):
	fig, ax = plt.subplots() 
	# x_vals = len(compute_percent_list)
	if(name != ""):
		name = name + " "
	plt.plot(compute_percent_list, 'o-', label=name+" compute %")
	plt.plot(infeed_percent_list, 'o-', label=name+" infeed %")
	plt.plot(flop_list, 'o-', label=name+" flops %")
	plt.plot(memory_list, 'o-', label=name+" memory %")
	plt.ylabel('Utilization %')
	plt.xlabel('Profiles')
	plt.title(name)
	plt.legend()
	# plt.ylim((0.0, 1.1))
	# plt.xlim((0.0, 1.0)) 
	plt.savefig(image_name)


if __name__ == '__main__':
	tf.flags.mark_flag_as_required("dir")

	# gs://bucket_name/path --> 'gs:','','bucket_name','path'
	bucket_name = str(FLAGS.dir).split('/')[2]
	gcs_file = os.path.join( *str(FLAGS.dir).split('/')[3:] )

	gcs_file_profiels = os.path.join( FLAGS.dir, "plugins", "profile" )
	gcs_file_base = os.path.join( gcs_file, "plugins", "profile" )
	profile_list = tf.gfile.ListDirectory(gcs_file_profiels)
	profile_list = [os.path.join(gcs_file_base, profile) for profile in profile_list]


	print(bucket_name)
	print(profile_list)


	blobList = FindOverviewFiles(bucket_name, profile_list)

	heder = [FLAGS.name]
	compute_percent_list = []
	infeed_percent_list = []
	flop_list = []
	memory_list = []

	for blob in blobList:
		file = tf.gfile.GFile(blob, 'rb')
		overview_data = json.load(file)
		
		compute_ms_average = FindJSONKeyValue("compute_ms_average", overview_data)
		compute_ms_average = float(compute_ms_average[0])
		
		infeed_ms_average = FindJSONKeyValue("infeed_ms_average", overview_data)
		infeed_ms_average = float(infeed_ms_average[0])
		
		steptime_ms_average = FindJSONKeyValue("steptime_ms_average", overview_data)
		steptime_ms_average = float(steptime_ms_average[0])

		flop_rate_utilization_relative_to_roofline = FindJSONKeyValue("flop_rate_utilization_relative_to_roofline", overview_data)
		flop_rate_utilization_relative_to_roofline = float( (flop_rate_utilization_relative_to_roofline[0]).replace('%','') )
		flop_rate_utilization_relative_to_roofline = flop_rate_utilization_relative_to_roofline / 100.0

		memory_bw_utilization_relative_to_hw_limit = FindJSONKeyValue("memory_bw_utilization_relative_to_hw_limit", overview_data)
		memory_bw_utilization_relative_to_hw_limit = float( (memory_bw_utilization_relative_to_hw_limit[0]).replace('%','') )
		memory_bw_utilization_relative_to_hw_limit = memory_bw_utilization_relative_to_hw_limit / 100.0

		file.close()
		print("SUCCESSFULLY to open: " + str(blob))
		print("\tcompute_ms_average: "+str(compute_ms_average))
		print("\tinfeed_ms_average: "+str(infeed_ms_average))
		print("\tsteptime_ms_average: "+str(steptime_ms_average))
		compute_percent = (compute_ms_average/steptime_ms_average) if steptime_ms_average > 0 else 0.0 
		infeed_percent = (infeed_ms_average/steptime_ms_average) if steptime_ms_average > 0 else 0.0
		print("\tcompute %: "+str( compute_percent ))
		print("\tinfeed %: "+str( infeed_percent ))
		print("\tflop_rate_utilization_relative_to_roofline %: "+str( flop_rate_utilization_relative_to_roofline ))
		print("\tmemory_bw_utilization_relative_to_hw_limit %: "+str( memory_bw_utilization_relative_to_hw_limit ))
		

		heder.append( os.path.basename(blob) )
		compute_percent_list.append(compute_percent)
		infeed_percent_list.append(infeed_percent)
		flop_list.append( flop_rate_utilization_relative_to_roofline )
		memory_list.append( memory_bw_utilization_relative_to_hw_limit )

	csv_file = open("utilization.csv","a+")
	event_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
	event_writer.writerow( heder )
	event_writer.writerow( [FLAGS.name + "_compute%"] + compute_percent_list )
	event_writer.writerow( [FLAGS.name + "_infeed%"] + infeed_percent_list )
	event_writer.writerow( [FLAGS.name + "_flop%"] + flop_list )
	event_writer.writerow( [FLAGS.name + "_mem%"] + memory_list )

	DrawUtilization(compute_percent_list, infeed_percent_list, flop_list, memory_list, name=FLAGS.name, image_name="TPUPointUtilizationImage.png")

