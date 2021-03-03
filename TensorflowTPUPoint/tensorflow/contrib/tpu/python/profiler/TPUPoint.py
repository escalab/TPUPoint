"""
TPUPoint
crated by ESCAL Lab at the University of California Riverside
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import subprocess
import sys
# import grpc
import json
import threading
import csv
if(sys.version_info.major == 2):
  import Queue as queue
else:
  import queue
# from werkzeug import Request
from distutils.version import LooseVersion
import tensorflow as tf
from time import gmtime, strftime, time, clock, sleep
from multiprocessing import Process, Queue
from threading import Thread
from google.cloud import storage  # pip install google-cloud-storage
from tensorflow.contrib.tpu import SummarizationClass
from tensorflow.contrib.tpu import AutoadjustClass


from tensorflow.core.profiler import profiler_analysis_pb2
from tensorflow.python.eager import profiler_client
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import errors


class GlobalStepHook(tf.estimator.SessionRunHook):
  def __init__(self):
    self._global_step_tensor = None
    self.value = None
  def begin(self):
    self._global_step_tensor = tf.train.get_global_step()
  def after_run(self, run_context, run_values):
    self.value = run_context.session.run(self._global_step_tensor)
  def __str__(self):
    return str(self.value)



class TPUPoint(object):
  def __init__(self,
    estimator=None, 
    gcp_project=None,
    tpu_zone=None,
    tpu=None,
    service_addr=None,
    logdir=None,
    workers_list=None,
    duration_ms= 30000,# 60000, # 60000 max. 2000 min
    Similarity=0.7,
    num_tracing_attempts=3,
    include_dataset_ops=True, # False for longer traces
    monitoring_level = 1, # 1 or 2 logging level # (TODO) Remove as monitoring is not a functionality
    num_queries = 4 ): #

    ### USER INFORMATION ###
    self.estimator = estimator #estimator
    self.new_estimator = None
    self.new_TPUConfig = None
    self.new_RunConfig = None

    self.gcp_project = gcp_project
    self.tpu_zone = tpu_zone
    self.tpu = tpu

    if(estimator != None):
      self.CreateNewEstimator(estimator._model_fn, estimator)

    self.service_addr = service_addr # a.k.a tpu_master
    self.master_tpu_unsecure_channel = None
    self.logdir = logdir
    self.workers_list = workers_list # a.k.a hostsnames = workers_list.split(",")
    self.duration_ms = duration_ms if duration_ms > 0 else 1000
    self.include_dataset_ops = include_dataset_ops
    self.num_tracing_attempts = num_tracing_attempts 
    self.monitoring_level = monitoring_level
    self.num_queries = num_queries
    self.queries_count = 0
    self.session_id = 0
    self.analysis_stub = None
    self.profiler_stub = None
    self.RUNNABLE = True
    self.STOPED = False
    self.STARTED = False
    self.bucket = None
    self.tf_version = tf.__version__
    # self.Check()
    self.EXECPATH =  os.path.dirname(os.path.abspath(__file__))
    self.SingleProfileFile = os.path.join(self.EXECPATH, 'SingleProfile.py')

    ### PROFILE THREADING ####
    self.profileDirList = []
    self.thread = threading.Thread(target=self.StartTracing)
    self.QUEUE_STOPED = False
    self.summarization_queue = queue.Queue()
    self.summarization_thread = threading.Thread(target=self.SummarizeSingleProfileWorker)

    ### OPTIMIZATION THREADING
    self.optimization_thread = None

    ### TIME MARKERS ###
    self.start_recording = None
    self.duration_recording = None
    self.ProfileTS = None
    self.ProfileDurations = []
    self.PhaseDurations = {}
    self.profile_count = 0
    self.phase_count = 0

    self.OverviewJSONName = 'TPUPoint_Overview.json'
    self.ProfileTimesName = 'TPUPoint_Profile_Times.csv'
    if(self.logdir != None):
      self.bucket_name = self.logdir.split("/")[2]
      self.bucket_dir = os.path.join( *self.logdir.split("/")[3:])
    else:
      self.bucket_name = None
      self.bucket_dir = None
    self.Similarity = Similarity
    self.autoadjustclass = None
    if(self.logdir != None):
      self.sumclass = SummarizationClass(logdir=self.logdir, 
        printing=True,
        file_name_suffix="",
        bucket_name=self.bucket_name, 
        Similarity=self.Similarity, 
        TotalExecTime=None)
      self.sumOutputFD = open(self.sumclass.outputfileName , 'w')
    else:
      self.sumclass = None
      self.sumOutputFD = None
    self.currentStep = None

    self.AttemptedProfiles = 0

    self.STOP_ESTIMATOR = False
    self.global_step = None
    self.phase_break = False
    self.pause_profiling = False
    self.analyzer = False

  ### HELPER FUNCTIONS ###

  def EPPrint(self, line, level=1):
    # class bcolors:
    #   HEADER = '\033[95m'
    #   OKBLUE = '\033[94m'
    #   OKGREEN = '\033[92m'
    #   WARNING = '\033[93m'
    #   FAIL = '\033[91m'
    #   ENDC = '\033[0m'
    #   BOLD = '\033[1m'
    #   UNDERLINE = '\033[4m'

    if level == 1: # Green
      tf.logging.info("\033[92m TPUPoint: \033[0m" + line)
    elif level == 2: # Yellow Warning 
      tf.logging.info("\033[93m TPUPoint: \033[0m" + line)
    elif level == 3: # Red Fail
      tf.logging.info("\033[91m TPUPoint: \033[0m" + line)
    else: # Green 
      tf.logging.info("\033[92m TPUPoint: \033[0m" + line)

  def CheckRunnable(self):
    self.Check()
    if self.RUNNABLE == False:
      return False
    return True

  def Check(self):
    def get_workers_list(cluster_resolver):
      JOB_NAME = 'worker'
      if cluster_resolver == None:
        EPPrint("cluster_resolver is None", 3)
      cluster_spec = cluster_resolver.cluster_spec()
      if cluster_spec == None:
        EPPrint("cluster_spec is None", 3)
      task_indices = cluster_spec.task_indices(JOB_NAME)
      if task_indices == None:
        EPPrint("task_indices is None", 3)
      workers_list = [
          cluster_spec.task_address(JOB_NAME, i).split(':')[0] for i in task_indices
      ]
      return ','.join(workers_list)


    if self.service_addr is None and self.tpu is None:
      self.EPPrint(' You must specify either --service_addr or --tpu.', 3)
      # sys.exit('ESCALProfiler: You must specify either --service_addr or --tpu.')
      self.RUNNABLE = False
      return
    tpu_cluster_resolver = None
    if self.service_addr is not None:
      if self.tpu is not None:
        # tf.logging.warn('Both --service_addr and --tpu are set. Ignoring --tpu and using --service_addr.', 2)
        self.EPPrint('Both --service_addr and --tpu are set. Ignoring --tpu and using --service_addr.', 2)
    else:
      tpu_cluster_resolver = (
          tf.contrib.cluster_resolver.TPUClusterResolver(
              [self.tpu], zone=self.tpu_zone, project=self.gcp_project))
      self.service_addr = tpu_cluster_resolver.get_master()
    self.service_addr = self.service_addr.replace('grpc://', '').replace(':8470', ':8466')

    # (TODO) Might need "dns:///" in front of service_addr for channel but need to test

    workers_list = ''
    if LooseVersion(self.tf_version) < LooseVersion('1.9'):
      # tf.logging.warn('Attempt to profile with legacy support under TensorFlow version %s' % self.tf_version)
      self.EPPrint( ('Attempt to profile with legacy support under TensorFlow version %s' % self.tf_version), 2)
    else:
      if self.workers_list is not None:
        self.workers_list = self.workers_list
      elif tpu_cluster_resolver is not None:
        self.workers_list = get_workers_list(tpu_cluster_resolver)

    if not self.logdir and not self.monitoring_level:
      # print('ESCALProfiler: logdir must be provided.')
      self.EPPrint('logdir must be provided.',3)
      # sys.exit('ESCALProfiler: logdir must be provided.')
      self.RUNNABLE = False
      return
    elif self.logdir.startswith('gs://') != True:
      # print('ESCALProfiler: logdir must be on google cloud storage, not locally. i.e. gs://bucket_name')
      self.EPPrint('logdir must be on google cloud storage, not locally. i.e. gs://bucket_name', 3)
      self.RUNNABLE = False
      return

    # Check that the GCP bucket exist
    if self.bucket == None:
      # (TODO) storage bucket check not currently checking for the existence of the bucket
      # from google.cloud import storage
      # client = storage.Client()

      logdir_split = self.logdir.split("/")
      # bucket_name = logdir_split[2]
      
      # bucket =  client.get_bucket(bucket_name)

      destination_blob_name = logdir_split[3:-1]
      destination_blob_name.append("plugins")
      destination_blob_name.append("profile")

      # path = ""
      # for dest in destination_blob_name:
      #   path =  os.path.join(path, dest)
      #   path += "/"

      #   blob = bucket.blob(path)
      #   blob.upload_from_string('', content_type='application/x-www-form-urlencoded;charset=UTF-8')
        

      # except:
      #   self.RUNNABLE = False
      #   self.EPPrint(" The provided GCP bucket in logdir feild does not exist or cannot be created", 3)
      #   return


    # self.cmd is only to run a different process so execute the request i.e. the cpp tool
    # self.cmd = [executable_path]
    if self.logdir is not None:
      self.logdir = os.path.expandvars(os.path.expanduser(self.logdir))
    # self.cmd.append('--logdir=' + self.logdir)
    # self.cmd.append('--service_addr=' + self.service_addr)
    # self.cmd.append('--workers_list=' + self.workers_list)
    # self.cmd.append('--duration_ms=' + str(self.duration_ms))
    # self.cmd.append('--num_tracing_attempts=' + str(1)) # str(self.num_tracing_attempts))
    # self.cmd.append('--include_dataset_ops=' + str(self.include_dataset_ops).lower())
    # self.cmd.append('--monitoring_level=' + str(0)) # str(self.monitoring_level))
    # self.cmd.append('--num_queries=' + str(self.num_queries))

    self.master_tpu_unsecure_channel = self.service_addr.replace(':8466', '')
    if self.analysis_stub is None:
      # Workaround the grpc's 4MB message limitation.
      gigabyte = 1024 * 1024 * 1024
      options = [('grpc.max_message_length', gigabyte),
                 ('grpc.max_send_message_length', gigabyte),
                 ('grpc.max_receive_message_length', gigabyte)]
      tpu_profiler_port = self.master_tpu_unsecure_channel + ':8466' # port for profiler tool
      # channel = grpc.insecure_channel(tpu_profiler_port, options)

      # # the gRPC stud that will make requests
      # if self.analysis_stub == None:
      #   self.analysis_stub = tpu_profiler_analysis_pb2_grpc.TPUProfileAnalysisStub(channel) 
      # if self.profiler_stub == None:
      #   self.profiler_stub = tpu_profiler_pb2_grpc.TPUProfilerStub(channel)

    
    self.service_addr = str(self.service_addr)
    self.logdir = str(self.logdir)
    self.workers_list = [str(worker) for worker in self.workers_list.split(",")][0]
    self.include_dataset_ops = bool(self.include_dataset_ops)
    self.num_tracing_attempts = 1 # self.num_tracing_attempts

  def GetSingleValFromTensorInt(self, tensor):
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
    return int( possible_ret[0][1] )

  def CleanUp(self):
    """
    CleanUp() 
    uploads any file in the cwd with 'TPUPoint' in its name to the logdir specified
    and removes them from the cwd
    """
    if(self.bucket_name == None):
      return

    cwd = os.getcwd()
    stored_files = []
    for r,d,f in os.walk(cwd):
      for file in f:
        if("TPUPoint" in file):
          stored_files.append( os.path.join(r,file) )

    bucket_name = self.bucket_name
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    for stored_file in stored_files:
      stored_file_dir = stored_file
      stored_file_basename = os.path.basename(stored_file)

      destination_blob_name  = os.path.join( self.bucket_dir , 'TPUPoint', stored_file_basename )
      blob = bucket.blob(destination_blob_name)
      blob.upload_from_filename(stored_file_dir)

      os.remove( stored_file_dir )


  ### OVERVIEW JSON FUNCTIONS ####

  def ProfileOverview(self, ts, dur, name="Unnamed Profile", args={}):
    if(self.ProfileTS == None):
      self.ProfileTS = self.start_recording
      end = ts + dur
      dur = end - self.ProfileTS
    elif(self.duration_recording != None):
      overall_end = self.start_recording + self.duration_recording 
      end = min( (ts + dur) , overall_end)
      dur = end - self.ProfileTS
    else:
      end = ts + dur
      dur = end - self.ProfileTS
    profile = {
        "name": name,
        "pid": -1,
        "tid": 1,
        "ph": "X",
        "ts": self.ProfileTS,
        "dur": dur,
        "args": args
      }
    self.ProfileDurations.append(profile)
    self.ProfileTS = end

  def PhaseOverview(self, ts, dur, name="Unnamed Phase", count=None, args={}):
    if(count == None):
      count = self.phase_count

    profile = {
        "name": name,
        "pid": -1,
        "tid": 2,
        "ph": "X",
        "ts": ts,
        "dur": dur,
        "args": args
      }
    # self.PhaseDurations.append(profile)
    self.PhaseDurations[count] = profile

  def CreateOverviewPhases(self, PhaseBreakMarkerRatios):
    tpupoint_start  = self.start_recording
    tpupoint_dur = self.duration_recording
    tpupoint_end = tpupoint_start + tpupoint_dur

    phase_break_lis = []
    for profile_key in PhaseBreakMarkerRatios:
      profile_dur = self.ProfileDurations[profile_key]['dur']
      profile_start = self.ProfileDurations[profile_key]['ts']
      for phase_break_ratio in PhaseBreakMarkerRatios[profile_key]:
        phase_break_lis.append( profile_start + (profile_dur*phase_break_ratio) )

    phase_break_lis = [tpupoint_start] + phase_break_lis + [tpupoint_end]

    for i in range(len(phase_break_lis) - 1):
      ts = phase_break_lis[i]
      end = phase_break_lis[i+1]
      dur = end - ts
      name = "Phase("+str(i)+")"
      self.PhaseOverview(ts=ts, dur=dur, name=name, count=i)

  def CreatePhaseTimesCSV(self):
    file_name = self.ProfileTimesName
    ProfileTimesCSV = csv.writer(open(file_name, "a+"), delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)

    header = ['index','name','start', 'end', 'dur', 'paths']
    ProfileTimesCSV.writerow(header)

    for i, profile in enumerate(self.ProfileDurations):
      number = i
      name = profile['name']
      start = profile['ts']
      dur =  profile['dur']
      end = start + dur
      paths = profile['args']['Profiles']
      row = [number, name, start, end, dur, paths]
      ProfileTimesCSV.writerow(row)


  def WriteSummarizeOverview(self):
    meta = [
      {"ph": "M",
      "args": {"name": "TPUPoint"},
      "pid": -1,
      "name": "process_name"},
      
      {"tid": 0,
       "ph": "M",
       "args": {"name": "TPUPoint"},
       "pid": -1,
       "name": "thread_name"},
      
      {"tid": 1,
       "ph": "M",
       "args": {"name": "Profile Breakdown"},
       "pid": -1,
       "name": "thread_name"},
      
      {"tid": 2,
       "ph": "M",
       "args": {"name": "Phase Breakdown"},
       "pid": -1,
       "name": "thread_name"}]

    TPUPointProfiling = []
    
    OverallDur = 0.0
    OverallHost = 0.0
    OverallTPU = 0.0

    # Making sure the last profile represents the last area of profiled duration
    if(len(self.ProfileDurations) != 0):
      lastProfileKey = len(self.ProfileDurations)-1
      overallEnd = self.start_recording + self.duration_recording 
      lastProfileEnd = overallEnd - self.ProfileDurations[lastProfileKey]['ts']
      self.ProfileDurations[lastProfileKey]['dur'] = lastProfileEnd

    
    self.CreateOverviewPhases(self.sumclass.PhaseBreakMarkerRatios)

    self.CreatePhaseTimesCSV()

    TPUPointProfiling += [{
        "name": "TPUPoint" ,
        "pid": -1,
        "tid": 0,
        "ph": "X",
        "ts": self.start_recording,
        "dur": self.duration_recording,
      }]

    traceEvents = meta + TPUPointProfiling + self.ProfileDurations + self.PhaseDurations.values()

    data = {
      "displayTimeUnit":"ns",
      "metadata":{"highres-ticks": True},
      "traceEvents": traceEvents}

    json.dump(data, open(self.OverviewJSONName,'w'))

  def SummarizeClose(self):
    self.sumclass.WriteOutputJSON()
    self.sumclass.CSVClass.WritePhase()
    self.sumclass.CSVClass.WriteOverview()
    # self.sumclass.DrawImage()

  def DrawOverviewImage(self):
    # PhaseToPolygonCoordinateList = tf.contrib.tpu.profiler.Summarization.PhaseToPolygonCoordinateList
    def PhaseToPolygonCoordinateList(phase_list, similarity=0.0):
      ret_lis = []
      total_ts = min([phase['ts'] for phase in phase_list])
      total_dur = sum([phase['dur'] for phase in phase_list])


      for i, phase in enumerate(phase_list):
        phase_start = phase['ts']
        phase_end = phase['ts'] + phase['dur']

        start_x = (phase_start - total_ts)  / total_dur
        end_x = (phase_end - total_ts) / total_dur

        # bl, tl, tr, br
        bl = (start_x, similarity)
        tl = (start_x, similarity+0.08)
        tr = (end_x, similarity+0.08)
        br = (end_x, similarity)
        phase_coordinates = [bl, tl, tr, br]
        ret_lis.append(phase_coordinates)

      return ret_lis
    color_gen = tf.contrib.tpu.profiler.Summarization.color_gen
    DrawPhases = tf.contrib.tpu.profiler.Summarization.DrawPhases

    poly_coor_list = PhaseToPolygonCoordinateList(phase_list=self.PhaseDurations.values(), similarity=self.Similarity)
    DrawPhases(patches_list=poly_coor_list, graph_title="Phase_Overview", ylabel="Similarity", xlabel="Time")


  ### PROFILE & SUMMARIZATION FUNCTIONS ###

  def FindJSONKeyValue(self, key, data):
    ret = []
    if(isinstance(data,list)):
      for data_ in data:
        ret += self.FindJSONKeyValue(key, data_)
    elif(isinstance(data,dict)):
      for k, v in data.iteritems():
        if key in str(k):
          ret.append(v)
        else:
          if(isinstance(v,list) or isinstance(v,dict)):
            ret += self.FindJSONKeyValue(key, v)
    return ret

  def FindOverviewFiles(self, bucket_name, gcs_file):
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

  def SummarizeSingleProfileWorker(self):

    def ThreadSummarizeSingleProfile( gcs_file, profile_number, profile_index):
      self.EPPrint("STARTED SummarizaeSingleProfile("+str(profile_number)+")", 2)
      bucket_name = self.bucket_name 

      trace_files_list = self.sumclass.FindTraceFiles(
        bucket_name=bucket_name, 
        gcs_file=gcs_file
      )
      for trace_file in trace_files_list:
        self.EPPrint("\t TraceFile: " + str(trace_file), 2)

      for file_name in trace_files_list:
        self.sumclass.current_step, phase_break = self.sumclass.Analysis(
          file_name=file_name, 
          current_step=self.sumclass.current_step, 
          profile_number=profile_index )
        profile_index += 1
        # if true operational phase break from sumclass, 
        # no previous phase_breaks
        # and optimize_input_fn are all done, then request training to stop
        optimize_input_fn_done = True if (self.autoadjustclass == None) else (-1 in self.autoadjustclass.train_params_results)
        # optimize_input_fn_done = True
        if((self.sumclass.PhasesBreak) and (self.phase_break==False) and optimize_input_fn_done):
          # Stop estimator.train() in train_dynamic()
          self.phase_break = True
          self.STOP_ESTIMATOR = True 
          self.EPPrint("\t\t REQUESTING ESTIMATOR TRAINING TO STOP " , 3)

      self.EPPrint("FINISHED SummarizaeSingleProfile("+str(profile_number)+")", 2)
      return profile_index


    profile_index = 0
    while(not self.QUEUE_STOPED or not self.summarization_queue.empty()) :
      if(not self.summarization_queue.empty()):
        task = self.summarization_queue.get()
        gcs_file, profile_number = task
        # try:
        profile_index = ThreadSummarizeSingleProfile(gcs_file, profile_number, profile_index)
        # ThreadSummarizeSingleProfileUtilization(gcs_file, profile_number)
        # except:
        #   self.EPPrint("FAILED Summarization("+str(profile_number)+")")
        #   pass
        self.summarization_queue.task_done()
      else:
        pass

  def GetRecentProfilesRecorded(self):
    profileDirList = []


    logdir = self.logdir
    bucket_name = self.bucket_name

    prefix = logdir.split(bucket_name)[1]
    prefix = os.path.join(prefix, 'plugins', 'profile/')
    if( prefix[0] == '/' ):
      prefix = prefix[1:]

    depth = prefix.count('/')


    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_or_name=bucket_name)
    for blob in blobs:
      bname = str(blob.name)
      if( (prefix in bname) and ( bname[-1] == '/')  and (bname.count('/') > depth) ):
        if(bname not in self.profileDirList):
          profileDirList.append(bname)

    self.profileDirList += profileDirList
    return profileDirList

  def SingleProfile(self):
    if(self.pause_profiling == True):
      return False
    self.AttemptedProfiles += 1

    cmd = ['python', self.SingleProfileFile]
    cmd.append("--service_addr="+str(self.service_addr))
    cmd.append("--logdir="+str(self.logdir))
    cmd.append("--duration_ms="+str(self.duration_ms))
    cmd.append("--workers_list="+str(self.workers_list))
    cmd.append("--include_dataset_ops="+str(self.include_dataset_ops))
    cmd.append("--num_tracing_attempts=1")
    cmd = " ".join(cmd)


    start_recording = time() * 1e6
    returnCode = subprocess.call(cmd, shell=True) # Blocking subprocess


    # Get the list of the profiles recorded so far
    ProfileList = self.GetRecentProfilesRecorded()

    duration_recording = (time()*1e6) - start_recording
    args = {'Checkpoint': str(tf.train.latest_checkpoint(self.logdir)) }
    args['Profiles'] = ProfileList
    self.ProfileOverview(ts=start_recording, dur=duration_recording, name="Profile("+str(self.profile_count)+")", args=args)
    self.EPPrint("CREATED SingleProfile("+str(self.profile_count)+")", 2)
    self.profile_count += 1


    # Put the recent profile in the queue for the Summarization Thread to execute
    self.summarization_queue.put( (ProfileList, self.profile_count-1) )


    ####################################
    # Stoping Training from thread call
    # n = 40
    # if(self.AttemptedProfiles):
    #   self.EPPrint("\t AttemptedProfiles:" + str(self.AttemptedProfiles), 2) 
    # if((self.AttemptedProfiles == n) and (self.STOP_ESTIMATOR == False)):
    #   self.EPPrint("\t STOP_ESTIMATOR", 3)
    #   self.STOP_ESTIMATOR = True
    # sleep(2)
    ####################################

    return False # Non Empty Trace

  def StartTracing(self):
    hostnames = self.workers_list.split(",")
    empty_trace = False
    while(self.STOPED == False):
      self.SingleProfile()



  ### EXECUTION OF TPUPOINT & ESTIMATOR TRAINING
  
  def should_stop_fn(self):

    return self.STOP_ESTIMATOR


  def optimize_input_fn(self, input_fn, blocking=False, worst=False):
    """
    optimize_input_fn()
    Test different parameters to frind the optimal modifications to the input_fn
    """

    # # Apply optimization and then run stright though
    auto = AutoadjustClass(classifier=self.estimator, 
                        model_fn=self.estimator._model_fn if self.estimator != None else None , 
                        input_fn=input_fn, 
                        valrange=20, # 4
                        num_train_steps=1,
                        num_pipeline_tests=6, # 20
                        printing=True, # blocking
                        csvresults=True
                        )
    self.autoadjustclass = auto
    self.EPPrint(" Created auto class")
    self.autoadjustclass.PrintOriginalDataset()
    # self.autoadjustclass.TestingDatapipeline()
    self.autoadjustclass.PrintOriginalSizes(input_fn)
    
    optimize_fn = None
    if(worst == False):
      # optimize_fn = self.autoadjustclass.TestingDatapipeline2
      optimize_fn = self.autoadjustclass.TestingDatapipeline3
    else:
      optimize_fn = self.autoadjustclass.TestingDatapipeline4

    if(blocking):
      optimize_fn()
    else:
      pass
      # self.optimization_thread = threading.Thread(target=self.autoadjustclass.TestingDatapipeline2)
      # self.optimization_thread = threading.Thread(target=self.autoadjustclass.TestingDatapipeline3)
      self.optimization_thread = threading.Thread(target=optimize_fn )
      self.optimization_thread.start()


  def GetModifiedDataset(self):
    if(self.autoadjustclass == None):
      return None
    return self.autoadjustclass.GetModifiedDataset

  def train(self, estimator, input_fn, hooks=None, steps=None, max_steps=None, saving_listeners=None):
    """
    train()
    an alternative to TPUEstimator.train()
    """

    if(estimator == None):
      raise ValueError(
        'Cannot run .train() when TPUPoint initalized without estimator argument'
        'To use TPUPoint.train() initalize with TPUPoint(estimator=TPUEstimator(),...)'
      )


    # early_stop_hook = tf.estimator.experimental.make_early_stopping_hook(estimator=self.estimator, should_stop_fn=self.should_stop_fn, run_every_secs=5, run_every_steps=None)
    global_step_hook = GlobalStepHook()

    if(isinstance(hooks,list)):
      # hooks.append(early_stop_hook)
      hooks.append(global_step_hook)
    elif(hooks == None):
      hooks = [] 
      # hooks.append(early_stop_hook)
      hooks.append(global_step_hook)
    else:
      self.EPPrint('Error. hooks of type ' + str(type(hooks)), 3)
      raise ValueError(
        'TPUPoint.trian() expected hooks of type list.'
        'Got {} instead.'.format(str(type(hooks)))  
      )

    # #############

    min_params = self.autoadjustclass.GetBestParams()
    auto_input_fn = self.autoadjustclass.test_input_fn(ret_input_fn=input_fn, auto_params=min_params)

    ret = estimator.train(input_fn=auto_input_fn, hooks=hooks, steps=steps, max_steps=max_steps, saving_listeners=saving_listeners)
    return ret

  def train_naive(self, estimator, input_fn, hooks=None, steps=None, max_steps=None, saving_listeners=None):

    if(estimator == None):
      raise ValueError(
        'Cannot run .train() when TPUPoint initalized without estimator argument'
        'To use TPUPoint.train() initalize with TPUPoint(estimator=TPUEstimator(),...)'
      )


    # early_stop_hook = tf.estimator.experimental.make_early_stopping_hook(estimator=self.estimator, should_stop_fn=self.should_stop_fn, run_every_secs=5, run_every_steps=None)
    global_step_hook = GlobalStepHook()

    if(isinstance(hooks,list)):
      # hooks.append(early_stop_hook)
      hooks.append(global_step_hook)
    elif(hooks == None):
      hooks = [] 
      # hooks.append(early_stop_hook)
      hooks.append(global_step_hook)
    else:
      self.EPPrint('Error. hooks of type ' + str(type(hooks)), 3)
      raise ValueError(
        'TPUPoint.trian() expected hooks of type list.'
        'Got {} instead.'.format(str(type(hooks)))  
      )

    # #############

    min_params = self.autoadjustclass.GetWorstParams()
    auto_input_fn = self.autoadjustclass.test_input_fn(ret_input_fn=input_fn, auto_params=min_params)

    ret = estimator.train(input_fn=auto_input_fn, hooks=hooks, steps=steps, max_steps=max_steps, saving_listeners=saving_listeners)
    return ret

  def CreateNewEstimator(self, model_fn, estimator, warm_start_from=None):
    # Forcing more updates of host step causing more checks to see if TPUPoint requested a stop
    iterations_per_loop = '30s' #  str(int(duration_ms * 1e-3)) + 's'
    new_TPUConfig = tf.contrib.tpu.TPUConfig(
      iterations_per_loop = iterations_per_loop, # estimator._config._tpu_config.iterations_per_loop,
      num_shards = estimator._config._tpu_config.num_shards,
      num_cores_per_replica = estimator._config._tpu_config.num_cores_per_replica,
      per_host_input_for_training = estimator._config._tpu_config.per_host_input_for_training,
      tpu_job_name = estimator._config._tpu_config.tpu_job_name,
      initial_infeed_sleep_secs = estimator._config._tpu_config.initial_infeed_sleep_secs,
      input_partition_dims = estimator._config._tpu_config.input_partition_dims,
      eval_training_input_configuration = estimator._config._tpu_config.eval_training_input_configuration,
      experimental_host_call_every_n_steps = estimator._config._tpu_config.experimental_host_call_every_n_steps
    )

    # Making sure no checkpoints are deleted
    keep_checkpoint_max = None
    # self.estimator._config._keep_checkpoint_max = None 
    new_cluster = tf.contrib.cluster_resolver.TPUClusterResolver([self.tpu], zone=self.tpu_zone, project=self.gcp_project)

    new_RunConfig = tf.contrib.tpu.RunConfig(
      tpu_config = new_TPUConfig , # estimator._config.tpu_config ,
      evaluation_master = new_cluster, # estimator._config.evaluation_master ,
      cluster = estimator._config.cluster ,
      master = estimator._config.master if(not estimator._config.cluster) else None ,

      device_fn = estimator._config.device_fn ,
      eval_distribute = estimator._config.eval_distribute ,
      experimental_max_worker_delay_secs = estimator._config.experimental_max_worker_delay_secs ,
      keep_checkpoint_every_n_hours = estimator._config.keep_checkpoint_every_n_hours ,
      keep_checkpoint_max = keep_checkpoint_max, # estimator._config.keep_checkpoint_max ,
      log_step_count_steps = estimator._config.log_step_count_steps ,
      model_dir = estimator._config.model_dir ,
      protocol = estimator._config.protocol ,
      save_checkpoints_secs = estimator._config.save_checkpoints_secs ,
      save_checkpoints_steps = estimator._config.save_checkpoints_steps ,
      save_summary_steps = estimator._config.save_summary_steps ,
      session_config = estimator._config.session_config if(not estimator._config.cluster) else None ,
      session_creation_timeout_secs = estimator._config.session_creation_timeout_secs ,
      tf_random_seed = estimator._config.tf_random_seed ,
      train_distribute = estimator._config.train_distribute ,
      experimental_distribute= estimator._config._experimental_distribute,
    )
    if(not callable(model_fn)):
      model_fn = estimator._model_fn
    new_Estimator = tf.contrib.tpu.TPUEstimator(
      model_fn = model_fn, # estimator._model_fn ,
      model_dir = estimator.model_dir ,
      config = new_RunConfig, # estimator.config ,
      params = estimator.params ,
      use_tpu = estimator._ctx._use_tpu,
      train_batch_size = estimator._ctx._train_batch_size,
      eval_batch_size = estimator._ctx._eval_batch_size,
      predict_batch_size = estimator._ctx._predict_batch_size,
      # batch_axis = estimator.,
      eval_on_tpu = estimator._ctx._eval_on_tpu,
      export_to_tpu = estimator._export_to_tpu,
      export_to_cpu = estimator._export_to_cpu,
      warm_start_from = warm_start_from, # estimator._ctx,
      embedding_config_spec = estimator._ctx._embedding_config_spec,
      export_saved_model_api_version = estimator._export_saved_model_api_version,
    )

    self.new_estimator = new_Estimator
    self.new_TPUConfig = new_TPUConfig
    self.new_RunConfig = new_RunConfig

  def train_dynamic(self, model_fn, estimator, input_fn, hooks=None, steps=None, max_steps=None, saving_listeners=None, warm_start_from=None):
    # def train_dynamic(self, estimator, input_fn, hooks=None, steps=None, max_steps=None, saving_listeners=None):
    """
    dynamicly applying optimization to the train used in TPUEstimator.train()
    """

    self.CreateNewEstimator(model_fn, estimator, warm_start_from=warm_start_from)

    early_stop_hook = tf.estimator.experimental.make_early_stopping_hook(estimator=self.new_estimator, should_stop_fn=self.should_stop_fn, run_every_secs=5, run_every_steps=None)
    global_step_hook = GlobalStepHook()

    if(isinstance(hooks,list)):
      hooks.append(early_stop_hook)
      hooks.append(global_step_hook)
    elif(hooks == None):
      hooks = [] 
      hooks.append(early_stop_hook)
      hooks.append(global_step_hook)
    else:
      self.EPPrint('Error. hooks of type ' + str(type(hooks)), 3)
      raise ValueError(
        'TPUPoint.train_dynamic() expected hooks of type list.'
        'Got {} instead.'.format(str(type(hooks)))  
      )

    # min_params = self.autoadjustclass.GetBestParams()
    # auto_input_fn = self.autoadjustclass.test_input_fn(ret_input_fn=input_fn, auto_params=min_params)

    # ret = estimator.train(input_fn=auto_input_fn, hooks=hooks, steps=steps, max_steps=max_steps, saving_listeners=saving_listeners)
    self.Start()
    dynamic_train_start = time()
    ret = self.new_estimator.train(input_fn=input_fn, hooks=hooks, steps=steps, max_steps=max_steps, saving_listeners=saving_listeners)
    dynamic_paused_ts = time()
    self.EPPrint("TRAINING BEFORE OPTIMIZATION APPLIED TIME: " + str(dynamic_paused_ts - dynamic_train_start) , 3)
    self.Pause()

    # """
    current_steps = global_step_hook.value if(global_step_hook.value != None) else 0
    total_steps = max_steps if(max_steps != None) else steps

    steps = total_steps - current_steps
    self.EPPrint("\t\t Remaining Steps = " + str(steps), 3)
    # self.Stop()
    # return ret

    # while( current_steps < total_steps ):
    if(steps > 0):
      # apply optimization
      self.EPPrint("APPLY OPTIMIZATION", 3)
      if(not( -1 in self.autoadjustclass.train_params_results.keys() )):
        self.autoadjustclass.TestingDatapipeline3()
      min_params = self.autoadjustclass.GetBestParams()
      auto_input_fn = self.autoadjustclass.test_input_fn(ret_input_fn=input_fn, auto_params=min_params)
      # # turn off make_early_stopping_hook
      self.STOP_ESTIMATOR = False
      # train for the remaining duration
      if(max_steps == None):
        steps = total_steps - current_steps
      current_steps = global_step_hook.value
      dynamic_cont_ts = time()
      self.EPPrint("OPTIMIZATION APPLIED TIME: " + str(dynamic_cont_ts - dynamic_paused_ts) , 3)
      self.Start()
      self.new_estimator.train(input_fn=auto_input_fn, hooks=hooks, steps=steps, saving_listeners=saving_listeners)

    try:
      self.EPPrint("TRAINING AFTER OPTIMIZATION APPLIED TIME: " + str(time() - dynamic_cont_ts) , 3)
    except:
      dynamic_cont_ts = time()
      self.EPPrint("OPTIMIZATION APPLIED TIME: " + str(dynamic_cont_ts - dynamic_paused_ts) , 3)
      self.EPPrint("TRAINING AFTER OPTIMIZATION APPLIED TIME: " + str(time() - dynamic_cont_ts) , 3)
    self.Stop()
    #"""
    return ret

  def Start(self, analyzer=True):
    self.EPPrint("\tSTART TPUPoint")
    self.pause_profiling = False
    self.analyzer = analyzer
    if(self.CheckRunnable() == False):
      self.EPPrint("Did not pass the check", 3)
      return False
    
    if(not self.thread.isAlive()): # Check if already alive. Prevents multiple starts
      self.start_recording = time() * 1e6
      _ = self.GetRecentProfilesRecorded() # Updating local list of what records already exist
      self.thread.start() # starts the StartTracing thread
    
    if(not self.summarization_thread.isAlive()):
      self.summarization_thread.start() # start the Summarization thread

    return True

  def Stop(self):
    self.EPPrint("\tSTOP TPUPoint")


    self.EPPrint("Closing out helper threads", 2)
    
    # Join the main profile thread
    self.EPPrint("Closing profile threads 1/3", 2)
    self.STOPED = True
    if(self.thread.isAlive()):
      self.thread.join()

    self.duration_recording = (time()* 1e6) - self.start_recording

    # Joining optimization thread if still alive
    self.EPPrint("Closing optimization threads 2/3", 2)
    if(self.optimization_thread != None):
      if(self.optimization_thread.isAlive()):
        self.autoadjustclass.TestingEarlyStop = True
        self.optimization_thread.join()


    # Joining the summarization thread when summarization queue is done
    self.QUEUE_STOPED = True
    self.EPPrint("Closing summarization threads 3/3", 2)
    if(self.summarization_thread.isAlive()):
      self.summarization_thread.join()

    if(self.CheckRunnable() == False):
      return

    if(self.analyzer):
      # # Wrining out any of the last phase
      self.EPPrint("Writing out Phases", 2)
      self.SummarizeClose() # must go before WriteSummarizeOverview() to collect phase marking poitns

      # # Write the Overview of TPUPoint's time, the Profiling duraitons, and (TODO) Phase durations
      self.EPPrint("Writing out Overview", 2)
      self.WriteSummarizeOverview()

      self.EPPrint("Creating overview image", 2)
      self.DrawOverviewImage()

      print_line = str(len(self.ProfileDurations)) + "/" + str(self.AttemptedProfiles) + " successful profiles"
      self.EPPrint(print_line, 2)
      print_line = " Number of Phases: " + str(len(self.PhaseDurations.values()))
      self.EPPrint(print_line, 2)
      total_phase_time = sum([phase['dur'] for phase in self.PhaseDurations.values()])
      for phase_name in self.PhaseDurations:
        phase = self.PhaseDurations[phase_name]
        dur = phase['dur']
        percent = dur / total_phase_time
        print_line = "\t Phases("+str(phase_name)+")%: " + str(percent)
        self.EPPrint(print_line, 2)


    # self.STOPED = False

    self.EPPrint("Finished closing", 2)
    return True

  def Pause(self):
    self.EPPrint("\tPAUSE TPUPoint")
    self.pause_profiling = True






