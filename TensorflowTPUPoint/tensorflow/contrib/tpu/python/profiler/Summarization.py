"""
SUMMARIZATION
crated by ESCAL Lab at the University of California Riverside
"""
import json
import os
import gzip
import shutil
import sys
import multiprocessing
import time
import re
import resource
import copy
# from sympy import Interval, Union, Intersection
import pandas as pd

import concurrent.futures
import itertools
import csv
import tensorflow as tf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
# os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '2'
os.dup2(sys.stdout.fileno(), 1)
os.dup2(sys.stdout.fileno(), 2)
tf.logging.set_verbosity(tf.logging.INFO)

import zlib
# import cProfile
import threading

IGNORE_OPS = ['RunStep', 'Xprof', 'TPU ', 'XLA', 'TPUCompile']

TOP_TPU_OPS = ['reshape', 'sum', 'outfeed', 'biasaddgrad', 'fusion', 'infeeddequeuetuple', 'mul', 'infeed', 'copy', 'all-reduce', 'outfeedenqueuetuple']

TOP_CPU_OPS = ['concatv2', 'transferbuffertoinfeedlocked', 'rungraph', 'delinearizex32', 'send', 'buildpaddedoutput', 'infeedenqueuetuple', 'linearizex32', 'outfeeddequeuetuple']


class StepClass(object):

	def __init__(self):
		self.stepEvents = []
		self.stepStartTime = float('inf') 
		self.stepEndTime = 0.0
		self.stepName = None

		self.opEvents = []
		self.opNames = []
		self.opStartTime = float('inf') 
		self.opEndTime = 0.0

		self.HostTID_events = {}
		self.TPUTID_events = {}

		self.HostInfo = {}
		self.TPUInfo = {}

		self.IGNORE_OPS = IGNORE_OPS


	def Duration(self):
		stepDur = self.stepEndTime - self.stepStartTime
		opDur = self.opEndTime - self.opStartTime
		return max( 0.0, stepDur, opDur)

	def OpNames(self):
		op_name_set = self.opNames
		op_name_set = [op_name for op_name in op_name_set if(self.CheckIgnoreOp(op_name) == False)]
		return set(op_name_set)
		# return set(self.opNames)

	def StepName(self):

		return self.stepName

	def AddStep(self, step):
		self.stepEvents.append(step)
		if(step['ts'] < self.stepStartTime):
			self.stepStartTime = step['ts']
		end_time = step['ts'] + step['dur']
		if(end_time > self.stepEndTime):
			self.stepEndTime = end_time
		if(self.stepName == None):
			self.stepName = step['name']

	def AddOp(self, op):
		self.opEvents.append(op)
		if(op['ts'] < self.opStartTime):
			self.opStartTime = op['ts']
		end_time = op['ts'] + op['dur']
		if(end_time > self.opEndTime):
			self.opEndTime = end_time
		if(op['name'] not in self.opNames):
			self.opNames.append( op['name'] )

	def GetStartTime(self):
		# if(self.opStartTime == float('inf') ):
		# 	return self.stepStartTime
		return min(self.opStartTime , self.stepStartTime)

	def GetEndTime(self):
		# if(self.opEndTime == 0.0 ):
		# 	return self.stepEndTime
		return max(self.opEndTime, self.stepEndTime)

	def GetDurTime(self):

		return self.GetEndTime() - self.GetStartTime()

	def NumOps(self):

		return len(self.opEvents)

	def NumSteps(self):

		return len(self.stepEvents)

	def CheckIgnoreOp(self, opname):
		for ignore_op in self.IGNORE_OPS:
			if(ignore_op.lower() in opname.lower()):
				return True
		return False

	def SeperateByPIDTID(self, HostPID, HostTID, TpuPID, TpuTID):
		
		# 2D list, HostTID*EVENT_LIST
		# for TID in HostTID:
		# 	if(TID not in self.HostTID_events): 
		# 		self.HostTID_events[TID] = []
		for TID in HostTID:
			self.HostTID_events[TID] = self.HostTID_events.get(TID, [])

		# 3D list, TPUPID*TPUTID*EVENT_LIST
		for PID in TpuPID:
			# if(PID not in self.TPUTID_events):
			# 	self.TPUTID_events[PID] = {}
			self.TPUTID_events[PID] = self.TPUTID_events.get(PID, {})

			for TID in TpuTID:
				# if(TID not in self.TPUTID_events[PID]):
				# 	self.TPUTID_events[PID][TID] = []
				self.TPUTID_events[PID][TID] = self.TPUTID_events[PID].get(TID, [])

		for event in self.opEvents:
			if(self.CheckIgnoreOp(event['name'])):
				pass
			elif event['pid'] in HostPID:
					if event['tid'] in HostTID:
						self.HostTID_events[ int(event['tid']) ].append(event)
					else:
						# tf.logging.info("WARRNING: cannot place element with Host TID:\n\t" + str(event))
						pass

			elif event['pid'] in TpuPID: # TPU event
				if event['tid'] in TpuTID:
					self.TPUTID_events[ int(event['pid']) ][int(event['tid'])].append(event)
				else:
					# tf.logging.info("WARRNING: cannot place element with TPU TID:\n\t" + str(event))
					pass


			else:
				print("WARRNING: cannot place element with Host or TPU PID/TID:\n\t" + str(event))
				print("\t\t HostPID["+str(HostPID)+"] HostTID["+str(HostTID)+"] TPUPID["+str(TpuPID)+"] TPUPID["+str(TpuTID)+"]")

	def SortEvents(self):
		for tid_index in self.HostTID_events.keys():
			self.HostTID_events[tid_index] = sorted(self.HostTID_events[tid_index], key = lambda i: i[u'ts'])

		for pid_index in self.TPUTID_events:
			for tid_index in self.TPUTID_events[pid_index]:
				self.TPUTID_events[pid_index][tid_index] = sorted(self.TPUTID_events[pid_index][tid_index], key = lambda i: i[u'ts'])



	def TimeIntersects(self, Astart, Aend, Bstart, Bend):
		# Valid Intersections
		# time: ---------------->
		# A:		|==========|
		# 			|		   		 |
		# B: 	 |===|		   |
		# B: 	  | |===|	   |
		# B: 	  | 	 |===| |
		# B:		|		      |===|
		# 			|		   		 |
		if((Bstart >= Astart) and (Bstart < Aend)):
			return True
		if((Bstart < Astart) and (Bend > Astart)):
			return True
		return False

	def SumRange(self, lis):
		# lis is a list of tuples with start/end times i.e. [(start, end), (start, end),...]
		rangetime = 0.0
		for start, end in lis:
			diff = end - start
			if(diff < 0.0):
				print("ERROR: got a negative time range between s: " + str(start) + " and e: " + str(end))
				quit()
			rangetime += diff
		return rangetime

	def GetEventSelfTime(self, index, eventslist):
		event = eventslist[index]
		totaltime = float(event[u'dur'])
		starttime = float(event[u'ts'])
		endtime = starttime+totaltime
		if(totaltime == 0.0): # No time 
			return 0.0

		# Events are already sorted so no need to check if the next event doesn't overlap
		if( index+1 < len(eventslist) ):
			if( eventslist[index+1][u'ts'] > endtime):
				return totaltime

		resultRanges = [] # list of tubes that contain the start & end time for each valid range left
		resultRanges.append((starttime, endtime))

		for i in range(index+1, len(eventslist)):
			nextevent = eventslist[i]
			if float(nextevent[u'ts']) >= endtime: # starts after this event ends
				break

			Bstart = float(nextevent[u'ts'])
			Bdur = float(nextevent[u'dur'])
			Bend = Bstart+Bdur

			intersectBool = [self.TimeIntersects(Astart, Aend, Bstart, Bend) for Astart, Aend in resultRanges]

			newResultRanges = []

			for i in range(len(resultRanges)):
				# Only if there is an intersection at this range, calculate difference
				if(intersectBool[i]):
					Astart, Aend = resultRanges[i]

					Istart = max(Astart, Bstart)
					Iend = min(Aend, Bend)

					Lstart = Astart
					Lend = Istart
					Rstart = Iend
					Rend = Aend

					if((Lend-Lstart) > 0.0):
						newResultRanges.append((Lstart, Lend))
					if((Rend-Rstart) > 0.0):
						newResultRanges.append((Rstart, Rend))
				else:
					newResultRanges.append(resultRanges[i])

			resultRanges = newResultRanges

		return self.SumRange(resultRanges)

	def GetEventSelfTimes(self):
		for tid_index in self.HostTID_events: # (TODO) Opertunity for parallelism here
			tid_len = len( self.HostTID_events[tid_index] )
			for event_index in range(tid_len):
				event_selfTime = self.GetEventSelfTime(event_index, self.HostTID_events[tid_index])
				self.HostTID_events[tid_index][event_index][u'self'] = event_selfTime		

		for tpu_index in self.TPUTID_events:
			for tid_index in self.TPUTID_events[tpu_index]:
				tid_len = len( self.TPUTID_events[tpu_index][tid_index] )
				for event_index in range(tid_len):
					event_selfTime = self.GetEventSelfTime(event_index, self.TPUTID_events[tpu_index][tid_index])
					# future = executor.submit(self.GetSelfTime2, event_index, TPUTID_events[tpu_index][tid_index])
					# event_selfTime = future.result()
					self.TPUTID_events[tpu_index][tid_index][event_index][u'self'] = event_selfTime



	def GetEventInfo(self):
		for tid_index in self.HostTID_events:
			tid_len = len( self.HostTID_events[tid_index] )
			for event_index in range(tid_len):
				name = self.HostTID_events[tid_index][event_index][u'name']
				# if name not in self.HostTID_events:
				if name not in self.HostInfo:
					self.HostTID_events[tid_index][event_index][u'calls'] = 1
					# self.HostTID_events[name] = HostTID_events[tid_index][event_index]
					self.HostInfo[name] = self.HostTID_events[tid_index][event_index]
				else:
					# self.HostTID_events[name][u'dur'] += HostTID_events[tid_index][event_index][u'dur']
					# self.HostTID_events[name][u'self'] += HostTID_events[tid_index][event_index][u'self']
					# self.HostTID_events[name][u'calls'] += 1
					self.HostInfo[name][u'dur'] += self.HostTID_events[tid_index][event_index][u'dur']
					self.HostInfo[name][u'self'] += self.HostTID_events[tid_index][event_index][u'self']
					self.HostInfo[name][u'calls'] += 1


		
		for tpu_index in self.TPUTID_events:
			for tid_index in self.TPUTID_events[tpu_index]:
				tid_len = len( self.TPUTID_events[tpu_index][tid_index] )
				# if(tid_len != 0):
				for event_index in range(tid_len):
					name = self.TPUTID_events[tpu_index][tid_index][event_index][u'name']
					# if name not in self.TPUTID_events:
					if name not in self.TPUInfo:
						self.TPUTID_events[tpu_index][tid_index][event_index][u'calls'] = 1
						# self.TPUTID_events[name] = TPUTID_events[tpu_index][tid_index][event_index]
						self.TPUInfo[name] = self.TPUTID_events[tpu_index][tid_index][event_index]
					else:
						# self.TPUTID_events[name][u'dur'] += TPUTID_events[tpu_index][tid_index][event_index][u'dur']
						# self.TPUTID_events[name][u'self'] += TPUTID_events[tpu_index][tid_index][event_index][u'self']
						# self.TPUTID_events[name][u'calls'] += 1
						self.TPUInfo[name][u'dur'] += self.TPUTID_events[tpu_index][tid_index][event_index][u'dur']
						self.TPUInfo[name][u'self'] += self.TPUTID_events[tpu_index][tid_index][event_index][u'self']
						self.TPUInfo[name][u'calls'] += 1

	def GetSelfTimes(self, HostPID, HostTID, TpuPID, TpuTID):
		# start_phase = time.time()

		self.SeperateByPIDTID(HostPID, HostTID, TpuPID, TpuTID)
		# tf.logging.info("\033[92m TPUPoint: \033[0m\t\t\t\t\t CSVClass.AnalyzeStep() step.SeperateByPIDTID() TIME: " + str(time.time() - start_phase) ); start_phase = time.time()
		
		self.SortEvents()
		# tf.logging.info("\033[92m TPUPoint: \033[0m\t\t\t\t\t CSVClass.AnalyzeStep() step.SortEvents() TIME: " + str(time.time() - start_phase) ); start_phase = time.time()

		self.GetEventSelfTimes()
		# tf.logging.info("\033[92m TPUPoint: \033[0m\t\t\t\t\t CSVClass.AnalyzeStep() step.GetEventSelfTimes() TIME: " + str(time.time() - start_phase) ); start_phase = time.time()

		self.GetEventInfo()
		# tf.logging.info("\033[92m TPUPoint: \033[0m\t\t\t\t\t CSVClass.AnalyzeStep() step.GetEventInfo() TIME: " + str(time.time() - start_phase) ); start_phase = time.time()


class PhaseClass(object):
	def __init__(self, PhaseNumber=None):
		self.PhaseNumber = PhaseNumber
		self.HostInfo = {}
		self.TPUInfo = {}


	def AddInfo(self, HostInfo, TPUInfo):
		for name in HostInfo:
			try:
				self.HostInfo[name][u'dur'] += HostInfo[name][u'dur']
				self.HostInfo[name][u'self'] += HostInfo[name][u'self']
				self.HostInfo[name][u'calls'] += HostInfo[name][u'calls']
			except KeyError:
				self.HostInfo[name] = HostInfo[name]

		for name in TPUInfo:
			try:
				self.TPUInfo[name][u'dur'] += TPUInfo[name][u'dur']
				self.TPUInfo[name][u'self'] += TPUInfo[name][u'self']
				self.TPUInfo[name][u'calls'] += TPUInfo[name][u'calls']
			except KeyError:
				self.TPUInfo[name] = TPUInfo[name]


	def AddPhaseInfo(self, Phase):

		self.AddInfo(Phase.HostInfo, Phase.TPUInfo)


	def GetTopNOps(self, n_ops=10):
		HostInfoSoretdNames = sorted(self.HostInfo, key = lambda i: self.HostInfo[i][u'self'], reverse=True)
		TPUInfoSoretdNames = sorted(self.TPUInfo, key = lambda i: self.TPUInfo[i][u'self'], reverse=True)

		top_n_host = HostInfoSoretdNames[:n_ops]
		top_n_tpu = TPUInfoSoretdNames[:n_ops]

		return top_n_host, top_n_tpu




class CSVClass(object):
	def __init__(self, file_name_suffix=""):
		self.HostPID = None
		self.HostTID = None
		self.TpuPID = None
		self.TpuTID = None

		self.outputfileName_tpu = "TPUPointOPs_TPU_Phases"+file_name_suffix+".csv"
		self.outputfileName_cpu = "TPUPointOPs_CPU_Phases"+file_name_suffix+".csv"
		self.tpu_phase_writer = csv.writer(open(self.outputfileName_tpu, "a+"), delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
		self.cpu_phase_writer = csv.writer(open(self.outputfileName_cpu, "a+"), delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)


		self.outputfileName_tpu_overview = "TPUPointOPs_TPU_Overview"+file_name_suffix+".csv"
		self.outputfileName_cpu_overview = "TPUPointOPs_CPU_Overview"+file_name_suffix+".csv"
		self.tpu_overview_writer = csv.writer(open(self.outputfileName_tpu_overview, "a+"), delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
		self.cpu_overview_writer = csv.writer(open(self.outputfileName_cpu_overview, "a+"), delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)

		self.WriteHeaders()

		self.writeHeader_tpu = True
		self.writeHeader_cpu = True


		self.HostTID_events = {}
		self.TPUTID_events = {}
		self.PhaseClassList = {}


		self.IGNORE_OPS = IGNORE_OPS

	def CheckIDs(self):
		if(self.HostPID==None ):
			tf.logging.info("\033ERROR: \033[0m HostPID Idenfitiers are empty")
			return False
		if(self.HostTID==None):
			tf.logging.info("\033ERROR: \033[0m HostTID Idenfitiers are empty")
			return False
		if(self.TpuPID==None):
			tf.logging.info("\033ERROR: \033[0m TpuPID Idenfitiers are empty")
			return False
		if(self.TpuTID==None):
			tf.logging.info("\033ERROR: \033[0m TpuTID Idenfitiers are empty")
			return False
		return True

	def GetPIDTID(self, metadataSorted):
		"""
		GetPIDTID: Gets the TPU & CPU Process & Thread IDs for each
		Args:
			metadataSorted: data returned from 
		Return:
			HostPID: Intager of the host Process ID number
			HostTID: List of Host thread ID numbers
			TpuPID: List of TPU process ID numbers
			TpuTID: List of TPU thread ID numbers
					TPU TID are 1, 2, or 3 symbolizing the following as seen in the chrome://tracing
						1: Step
						2: TensorFlow Ops
						3: XLA Ops
		"""

		# metadataSorted = [ {metadatainfo} , ...]

		def mergeIDs(original, found):
			ret = list(set(found))
			if original != None:
				ret += original
			ret = list(set(ret))
			return ret

		HostPID = [] 
		HostTID = []
		TPUPID = []
		TPUTID = []

		for event in metadataSorted:
			if "args" in event:
				if "name" in event["args"]:
					# Found Host PID
					if "Host Threads"  in event["args"]["name"]:
						HostPID.append(int(event["pid"]))
					elif "TPU Core " in event["args"]["name"]:
						TPUPID.append(int(event["pid"]))
					if "tid" in event:
						if event["pid"] in HostPID:
							HostTID.append(int(event["tid"]))
						elif event["pid"] in TPUPID:
							TPUTID.append(int(event["tid"]))

		self.HostPID = mergeIDs(self.HostPID, HostPID)
		self.HostTID = mergeIDs(self.HostTID , HostTID)
		self.TpuPID = mergeIDs(self.TpuPID , TPUPID)
		self.TpuTID = mergeIDs(self.TpuTID , TPUTID)

	def WriteHeaders(self):
		header = [
			"name",
			"n calls", 
			"cumulative n time\n(sec)", 
			"cumulative self n time\n(sec)",
			# "Occurance Percentage",
			"Phase",
			# "PhaseDuration\n(sec)",
			# "PhaseCheckpoint",
		]
		self.WriteCSV(self.tpu_phase_writer, header)
		self.WriteCSV(self.cpu_phase_writer, header)
		# header = [
		# 	"name",
		# 	"n calls", 
		# 	"cumulative n time\n(sec)", 
		# 	"cumulative self n time\n(sec)",
		# 	# "Occurance Percentage",
		# 	"Phase",
		# 	# "PhaseDuration\n(sec)",
		# 	# "PhaseCheckpoint"
		# ]
		self.WriteCSV(self.tpu_overview_writer, header)
		self.WriteCSV(self.cpu_overview_writer, header)

	def WriteCSV(self, event_writer, row):

		event_writer.writerow(row)

	def WriteSingePhase(self, cpu_writer, tpu_writer, phase):
		HostInfoSoretdNames = sorted(phase.HostInfo, key = lambda i: phase.HostInfo[i][u'self'], reverse=True)
		TPUInfoSoretdNames = sorted(phase.TPUInfo, key = lambda i: phase.TPUInfo[i][u'self'], reverse=True)
		phase_number = phase.PhaseNumber

		for name in HostInfoSoretdNames:
			n_calls =  phase.HostInfo[name][u'calls'] / 1e6 # micro sec -> sec
			cumulative_n_time = phase.HostInfo[name][u'dur'] / 1e6 # micro sec -> sec
			cumulative_self_n_time = phase.HostInfo[name][u'self']
			row = [name, n_calls, cumulative_n_time, cumulative_self_n_time, "Phase "+str(phase_number) ]
			# self.WriteCSV(event_writer=self.cpu_phase_writer, row=row)
			self.WriteCSV(event_writer=cpu_writer, row=row)

		for name in TPUInfoSoretdNames:
			n_calls =  phase.TPUInfo[name][u'calls'] / 1e6 # micro sec -> sec
			cumulative_n_time = phase.TPUInfo[name][u'dur'] / 1e6 # micro sec -> sec
			cumulative_self_n_time = phase.TPUInfo[name][u'self']
			row = [name, n_calls, cumulative_n_time, cumulative_self_n_time, "Phase "+str(phase_number) ]
			# self.WriteCSV(event_writer=self.tpu_phase_writer, row=row)
			self.WriteCSV(event_writer=tpu_writer, row=row)

	def WritePhase(self):
		for phase_number in self.PhaseClassList:
			phase = self.PhaseClassList[phase_number]
			self.WriteSingePhase(cpu_writer=self.cpu_phase_writer, tpu_writer=self.tpu_phase_writer, phase=phase)

	def WriteOverview(self):
		OverviewPhase = PhaseClass("Overview")
		for phase_number in self.PhaseClassList: 
			OverviewPhase.AddPhaseInfo( self.PhaseClassList[phase_number] )
		self.WriteSingePhase(cpu_writer=self.cpu_overview_writer, tpu_writer=self.tpu_overview_writer, phase=OverviewPhase)

	def AnalyzeStep(self, phase_number, step):
		# step_getselftime_start = time.time()
		step.GetSelfTimes(self.HostPID, self.HostTID, self.TpuPID, self.TpuTID)
		# tf.logging.info("\033[92m TPUPoint: \033[0m\t\t\t\t TIME step.GetSelfTimes(): " + str(time.time() - step_getselftime_start) )


		# phase_init = time.time()
		self.PhaseClassList[phase_number] = self.PhaseClassList.get(phase_number, PhaseClass(phase_number) )
		# tf.logging.info("\033[92m TPUPoint: \033[0m\t\t\t\t TIME PhaseClassList initalize: " + str(time.time() - phase_init) )

		# addinfo_start = time.time()
		self.PhaseClassList[phase_number].AddInfo(HostInfo=step.HostInfo, TPUInfo=step.TPUInfo)
		# tf.logging.info("\033[92m TPUPoint: \033[0m\t\t\t\t TIME AddInfo(): " + str(time.time() - addinfo_start) )

	def PhaseTopNOps(self, phase_number=None, n_ops=10):
		if(phase_number == None):
			if( len(self.PhaseClassList) == 0):
				return [], []
			phase_number = max( self.PhaseClassList.keys() )
		top_n_host, top_n_tpu = self.PhaseClassList[phase_number].GetTopNOps(n_ops=n_ops)
		return top_n_host, top_n_tpu




class SummarizationClass(object):

	def __init__(self, logdir, printing=True, file_name_suffix="", bucket_name=None, Similarity=0.70, TotalExecTime=None):
		self.logdir = logdir
		self.printing = printing
		self.outputfileName = "TPUPoint_Summarization_"+file_name_suffix+".json"
		self.Similarity=Similarity

		# gs://bucket_name/path --> 'gs:','','bucket_name','path'
		self.bucket_name = str(self.logdir).split('/')[2]
		self.gcs_path = os.path.join( *str(self.logdir).split('/')[3:] )

		self.gcs_profile_path = os.path.join( self.logdir , "plugins", "profile")
		if( tf.io.gfile.exists( self.gcs_profile_path )==False ):
			tf.io.gfile.makedirs( self.gcs_profile_path )
		self.profile_list = tf.gfile.ListDirectory( self.gcs_profile_path )
		self.profile_list = [os.path.join(self.gcs_profile_path , profile) for profile in self.profile_list]

		self.outputJSONData = {
			"displayTimeUnit":"ns",
			"metadata":{"highres-ticks":True},
			"traceEvents":[
				{"ph":"M", "args": {"name": "TPUPoint"}, "pid": -1, "name": "process_name"}, 
				{"tid": 0, "ph": "M", "args": {"name": "TPUPoint"}, "pid": -1, "name": "thread_name"}, 
				{"tid": 1, "ph": "M", "args": {"name": "Profile Breakdown"}, "pid": -1, "name": "thread_name"}, 
				{"tid": 2, "ph": "M", "args": {"name": "Phase Breakdown"}, "pid": -1, "name": "thread_name"}
			],
		}
		self.Profiles = []
		self.Phases = []
		self.PhaseBreakMarkers = []
		self.PhaseBreakMarkerRatios = {}

		self.current_step = None
		self.PhasesBreak = False

		self.IGNORE_OPS = IGNORE_OPS
		self.IGNORE_OPS_LOWER = [l.lower() for l in self.IGNORE_OPS]

		self.CSVClass = CSVClass(file_name_suffix)

		self.ProfileTimes = {}
		self.LoadProfileTimesCSV()

	def LoadProfileTimesCSV(self):
		bucket_name = self.bucket_name
		path = self.logdir
		if(bucket_name not in path):
			path = os.path.join(bucket_name, path)
		if("gs://" not in path):
			path = os.path.join("gs://" , path)
		if("TPUPoint" not in path):
			path = os.path.join(path , "TPUPoint")

		if(not tf.gfile.Exists(path)):
			return
		files_list = tf.gfile.ListDirectory(path)
		if("TPUPoint_Profile_Times.csv" not in files_list):
			return
		ProfileTimesCSVPath = os.path.join(path , "TPUPoint_Profile_Times.csv")
		ProfileTimes = pd.read_csv( tf.gfile.GFile(ProfileTimesCSVPath, 'rb') )
		self.ProfileTimes = ProfileTimes.to_dict('index')

	def GetPhasePercentagesListTimeAccurage(self):
		if(len(self.ProfileTimes) == 0):
			totoal_time = sum([phase['dur'] for phase in self.Phases])
			phase_percentage = [phase['dur']/totoal_time for phase in self.Phases]
			phase_dur = [phase['dur'] for phase in self.Phases]
			return phase_dur, phase_percentage
		
		profile_start_time = min([self.ProfileTimes[profile]['start'] for profile in self.ProfileTimes])
		profile_end_time = max([self.ProfileTimes[profile]['end'] for profile in self.ProfileTimes])
		profile_total_dur = profile_end_time - profile_start_time

		phase_break_lis = []
		for profile_key in self.PhaseBreakMarkerRatios:
			profile_dur = self.ProfileTimes[profile_key]['dur']
			profile_start = self.ProfileTimes[profile_key]['start']
			for phase_break_ratio in self.PhaseBreakMarkerRatios[profile_key]:
				phase_break_lis.append( profile_start + (profile_dur*phase_break_ratio) )
		phase_break_lis = [profile_start_time] + phase_break_lis + [profile_end_time]
		
		phase_real_time_duration = []
		phase_real_time_percentages = []
		for i in range(len(phase_break_lis) - 1):
			ts = phase_break_lis[i]
			end = phase_break_lis[i+1]
			dur = end - ts
			phase_real_time_duration.append(dur)
			phase_real_time_percentages.append( dur / profile_total_dur)

		return phase_real_time_duration, phase_real_time_percentages




	def SumPrint(self, string, status=0):
		if(self.printing == False):
			return
		if(status<=0): # GREEN
			prtstr = "\033[92m TPUPoint: \033[0m" + str(string)
			tf.logging.info(prtstr)
		elif(status==2): # YELLOW
			prtstr = "\033[93m TPUPoint: \033[0m" + str(string)
			tf.logging.info(prtstr)
		elif(status==3): # RED
			prtstr = "\033[91m TPUPoint: \033[0m" + str(string)
			tf.logging.info(prtstr)
		else: # RED Verbose Info. Occurs Less often
			prtstr = "\033[91m TPUPoint: \033[0m" + str(string)
			tf.logging.info(prtstr)


	def FindTraceFiles(self, bucket_name, gcs_file):
		blobList = []
		if(isinstance(gcs_file,list)):
			for i in range(len(gcs_file)):
				if(  os.path.join( "gs://" , bucket_name) not in gcs_file[i] ):
					gcs_file_ = os.path.join( "gs://" , bucket_name, gcs_file[i] )
				else:
					gcs_file_ = gcs_file[i]
				file_list = tf.gfile.ListDirectory(gcs_file_)
				file_list = [file for file in file_list if "trace.json.gz" in file]
				file_list = [os.path.join(gcs_file_ , file) for file in file_list]

				if(file_list == []):
					blobList += [gcs_file_ + ".None"]
				else:
					blobList += file_list
		else:
			# gcs_file_ =  os.path.join( "gs://" , bucket_name, gcs_file )
			if(  os.path.join( "gs://" , bucket_name) not in gcs_file ):
				gcs_file_ = os.path.join( "gs://" , bucket_name, gcs_file[i] )
			else:
				gcs_file_ = gcs_file
			file_list = tf.gfile.ListDirectory(gcs_file_)
			file_list = [file for file in file_list if "trace.json.gz" in file]
			file_list = [os.path.join(gcs_file_ , file) for file in file_list]
			blobList += file_list

			if(file_list == []):
				blobList += [gcs_file_ + ".None"]
			else:
				blobList += file_list
		return blobList


	def GfileToJSONData(self, file_path):
		file = tf.gfile.GFile(file_path, 'rb')
		jsonfile = zlib.decompress(file.read(), 16+zlib.MAX_WBITS)
		data = json.loads(jsonfile)
		data = data['traceEvents'][:-1]
		del jsonfile
		del file
		return data


	def DataToStepList(self, file_path):
		# DataToStepList_start_time = time.time()
		data = self.GfileToJSONData(file_path)
		# self.SumPrint("\t\t\t\t TIME self.GfileToJSONData(): " + str(time.time() - DataToStepList_start_time), 2);DataToStepList_start_time = time.time()
		allevents = []
		allmetadata = []
		for event in data:
			if 'X' in str(event['ph']):
				allevents.append(event)
			else:
				allmetadata.append(event)
		# allevents = [event for event in data if 'X' in str(event['ph'])]
		# allmetadata = [event for event in data if 'M' in str(event[u'ph'])]
		# self.SumPrint("\t\t\t\t TIME allevents & allmetadata: " + str(time.time() - DataToStepList_start_time), 2);DataToStepList_start_time = time.time()

		self.CSVClass.GetPIDTID(allmetadata)
		# self.SumPrint("\t\t\t\t self.CSVClass.GetPIDTID(): " + str(time.time() - DataToStepList_start_time), 2);DataToStepList_start_time = time.time()

		ops, stps = self.SortStepsAndOps(allevents)
		# self.SumPrint("\t\t\t\t self.SortStepsAndOps(): " + str(time.time() - DataToStepList_start_time), 2);DataToStepList_start_time = time.time()
		StepList = self.CreateStepObjectList(ops, stps)
		# self.SumPrint("\t\t\t\t self.CreateStepObjectList(): " + str(time.time() - DataToStepList_start_time), 2);DataToStepList_start_time = time.time()

		# (TODO) Remove. This is for debugging purposes
		# num_steps = [step.NumSteps() for step in StepList if step.NumSteps() > 0]
		# num_ops = [step.NumOps() for step in StepList]
		# self.SumPrint("\t\t Number of step objects: " + str(len(StepList)) )
		# self.SumPrint("\t\t Number of unique steps: " + str(sum(num_steps)))
		# self.SumPrint("\t\t Number of ops totoal: " + str(sum(num_ops))  )
		del data
		return StepList

	
	def CompareSteps(self, previous_step, next_step):
		# # Jaccard Similarity but only the intersection(A,B)/B events where B contains less events
		s1 = previous_step.OpNames() # set(stepA[1].keys())
		s2 = next_step.OpNames() # set(stepB[1].keys())

		div = float( min( len(s1) , len(s2) ) )
		num = float( len( s1.intersection(s2) ) )
		if(div == 0):
			ret = 1.0
		else:
			ret =  num / div

		difference = s1.union(s2) - s1.intersection(s2)

		return ret, difference


	def Analysis(self, file_name, current_step, profile_number):
		phase_break = False

		# start_time = time.time()
		# Analysis_start_time = start_time

		# Example Phase testPhase = { "name": "Phase", "pid": -1, "ts": 0.0, "ph": "X", "dur": 50000.0}
		profile = {"name":"Profile("+str(profile_number)+")", "ph": "X", "pid": -1, "tid":1, "ts": 100.0*profile_number, "dur": 100.0}
		self.Profiles.append(profile)


		if("None" in file_name):
			self.SumPrint("\t\t No trace file found", 2)
			return None, phase_break

		StepList = self.DataToStepList(file_name)

		# self.SumPrint("\t\t\t TIME DataToStepList(): " + str(time.time() - start_time), 2);start_time = time.time()

		profile_start = StepList[0].GetStartTime()
		profile_end = StepList[-1].GetEndTime()
		profile_dur = profile_end - profile_start

		if(current_step == None):
			current_step = StepList.pop(0)

			self.CSVClass.AnalyzeStep(phase_number=self.GetCurrentPhaseNumber(), step=current_step)
			# self.SumPrint("\t\t\t TIME self.CSVClass.AnalyzeStep()1: " + str(time.time() - start_time), 2);start_time = time.time()

		for step in StepList:

			similarity, diff = self.CompareSteps(current_step, step)
			# self.SumPrint("\t\t\t TIME self.CompareSteps(): " + str(time.time() - start_time), 2);start_time = time.time()

			if(similarity < self.Similarity):
				phase_break = True
				self.SumPrint("\t\t BREAK SIMILARITY: " + str(similarity) , 3)
				# self.SumPrint("\t\t\t current_step.NumSteps(): "+str(current_step.NumSteps())+" next_step.NumSteps(): "+str(step.NumSteps()) , 3 )
				# self.SumPrint("\t\t\t current_step.OpNames(): "+str(len(current_step.OpNames()))+" next_step.OpNames(): "+str(len(step.OpNames())) , 3 )
				self.SumPrint("\t\t\t intersection: "+str(len( current_step.OpNames().intersection(step.OpNames()) ))+" div: "+str( min(len(current_step.OpNames()) , len(step.OpNames())) ) , 3 )
				# self.SumPrint("\t\t\t diff: " + str(diff), 3 )

				pahseBreakRatio = (step.GetStartTime() - profile_start) / profile_dur
				phaseStart = (100.0*profile_number) + (pahseBreakRatio * 100.0)
				self.PhaseBreakMarkers.append(phaseStart)
				self.PhaseBreakMarkerRatios[profile_number] = self.PhaseBreakMarkerRatios.get(profile_number, []) + [pahseBreakRatio]


			current_step = step
			self.CSVClass.AnalyzeStep(phase_number=self.GetCurrentPhaseNumber(), step=current_step)
			# self.SumPrint("\t\t\t TIME self.CSVClass.AnalyzeStep()2: " + str(time.time() - start_time), 2);start_time = time.time()

		# self.SumPrint("\t\t\t TIME Analysis Time: " + str(time.time() - Analysis_start_time), 2)
		# self.current_step = current_step
		# self.PhasesBreak = phase_break

		if(phase_break):
			top_n_host, top_n_tpu = self.CSVClass.PhaseTopNOps()
			top_host_count = set(TOP_CPU_OPS).intersection(set(top_n_host))
			top_tpu_count = set(TOP_TPU_OPS).intersection(set(top_n_tpu))
			if(top_host_count == 0 or top_tpu_count == 0):
				phase_break = False
			else:
				self.PhasesBreak = True

		return current_step, phase_break


	def Summarization(self):
		self.SumPrint("STARTING SUMMARIZATION @ similarity: " + str(self.Similarity))
		start_time = time.time() 
		trace_files_list = self.FindTraceFiles(self.bucket_name, self.profile_list)
		

		for index_, file_name in enumerate(trace_files_list):
			file_total_start = time.time()
			self.SumPrint("FILE["+str(index_)+"]: " + str(file_name))

			# pr = cProfile.Profile()
			# pr.enable()

			# Analysis_start_time = time.time()
			self.current_step, phase_break = self.Analysis(file_name=file_name, current_step=self.current_step, profile_number=index_)
			# self.Analysis(file_name=file_name, current_step=self.current_step, profile_number=index_)
			# self.SumPrint("\t\t TOTAL self.Analysis() Time: " + str(time.time() - Analysis_start_time) )
			
			# pr.print_stats()

			if(phase_break): 
				self.SumPrint("\t\t\t TPUPoint Should apply optimization: " + str(phase_break), 3)


			# (TODO) Remove. For debugging purposes only
			mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
			self.SumPrint("\t\t SINGLE Summarization Time: " + str(time.time() - file_total_start) + " mem: " + str(mem_usage) )


		self.WriteOutputJSON()
		self.CSVClass.WritePhase()
		self.CSVClass.WriteOverview()

		self.SumPrint("\t TOTAL OVERALL Summarization Time: " + str(time.time() - start_time) )
		# self.SumPrint("\t\t\t SUMMARIZATION.PROFILE: " + str(self.Profiles) )
		# self.SumPrint("\t\t\t SUMMARIZATION.PHASE: " + str(self.Phases) )		


	def WriteOutputJSON(self):
		profile_end = self.Profiles[-1]["ts"] + self.Profiles[-1]["dur"]
		current_phase_start = self.Profiles[0]["ts"]
		phase_number = 0

		for phase_break in self.PhaseBreakMarkers:
			phase_start = current_phase_start
			phase_dur = phase_break - phase_start
			phase = {"name":"Phase("+str(phase_number)+")", "ph": "X", "pid": -1, "tid":2, "ts": phase_start, "dur": phase_dur}
			self.Phases.append(phase)

			phase_number += 1
			current_phase_start = phase_break

		phase_start = current_phase_start
		phase_dur = profile_end - phase_start
		phase = {"name":"Phase("+str(phase_number)+")", "ph": "X", "pid": -1, "tid":2, "ts": phase_start, "dur": phase_dur}
		self.Phases.append(phase)


		self.outputJSONData["traceEvents"] += self.Profiles
		self.outputJSONData["traceEvents"] += self.Phases

		with open(self.outputfileName, "w") as outfile:
			json.dump(self.outputJSONData, outfile)

	def DrawImage(self):
		poly_coor_list = PhaseToPolygonCoordinateList(self.Phases, similarity=self.Similarity)
		DrawPhases(poly_coor_list, graph_title="Phases Over Profiles", ylabel="Similarity", xlabel="Profiles")


	### Helper Methods

	def GetCurrentPhaseNumber(self):

		return len(self.PhaseBreakMarkers)

	def CleanEventName(self, name, ignoreop=False):
		new_name = str(name)
		if(ignoreop):
			return new_name
		# if("linearizex32" in name.lower()):
		# 	self.SumPrint("\t\t\t CHANGE FROM "+name+" --> linearizex32", 3)
		# 	return "linearizex32"
		new_name = new_name.split(' ')[0]
		new_name = new_name.split(':')[-1]
		new_name = new_name.split(' ')[0]
		new_name = new_name.split('.')[0]
		new_name = re.sub('\[.+\]', '', new_name) 
		new_name = new_name.replace('_','')
		new_name = new_name.lower()
		return new_name

	def CheckStepOrOp(self, event):
		try:
			int(event[u'name'])
			return True
		except ValueError:
			return False

	def CheckIgnoreOp(self, opname):
		opname_lower = opname.lower()
		# for ignore_op in self.IGNORE_OPS:
		# 	if(ignore_op.lower() in opname.lower()):
		# 		return True

		ret_bool = [(ignore_op in opname_lower) for ignore_op in self.IGNORE_OPS_LOWER]
		return sum(ret_bool)

	def SortStepsAndOpsWorker(self, data):
		ops = []
		stps = []

		for event in data:
			# if( self.CheckStepOrOp(event) ):
			if( event[u'name'].isdigit() == False ):
				event[u'name'] = self.CleanEventName(name=event[u'name'], ignoreop=self.CheckIgnoreOp(event[u'name']) )
				ops.append(event)
			else:
				stps.append( event )
		return ops, stps

	def SortStepsAndOps(self, data):
		split = multiprocessing.cpu_count()
		split = len(data) / split
		split = max(1, split)

		ops = []
		stps = []

		ops, stps = self.SortStepsAndOpsWorker(data)

		# with concurrent.futures.ThreadPoolExecutor() as executor:
		# 	future_list = []
		# 	for i in xrange(0, len(data), split) :
		# 		future = executor.submit(self.SortStepsAndOpsWorker, data[i:i+split])
		# 		future_list.append(future)
		# 	for future in concurrent.futures.as_completed(future_list):
		# 		ret_ops, ret_stps = future.result()
		# 		ops += ret_ops
		# 		stps += ret_stps

		return ops, stps

	def CreateStepObjectList(self, eventsList, stepsList):
		stepsList = sorted(stepsList, key=lambda x: x["ts"])
		eventsList = sorted(eventsList, key=lambda x: x["ts"])

		length = len(stepsList)
		stpRet = []
		eventsRet = []

		ret = []
		
		StepList = []

		for i in range(len(stepsList)):

			if(StepList == []):
				step_obj = StepClass()
				step_obj.AddStep( stepsList[i] )
				StepList.append( step_obj )

			elif( StepList[-1].StepName() == stepsList[i]['name'] ):
				StepList[-1].AddStep( stepsList[i] ) 

			else:
				step_obj = StepClass()
				step_obj.AddStep( stepsList[i] )
				StepList.append( step_obj )

		# No step events in profile.
		# Create empty step event and assign all events to that step
		if(StepList == []):
			step_obj = StepClass()
			for event in eventsList:
				step_obj.AddOp( event )
			StepList.append( step_obj )
			return StepList


		current_index = 0
		current_step = StepList[current_index]
		current_start = current_step.GetStartTime()
		current_end = current_step.GetEndTime()
		current_index += 1


		prefixStep = StepClass()
		suffixStep = StepClass()

		for i in range(len(eventsList)):
			event = eventsList[i]
			event_start = event['ts']
			# event_end = event['ts'] + event['dur']
			if( (event_start <  current_start ) ): # occured before all steps
				prefixStep.AddOp( event )
			elif( (event_start > current_start) and (event_start < current_end) ): # occurs durring step
				current_step.AddOp( event )
			elif( current_index >= len(StepList) ): # occurs after all steps
				suffixStep.AddOp( event )
			else:
				current_step = StepList[current_index]
				current_start = current_step.GetStartTime()
				current_end = current_step.GetEndTime()
				current_index += 1
				current_step.AddOp( event )



		if( prefixStep.NumOps() > 0 ):
			# StepList = [prefixStep] + StepList
			for opEvents in prefixStep.opEvents:
				StepList[0].AddOp( opEvents )

		if( suffixStep.NumOps() > 0):
			# StepList = StepList + [suffixStep]
			for opEvents in suffixStep.opEvents:
				StepList[-1].AddOp( opEvents )

		return StepList
		#####################



def PhaseToPolygonCoordinateList(phase_list, similarity=0.0):
	ret_lis = []
	total_dur = sum([phase['dur'] for phase in phase_list])
	for i, phase in enumerate(phase_list):
		phase_start = phase['ts']
		phase_end = phase['ts'] + phase['dur']

		start_x = phase_start / total_dur
		end_x = phase_end / total_dur

		# bl, tl, tr, br
		bl = (start_x, similarity)
		tl = (start_x, similarity+0.08)
		tr = (end_x, similarity+0.08)
		br = (end_x, similarity)
		phase_coordinates = [bl, tl, tr, br]
		ret_lis.append(phase_coordinates)

	return ret_lis

def color_gen():
	while 1:
		for c in itertools.cycle(['tab:red', 'tab:green', 'tab:blue', 'tab:orange']):
			yield c

def DrawPhases(patches_list, graph_title="", ylabel="Similarity", xlabel="Profiles"):
	fig, ax = plt.subplots() 
	start_height = patches_list[0][0][1]
	color_ = color_gen()  
	for patch in patches_list:
		# color = (random.random(), random.random(), random.random())
		# ax.add_patch( plt.Polygon( patch, color=color )  ) 
		current_height = patch[0][1]
		if(current_height != start_height):
			start_height = current_height
			color_ = color_gen()
		ax.add_patch( plt.Polygon( patch, color=color_.next() )  ) 
	plt.ylabel(ylabel)
	plt.xlabel(xlabel)
	plt.title(graph_title)
	plt.ylim((0.0, 1.12))
	plt.xlim((0.0, 1.0)) 
	if(graph_title != ""):
		graph_title = "_"+graph_title
	plt.savefig("TPUPointPhaseImage"+graph_title+".png")

def RunSummarization(logdir, bucket_name, output_prefix, Similarity):
	if(output_prefix != ""): 
		_output_prefix = "_"+str(output_prefix)
	else:
		_output_prefix = output_prefix
	file_name_suffix = str(_output_prefix)+"_"+str(Similarity)
	sumclass = SummarizationClass(logdir=logdir, printing=True, file_name_suffix=file_name_suffix, bucket_name=bucket_name, Similarity=Similarity, TotalExecTime=None)
	sumclass.Summarization()
	phase_dur, phase_percent = sumclass.GetPhasePercentagesListTimeAccurage()
	num_phases = len(sumclass.Phases)
	poly_coor_list = PhaseToPolygonCoordinateList(sumclass.Phases, similarity=Similarity)

	return num_phases, poly_coor_list, Similarity, phase_percent

def main(logdir, output_prefix):
	bucket_name = logdir.split("/")[2]
	original_output_prefix = output_prefix
	if(output_prefix != ""): output_prefix = "_"+str(output_prefix)

	TotalExecTime = None

	similarity_writter = csv.writer(open("TPUPointSimilarityTesting.csv", "a+"), delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
	
	# SimilarityList = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	SimilarityList = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]


	totoal_patches = []
	num_phases_list = []
	num_similarity_list = []
	num_phases_percentages_list = []




	# with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
	with concurrent.futures.ProcessPoolExecutor() as executor:
		future_list = []
		for Similarity in SimilarityList:

			future = executor.submit(RunSummarization, logdir, bucket_name, output_prefix, Similarity)
			future_list.append(future)

		# for future in concurrent.futures.as_completed(future_list):
		for future in future_list:
			num_phases, poly_coor_list, ret_similarity, phase_percent = future.result()

			num_phases_list.append(num_phases)
			num_similarity_list.append(ret_similarity)
			num_phases_percentages_list.append( str(phase_percent) )
			for poly_coor in poly_coor_list:
				totoal_patches.append(poly_coor)

	# for Similarity in SimilarityList:
	# 	if(output_prefix != ""): 
	# 		_output_prefix = "_"+str(output_prefix)
	# 	else:
	# 		_output_prefix = output_prefix
	# 	file_name_suffix = str(_output_prefix)+"_"+str(Similarity)
	# 	sumclass = SummarizationClass(logdir=logdir, file_name_suffix=file_name_suffix, bucket_name=bucket_name, Similarity=Similarity, TotalExecTime=TotalExecTime)
	# 	sumclass.Summarization()
	# 	num_phases = len(sumclass.Phases)
	# 	num_phases_list.append(num_phases)
	# 	poly_coor_list = PhaseToPolygonCoordinateList(sumclass.Phases, similarity=Similarity)
	# 	for poly_coor in poly_coor_list:
	# 		totoal_patches.append(poly_coor)

	DrawPhases(totoal_patches, graph_title=original_output_prefix)
	similarity_writter.writerow(["similarity"] + num_similarity_list)
	similarity_writter.writerow(["num_phases"] + num_phases_list)
	similarity_writter.writerow(["phase_percentage"] + num_phases_percentages_list)



if __name__ == '__main__':

	if (len(sys.argv) < 3):
		tf.logging.info("\033[91m Failed: \033[0m missing command line arguments \n\"Summarization.py <gs://bucket/dir/>")
		exit()
	if "gs://" not in sys.argv[1]:
		tf.logging.info("\033[91m Failed: \033[0m bucket dir name must be in \"gs://bucket/dir/\" format")
		exit()


	logdir = sys.argv[1]
	output_prefix = sys.argv[2]

	main(logdir=logdir, output_prefix=output_prefix)
