#! /usr/bin/env python
#!# cbekar

import os
from datetime import datetime

print "Parsing experiment files\n"
homedir = os.getenv('HOME')
expdir = homedir + "/Research/KUDA"

exprmap = {}

# create & open the final output file
resultfile = open(expdir + "/" + "RESULTS.txt", 'w')

NUMOFEXP = 10

def mywrite(str):
	resultfile.write(str)
def mywriteline(str):
	resultfile.write(str + "\n")

# write current date and time
mywriteline(datetime.now().isoformat(' '))

# do it for benchmarks
mywriteline("Datarace results:".upper() + "\t\t\t\tProgram Time" + "\tAnalysis Time"+ "\tReal Time")

filtered = ''
#this loop parses, filters and reduces several runs of the same experiment
for filename in os.listdir(expdir + "/experiments"):
	if filename.find("datarace_") != 0:
		print "Skipping phase1", filename, "\n"
		continue
	# we shall have a set of filesets seen before and filter them out, just for performance
	rawfile = filename.split('.')[0].split('_run')[0]
	# rawfile = "check_suite_bench_input_config(_hw_memtype_#kernel_#block_#thread)"
	if rawfile in exprmap:
		filtered = exprmap[rawfile]
	else:
		filtered = "\n***************************************************\n" + rawfile.upper() + "\n***************************************************\n\n"
	
	#entry = "" 	# this will have the significant parsed total time, we can take an array of entries in the later versions
	stats = [[]*3]*10 # this map will have as much entry as the number of runs key = run, value = statlist
	statlist = [] 	# this will be a temp list for the values of stats map
	
	for run in range(NUMOFEXP):
		# get the full path of the file
		nextexpfilename = rawfile + "_run" + str(run)
		nextexpfilepath = os.path.join(expdir + '/experiments/', nextexpfilename + '.txt')
		# read the file
		with open(nextexpfilepath, "r") as parsethis:
			statlist = []
			for line in parsethis:
				if line.find('real') != -1:
					temp1 = (line)
					temp1 = temp1.split('\t')[1]
					temp2 = temp1.split('m')
					temp1 = temp2[0]
					temp2 = temp2[1]
					temp1 = int(temp1) * 60000 # now mseconds
					temp2 = temp2.strip('s\n')
					temp2 = temp2.split('.')
					temp2 = str(temp2[0]) + str(temp2[1])
					temp1 = temp1 + int(temp2)
					statlist.append(temp1)
				if line.find('TIME for program') != -1:
					temp1 = (line)
					temp2 = temp1.split('seconds,')[0]
					temp2 = temp2.split(':')[1]
					temp2 = temp2.strip()
					temp1 = temp1.split('seconds,')[1]
					temp1 = temp1.split('micro')[0]
					temp1 = temp1.strip()
					temp1 = int(temp1) / 1000
					temp1 += int(temp2) *1000
					statlist.append(temp1)
				if line.find('TIME for analysis') != -1:
					temp1 = (line)
					temp2 = temp1.split('seconds,')[0]
					temp2 = temp2.split(':')[1]
					temp2 = temp2.strip()
					temp1 = temp1.split('seconds,')[1]
					temp1 = temp1.split('micro')[0]
					temp1 = temp1.strip()
					temp1 = int(temp1) / 1000
					temp1 += int(temp2) *1000
					statlist.append(temp1)
			#end for
			if len(statlist) == 1:
				statlist[:0] = 0,0 # this is for such configurations that has only "real" significant value
			stats[run] = statlist
		#averagemap[run] = filtered
	# end for
	# now we have stat = statlist
		#averagemap[run] = filtered
		
	# end for
	# we should pop the max and min and average the remains #!#
	#print "presort".upper
	#print stats

	#stats.sort()
	#print "aftersort".upper
	#print stats
	exprmap[rawfile] = stats
#end for

#print 'Statistics', exprmap
# now we have stats[] list, presumably having 10 entries, each entry having the significant runtime informations, which may vary in number!
# print 'Statistics for ', rawfile , 'is',  stats
# we should pop the max and min and average the remains, therefor having a triple of significant result per experiment/bench/configuration batch
parse = {}

for key in exprmap:
	avgruntime = [0,0,0]
	
	for i in range(NUMOFEXP):
		avgruntime[0] += int(exprmap[key][i][0])
		avgruntime[1] += int(exprmap[key][i][1])
		avgruntime[2] += int(exprmap[key][i][2])
	MIN1 = 99999999999
	MAX1 = 0
	MIN2 = 99999999999
	MAX2 = 0
	MIN3 = 99999999999
	MAX3 = 0
	for i in range(NUMOFEXP):
		if(int(exprmap[key][i][0]) > MAX1): MAX1 = int(exprmap[key][i][0])
		if(int(exprmap[key][i][0]) < MIN1): MIN1 = int(exprmap[key][i][0])
		if(int(exprmap[key][i][1]) > MAX2): MAX2 = int(exprmap[key][i][1])
		if(int(exprmap[key][i][1]) < MIN2): MIN2 = int(exprmap[key][i][1])
		if(int(exprmap[key][i][2]) > MAX3): MAX3 = int(exprmap[key][i][2])
		if(int(exprmap[key][i][2]) < MIN3): MIN3 = int(exprmap[key][i][2])
	#print MIN1, MIN2, MIN3, MAX1, MAX2, MAX3
	avgruntime[0] -= MIN1 + MAX1
	avgruntime[1] -= MIN2 + MAX2
	avgruntime[2] -= MIN3 + MAX3
	avgruntime[1] /= NUMOFEXP-2
	avgruntime[2] /= NUMOFEXP-2
	avgruntime[0] /= NUMOFEXP-2
	parse[key] = avgruntime

strbuffer = []

for key in exprmap:
	strbuffer.append(key + '\t\t' + str(parse[key][0]) + "\t\t" + str(parse[key][1]) + "\t\t" + str(parse[key][2]))

strbuffer.sort()

# write out the contents of strbuffer
for i in range(len(strbuffer)):
	mywriteline(strbuffer[i])

resultfile.close()

print "Done.\n"
