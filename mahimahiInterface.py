import socket
import sys
import numpy as np
from time import sleep, time
from numpy import uint32
from numpy import int32
from numpy import uint64
from numpy import int64


class RLMahimahiInterface():
    def __init__(self):
        self.enLog = 1
        self.enRL = 0
        self.RLTargetQlen = 40
        self.RLDrop = 0.1

        self.last_id = 0
        self.last_sum = 0
        self.last_abs = 0
        self.last_square = 0
        self.last_qdelay = 0
        self.last_dqbyte = 0
        self.last_eqbyte = 0
        self.last_dqpkg = 0
        self.last_qempty_time = 0
        self.last_acc_qdelay = 0
        self.old_dp = 0
        self.action_delay = 0
        self.action_ts = 0
        pass

    def __WriteProc(self):
        cmd = "W " + str(self.enLog) + " " + str(self.enRL)
        cmd = cmd + " " + str(self.RLTargetQlen)
        dp = self.RLDrop
        cmd = cmd + " " + str(dp)
        # print(cmd)
        # Popen("sudo echo '%s' > /proc/rlint" % cmd,shell=True).wait()
        self.sk.send(cmd)

    def ConnectToMahimahi(self, ip='100.64.0.4', port=4999):
        self.TCP_IP = ip
        self.TCP_PORT = port
        print ip, " ", port
        self.sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sk.connect((ip, port))
        self.sk.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        pass

    def SetLogState(self, state):
        self.enLog = state
        self.__WriteProc()
        pass

    def SetRLState(self, state):
        self.enRL = state
        self.__WriteProc()
        pass

    def SetTargetQlen(self, qlen):
        self.RLTargetQlen = qlen
        self.__WriteProc()
        pass

    def SetDropRate(self, dropRate):
        dropRate = np.minimum(dropRate, 1.0)
        dropRate = np.maximum(dropRate, 0.0)
        self.RLDrop = dropRate  # * 0.25 + self.old_dp*0.75
        self.old_dp = self.RLDrop
        self.__WriteProc()
        self.action_delay = time() - self.action_ts
        # print self.action_delay

        pass

    def GetState(self, dump=True):
        ## Read State From Kernel Module
        ## State[0] Number of packets enqueued since last query
        ## State[1] Average queue length since last query
        ## State[2] Average of (queue_length_when_packet_enqueues - target_queue_length) since last query
        ## State[3] Average of ABS(queue_length_when_packet_enqueues - target_queue_length) since last query
        ## State[4] Average of SQR(queue_length_when_packet_enqueues - target_queue_length) since last query
        ## State[5] Number of packets dequeued since last query
        ## State[6] Accumulated queue delay since last query (millisecond)
        ## State[7] Accumulated bytes dequeued since last query (throughput)
        ## State[8] Queue Delay which is calculated in the same way as PIE.
        ## State[9] Drop rate of PIE
        ## State[10] Accumulated bytes enqueued since last query (throughput)
        ## state[11] Queue Empty Time in milliseconds
        ## State[12] max qdelay
        ## State[13] min qdelay
        ## State[14] acc qdelay
        ## State[15] The throughput of the next 160ms (8 X 20ms)
        ## You may need to convert R into correct reward formats.

        # info = os.popen("cat /proc/rlint").read()
        # info = info.split()
        self.action_ts = time()
        self.sk.send("R")
        t0 = time()
        # print "com send: " ,time() - self.action_ts
        info = self.sk.recv(256)
        # print info
        # print "com recv: " ,time() - t0
        # print("INFO")
        # print "com: ", time() - self.action_ts

        info = info.split()
        # print(info)

        n = uint64(info[0]) - self.last_id
        s = int64(info[1]) - self.last_sum
        a = uint64(info[2]) - self.last_abs
        sq = uint64(info[3]) - self.last_square
        qdelay = uint64(info[4]) - self.last_qdelay
        dqbyte = uint64(info[5]) - self.last_dqbyte
        eqbyte = uint64(info[11]) - self.last_eqbyte
        dqpkg = uint64(info[6]) - self.last_dqpkg
        qempty_time = uint64(info[12]) - self.last_qempty_time
        max_qdelay = uint64(info[13])
        min_qdelay = uint64(info[14])
        acc_qdelay = uint64(info[15]) - self.last_acc_qdelay

        self.last_id = uint64(info[0])
        self.last_sum = int64(info[1])
        self.last_abs = uint64(info[2])
        self.last_square = uint64(info[3])
        self.last_qdelay = uint64(info[4])
        self.last_dqbyte = uint64(info[5])
        self.last_dqpkg = uint64(info[6])
        self.last_eqbyte = uint64(info[11])
        self.last_qempty_time = uint64(info[12])
        self.last_acc_qdelay = uint64(info[15])

        PIEQDelay = uint64(info[7])
        PIEDP = float(info[8])
        # print PIEQDelay/15625.0

        if dqpkg > 0:
            tmp = dqpkg
        else:
            tmp = 1

        if n > 0:
            ret = [n, float(s + n * self.RLTargetQlen) / n, float(s) / n, float(a) / n, float(sq) / n, dqpkg,
                   float(qdelay) * 64 / 1000000 / tmp, float(dqbyte), PIEQDelay, PIEDP, float(eqbyte),
                   float(qempty_time), float(max_qdelay), float(min_qdelay), float(acc_qdelay / tmp), info[16:24]]
        else:
            ret = [0, 0, 0, 0, 0, 0, 0, 0, PIEQDelay, PIEDP, 0, 0, 0, 0, 0, info[16:24]]

        if dump == True:
            info = "Enq %(enq)4d, Deq %(deq)4d,  Avg_Q_Len %(qlen)7d,   Avg_Q_Delay %(qdelay)7d ms,  BytesDequeued %(debyte)8d bytes" % {
                "enq": n, "deq": dqpkg, "qlen": ret[8], "qdelay": ret[8], "debyte": dqbyte};
            print(info)

        return ret


import socket
import sys
import numpy as np
from time import sleep, time
from numpy import uint32
from numpy import int32
from numpy import uint64
from numpy import int64


class RLMahimahiInterface():
	def __init__(self):
		self.enLog = 1
		self.enRL = 0
		self.RLTargetQlen = 40
		self.RLDrop = 0.1

		self.last_id = 0
		self.last_sum = 0
		self.last_abs = 0
		self.last_square = 0
		self.last_qdelay = 0
		self.last_dqbyte = 0
		self.last_eqbyte = 0
		self.last_dqpkg = 0
		self.last_qempty_time = 0
		self.last_acc_qdelay = 0
		self.old_dp = 0
		self.action_delay = 0
		self.action_ts = 0
		pass

	def __WriteProc(self):
		cmd = "W " + str(self.enLog) + " " + str(self.enRL)
		cmd = cmd + " " + str(self.RLTargetQlen)
		dp = self.RLDrop
		cmd = cmd + " " + str(dp)
		# print(cmd)
		# Popen("sudo echo '%s' > /proc/rlint" % cmd,shell=True).wait()
		self.sk.send(cmd)

	def ConnectToMahimahi(self, ip='100.64.0.4', port=4999):
		self.TCP_IP = ip
		self.TCP_PORT = port
		print ip, " ", port
		self.sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.sk.connect((ip, port))
		self.sk.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
		pass

	def SetLogState(self, state):
		self.enLog = state
		self.__WriteProc()
		pass

	def SetRLState(self, state):
		self.enRL = state
		self.__WriteProc()
		pass

	def SetTargetQlen(self, qlen):
		self.RLTargetQlen = qlen
		self.__WriteProc()
		pass

	def SetDropRate(self, dropRate):
		dropRate = np.minimum(dropRate, 1.0)
		dropRate = np.maximum(dropRate, 0.0)
		self.RLDrop = dropRate  # * 0.25 + self.old_dp*0.75
		self.old_dp = self.RLDrop
		self.__WriteProc()
		self.action_delay = time() - self.action_ts
		# print self.action_delay

		pass

	def GetState(self, dump=True):
		## Read State From Kernel Module
		## State[0] Number of packets enqueued since last query
		## State[1] Average queue length since last query
		## State[2] Average of (queue_length_when_packet_enqueues - target_queue_length) since last query
		## State[3] Average of ABS(queue_length_when_packet_enqueues - target_queue_length) since last query
		## State[4] Average of SQR(queue_length_when_packet_enqueues - target_queue_length) since last query
		## State[5] Number of packets dequeued since last query
		## State[6] Accumulated queue delay since last query (millisecond)
		## State[7] Accumulated bytes dequeued since last query (throughput)
		## State[8] Queue Delay which is calculated in the same way as PIE.
		## State[9] Drop rate of PIE
		## State[10] Accumulated bytes enqueued since last query (throughput)
		## state[11] Queue Empty Time in milliseconds
		## State[12] max qdelay
		## State[13] min qdelay
		## State[14] acc qdelay
		## State[15] The throughput of the next 160ms (8 X 20ms)
		## You may need to convert R into correct reward formats.

		# info = os.popen("cat /proc/rlint").read()
		# info = info.split()
		self.action_ts = time()
		self.sk.send("R")
		t0 = time()
		# print "com send: " ,time() - self.action_ts
		info = self.sk.recv(256)
		# print info
		# print "com recv: " ,time() - t0
		# print("INFO")
		# print "com: ", time() - self.action_ts

		info = info.split()
		# print(info)

		n = uint64(info[0]) - self.last_id
		s = int64(info[1]) - self.last_sum
		a = uint64(info[2]) - self.last_abs
		sq = uint64(info[3]) - self.last_square
		qdelay = uint64(info[4]) - self.last_qdelay
		dqbyte = uint64(info[5]) - self.last_dqbyte
		eqbyte = uint64(info[11]) - self.last_eqbyte
		dqpkg = uint64(info[6]) - self.last_dqpkg
		qempty_time = uint64(info[12]) - self.last_qempty_time
		max_qdelay = uint64(info[13])
		min_qdelay = uint64(info[14])
		acc_qdelay = uint64(info[15]) - self.last_acc_qdelay

		self.last_id = uint64(info[0])
		self.last_sum = int64(info[1])
		self.last_abs = uint64(info[2])
		self.last_square = uint64(info[3])
		self.last_qdelay = uint64(info[4])
		self.last_dqbyte = uint64(info[5])
		self.last_dqpkg = uint64(info[6])
		self.last_eqbyte = uint64(info[11])
		self.last_qempty_time = uint64(info[12])
		self.last_acc_qdelay = uint64(info[15])

		PIEQDelay = uint64(info[7])
		PIEDP = float(info[8])
		# print PIEQDelay/15625.0

		if dqpkg > 0:
			tmp = dqpkg
		else:
			tmp = 1

		if n > 0:
			ret = [n, float(s + n * self.RLTargetQlen) / n, float(s) / n, float(a) / n, float(sq) / n, dqpkg,
				   float(qdelay) * 64 / 1000000 / tmp, float(dqbyte), PIEQDelay, PIEDP, float(eqbyte),
				   float(qempty_time), float(max_qdelay), float(min_qdelay), float(acc_qdelay / tmp), info[16:24]]
		else:
			ret = [0, 0, 0, 0, 0, 0, 0, 0, PIEQDelay, PIEDP, 0, 0, 0, 0, 0, info[16:24]]

		if dump == True:
			info = "Enq %(enq)4d, Deq %(deq)4d,  Avg_Q_Len %(qlen)7d,   Avg_Q_Delay %(qdelay)7d ms,  BytesDequeued %(debyte)8d bytes" % {
				"enq": n, "deq": dqpkg, "qlen": ret[8], "qdelay": ret[8], "debyte": dqbyte};
			print(info)

		return ret
