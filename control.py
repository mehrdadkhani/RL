from rlcommon import *
import code

redis_server = redis.StrictRedis(host='localhost',port=6379,db=0)

def load(name):
	cmd = {'cmd':'load','name':name}
	redis_server.publish('cmd',pickle.dumps(cmd))


def store(name):
	cmd = {'cmd':'store','name':name}
	redis_server.publish('cmd',pickle.dumps(cmd))


def stop_training():
	cmd = {'cmd':'stop_training'}
	redis_server.publish('cmd',pickle.dumps(cmd))



def resume_training():
	cmd = {'cmd':'resume_training'}
	redis_server.publish('cmd',pickle.dumps(cmd))

def nomore():
	cmd = {'cmd':'nomore'}
	redis_server.publish('cmd',pickle.dumps(cmd))

def more():
	cmd = {'cmd':'more'}
	redis_server.publish('cmd',pickle.dumps(cmd))

def exp():
	cmd = {'cmd':'exp'}
	redis_server.publish('cmd',pickle.dumps(cmd))
	

print("Commands:")
print("load(name)\nstore(name)\nstop_training\nresume_training\n")

code.interact(local=locals())
