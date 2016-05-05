import mountaincar
from Tilecoder import numTilings, tilecode, numTiles
from Tilecoder import numTiles as n
from pylab import *  #includes numpy

numEpisodes = 200
numRuns = 10
alpha = 0.75/numTilings
gamma = 1
lmbda = 0.9
Epi = Emu = epsilon = 0
F = [-1]*numTilings
theta = -0.01*np.random.random_sample(numTilings*n*3)
e = np.zeros(numTilings*n*3)

def Qs(F):
    return [theta[a*numTilings*n+index] for index in F]

def writeF():
    fout = open('value', 'w')
    F = [0]*numTilings
    steps = 50
    for i in range(steps):
        for j in range(steps):
            tilecode(-1.2+i*1.8/steps, -0.07+j*0.14/steps, F)
            height = -max(Qs(F))
            fout.write(repr(height) + ' ')
        fout.write('\n')
    fout.close()

def writeAvgret():
    fout = open('avgret.dat', 'w')
    for i in range(len(avgret)):
        fout.write(str(i)+' '+str(avgret[i])+'\n')
    fout.close()

def q(s, a):
    p = s[0]
    v = s[1]
    tilecode(p, v, F)
    return np.sum([theta[a*numTilings*n+index] for index in F])

#return the maximizing action for a state
def greedy_policy(s):
    return np.argmax([q(s,a) for a in [0,1,2]])

#return the maximizing action for the state with probability (1-Emu), or
#a random action with probability Emu
def epsilon_greedy_policy(s):
    rand = np.random.random()
    if(rand>Emu):
        return greedy_policy(s)
    return np.random.choice([0,1,2])

def test_params(_lmbda, _alpha, _epsilon):
	global theta, e
	Epi = Emu = _epsilon
	alpha = _alpha
	lmbda = _lmbda
	runSum = 0.0
	for run in xrange(numRuns):
		e = np.zeros(numTilings*n*3)
		theta = -0.01*np.random.random_sample(numTilings*n*3)
		returnSum = 0.0
		for episodeNum in xrange(numEpisodes):
		    G = 0
		    S = mountaincar.init()
		    step = 0
		    while(S!=None):
		        step+=1
		        A = epsilon_greedy_policy(S)
		        R, S_next = mountaincar.sample(S,A)
		        G+=R
		        #since value of terminal state is 0 by definition
		        #computation for delta is simplified
		        if(S_next==None):
		            delta = R - q(S,A)
		        else:
		            delta = R+Epi*np.average([q(S_next,a) for a in [0,1,2]]) +\
		                (1-Epi)*np.max([q(S_next,a) for a in [0,1,2]]) - q(S,A)
		        e*=gamma*lmbda
		        tilecode(S[0], S[1], F)
		        for index in [i+A*numTilings*n for i in F]:
		            e[index] = 1
		        theta +=alpha*delta*e
		        S=S_next
		        if(step >10000): return -10000000000
		    returnSum = returnSum + G
		runSum += returnSum
	return runSum/numRuns


best_value = -1000000000
best_Q = None
p = [lmbda, alpha]
best_p = p[:]
dp = [0.05, 0.05]
for i in range(100):
	for j in range(len(p)):
		p[j] += dp[j]
		if(p[j]<0): p[j] = 0
		if(p[j]>1): p[j] = 1
		value= test_params(p[0], p[1], 0)
		if(value > best_value):
			best_value = value
			best_p = p[:]
			dp[j]*=1.1
		else:
			p[j]-= 2*dp[j]
			if(p[j]<0): p[j] = 0
			if(p[j]>1): p[j] = 1
			value= test_params(p[0], p[1], 0)
			if(value > best_value):
				best_value = value
				best_p = p[:]
				dp[j] *=1.1
			else:
				p[j] +=dp[j]
				if(p[j]<0): p[j] = 0
				if(p[j]>1): p[j] = 1
				dp[j]*=0.95
		print "Best Value: ", best_value, " Current Value: ", value, " Best Paramters: ", str(best_p)
print "[",
for i in range(0,180):
	print "[",best_Q[i,0],",",best_Q[i,1],"],"
print "[",best_Q[180,0],",",best_Q[180,1],"]",
print "]"


