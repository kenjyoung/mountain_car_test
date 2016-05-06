import mountaincar
from Tilecoder import numTilings, tilecode, numTiles
from Tilecoder import numTiles as n
from pylab import *  #includes numpy

numEpisodes = 10
alpha = 0.5/numTilings
lmbda = 0.9
Epi = 0
Emu = 0.1
F = np.zeros(numTilings*n)
F_next = np.zeros(numTilings*n)
thetaQ = -0.01*np.random.random_sample(numTilings*n*3)
thetaV = -0.01*np.random.random_sample(numTilings*n)
e = np.zeros(numTilings*n*3)
avgret = np.zeros(numEpisodes)
avgstep = np.zeros(numEpisodes)

def Qs(F):
    return [np.sum([thetaQ[a*numTilings*n+index] for index in F]) for a in [0,1,2]]

def V(F):
    return np.sum([thetaV[numTilings*n+index] for index in F])

def writeQ():
    fout = open('actionValue', 'w')
    F = [0]*numTilings
    steps = 50
    for i in range(steps):
        for j in range(steps):
            tilecode(-1.2+i*1.8/steps, -0.07+j*0.14/steps, F)
            height = -max(Qs(F))
            fout.write(repr(height) + ' ')
        fout.write('\n')
    fout.close()

def writeV():
    fout = open('stateValue', 'w')
    F = [0]*numTilings
    steps = 50
    for i in range(steps):
        for j in range(steps):
            tilecode(-1.2+i*1.8/steps, -0.07+j*0.14/steps, F)
            height = -V(F)
            fout.write(repr(height) + ' ')
        fout.write('\n')
    fout.close()

def writeAvgret():
    fout = open('avgret.dat', 'w')
    for i in range(len(avgret)):
        fout.write(str(i)+' '+str(avgret[i])+' '+str(avgstep[i])+'\n')
    fout.close()

def q(s, a):
    p = s[0]
    v = s[1]
    tilecode(p, v, F)
    return np.sum([thetaQ[a*numTilings*n+index] for index in F])

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

thetaQ = -0.01*np.random.random_sample(numTilings*n*3)
returnSum = 0.0
G = 0
for episodeNum in xrange(numEpisodes):
    e = np.zeros(numTilings*n*3)
    S = mountaincar.init()
    step = 0
    while(S!=None):
        step+=1
        A = epsilon_greedy_policy(S)
        R, S_next = mountaincar.sample(S,A)
        G+=R
        #value of terminal state is 0 by definition so no need to compute
        #q values for it
        if(S_next==None):
            delta = R - q(S,A)
        #otherwise expected q value is just the average value weighted by the 
        #we choose randomly plus the max value weighted by probabilty we choose
        #greedily
        else:
            delta = R+Epi*np.average([q(S_next,a) for a in [0,1,2]]) +\
                (1-Epi)*np.max([q(S_next,a) for a in [0,1,2]]) - q(S,A)
        e*=lmbda
        tilecode(S[0], S[1], F)
        for index in [i+A*numTilings*n for i in F]:
            e[index] = 1
        thetaQ +=alpha*delta*e
        S=S_next

print "Overall performance: Average return per episode: ", G/numEpisodes
writeQ()
writeAvgret()

b = np.zeros(numTilings*n)
B = np.zeros((numTilings*n,numTilings*n))
for episodeNum in xrange(numEpisodes):
    S = mountaincar.init()
    e = np.zeros(numTilings*n)
    while(S!=None):
        A = greedy_policy(S)
        R, S_next = mountaincar.sample(S,A)
        tilecode(S[0], S[1], F)
        tilecode(S_next[0],S_next[1],F_next)
        e*=lmbda
        for index in F:
            e[index] += 1
        B = B + np.outer(e,F-F_next)
        b = b + e*R
        S = S_next
    returnSum = returnSum + G
thetaV = np.linalg.lstsq(B,b)
writeV()




