import mountaincar
from Tilecoder import numTilings, tilecode, numTiles
from Tilecoder import numTiles as n
from pylab import *  #includes numpy

numRuns = 50
numEpisodes = 200
alpha = 0.5/numTilings
gamma = 1
lmbda = 0.9
Epi = Emu = epsilon = 0
F = [-1]*numTilings
theta = -0.01*np.random.random_sample(numTilings*n*3)
e = np.zeros(numTilings*n*3)
avgret = np.zeros(numEpisodes)
avgstep = np.zeros(numEpisodes)

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
        fout.write(str(i)+' '+str(avgret[i])+' '+str(avgstep[i])+'\n')
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

runSum = 0.0
for run in xrange(numRuns):
    theta = -0.01*np.random.random_sample(numTilings*n*3)
    e = np.zeros(numTilings*n*3)
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
            e*=gamma*lmbda
            tilecode(S[0], S[1], F)
            for index in [i+A*numTilings*n for i in F]:
                e[index] = 1
            theta +=alpha*delta*e
            S=S_next
        returnSum = returnSum + G
        #running average for each episode number
        avgret[episodeNum] = (avgret[episodeNum]*run + G)/(run+1)
        avgstep[episodeNum] = (avgstep[episodeNum]*run + G)/(run+1)
        print "Episode: ", episodeNum, "Steps:", step, "Return: ", G
    print "Average return:", returnSum/numEpisodes
    runSum += returnSum
print "Overall performance: Average sum of return per run:", runSum/numRuns
writeF()
writeAvgret()


#Additional code here to write average performance data to files for plotting...
#You will first need to add an array in which to collect the data


