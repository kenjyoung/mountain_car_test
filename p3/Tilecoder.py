import numpy as np

numTilings = 4
numTiles = 81
    

def tilecode(p,v,tileIndices):
	#get the index within each individual tiling
    for i in range(numTilings):  
    	#set equal to tiling origin
    	tileIndices[i]  = i*numTiles
    	#move up velocity axis 
    	tileIndices[i] += int((v+0.07)/0.0175+i/float(numTilings))*9
    	#move across position axis
    	tileIndices[i] += int((p+1.2)/0.225+i/float(numTilings))

    
    
def printTileCoderIndices(x,y):
    tileIndices = [-1]*numTilings
    tilecode(x,y,tileIndices)
    print 'Tile indices for input (',x,',',y,') are : ', tileIndices
    
