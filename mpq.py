import torch
import math
from tqdm import tqdm

scale=lambda w,m,M:(w-m)/(M-m)
unscale=lambda w,m,M:(M-m)*w+m

def quantize(ws,b,m,M):
    bb=(1<<b)
    ws=scale(ws,m,M)*bb
    ws=ws.to(torch.int64).to(torch.float64)/bb
    ws=unscale(ws,m,M)
    return ws

quantize_local= lambda ws,b:quantize(ws,b,torch.min(ws),torch.max(ws))

absw=lambda w:math.prod(w.shape)

def MCKP(dL,w,Bset,targetb,layernum):
    #initialize each layer with the minimum available bits
    bs=[min(Bset)]*layernum 
    for l in range(layernum):
        #remove dominated item of bits
        #if( (b<=bp) and (dL[l][b]<=dL[l][bp]) ):#dominate b>>bp
        for i,b in enumerate(Bset):
            for j in range(i,len(Bset)):            
                if( dL[l][i]>dL[l][j]):#not dominate !(bi>>bj)
                    break
            if(j==len(Bset)-1):
                bs[l]=b
                break

    print("bs",bs)

    while sum(bs) < targetb*layernum :
        prio=[0]*layernum
        for l in range(layernum):        
            bn=Bset[Bset.index(bs[l])+1] #next available bit

            prio[l]=(dL[l][bn]-dL[l][b[l]])/((bn-b[l])*absw(w[l]))
        lstar=prio.index(max(prio))
        bs[lstar]=bn

    return bs

def showshape(out,w,deltaw):
        print("out.shape",out.shape)
    #    print("loss.grad",loss.grad)
        print("out.grad",out.grad)
        print("out.grad.shape",out.grad.shape)#d out[y]/d out
        for l,ws in enumerate(w):
            print(f"w[{l}].grad.shape",ws.grad.shape) #d out[y]/d w_0
        print("deltaw[0][0].shape",deltaw[0][0].shape)

def getBitAssignment(dataloader,model,criterion,Bset,targetb,batchsize,except_activation=True,DEBUG=False,itenum=1000000):  
    #mode.train()  
    w=list(model.parameters())
    if(except_activation):
        w=[ wi for wi in w if wi.dim()!=1]
    layernum=len(w)
    deltaw=[ [quantize_local(wi,b)-wi for b in Bset] for wi in w]
    deltaL=[[0.]*len(Bset)]*layernum

    if(DEBUG):
        print("layernum",layernum)

    N=len(dataloader) #batchsize
    for n, (x, y) in enumerate(tqdm(dataloader)):    
        out=model(x)
        #loss = criterion(out, y)
        out.retain_grad()

        outsum=out[0,y[0]]
        for t in range(1,batchsize):
            outsum+=out[t,y[t]]
        
        outsum.backward()

        if(DEBUG):
            showshape(out,w,deltaw)

        for l in range(layernum):    
            for bi in range(len(Bset)):
                dd=[1/(2*N)*torch.sum(w[l].grad * deltaw[l][bi])**2/out[t,y[t]]**2 for t in range(batchsize)]
                deltaL[l][bi]=sum(dd)/batchsize

        if(n>itenum):
            break
    if(DEBUG):
       print("deltaL ",deltaL)
#    print("deltaL[0][0].shape",deltaL[0][0].shape)
        
    return MCKP(deltaL,w,Bset,targetb,layernum)

