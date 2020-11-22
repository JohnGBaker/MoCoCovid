import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook

verbose=False
def vprint(*args):
    if verbose: print(*args)

def model(ts,pars):
  '''  
    Model:
      model func form: 
        y = z - q * exp(-d*(t-t0))
      note y = ln (number of cases)
        z = asymptote
        d = exponential rate of approach to asmptote
        q*d*exp(d*t0) = early rate of exponential growth
        z - q = y(t0)
  '''
  #pars
  z=pars['z']
  d=pars['d']
  q=pars['q']
  t0=pars['t0']

  dt = ts-t0
  #print(d,t0)
  result = z - q * np.exp(-dt*d)

  return result

def model_err(ts, pars, sigma, pm):
  #pm=+/-1
  #sigma is weighted sum of residual^2
  z0=pars['z']
  d0=pars['d']
  q0=pars['q']
  t0=pars['t0']

  dt = ts-t0

  #in this we take err in d as fractional
  d = d0*(1+sigma)**-pm
  E0 = np.exp(t0*d0)
  E  = np.exp(t0*d)
  #define perturbed q,z so that function value is preserved at t=0
  #and raised/lowered by sigma at t=t0
  #q =  q0 + ( pm*sigma - q0*(1-E/E0) ) / (E - 1)
  q = q0 + ( pm*sigma - q0*(E-E0) ) / ( E -1 )
  #try to assure that f+/f>1 and f'+/f'>1 :dq=q0*|dd|*t0 ??
  q = q0*(E0/E) 
  vprint('dq=',q-q0)
  z = z0 + pm*sigma - (q-q0)
  vprint('dz',z-z0)
  result = z - q * np.exp(-dt*d)
  return result

def model_llsf(ts,ys,ws,pars0,stats=True):
  '''
    Perform linearlized least squares fit on model function/
      Arguments:
        ts = ordinate values
        ys = data values
        ws = weights
        pars0 = linearization point
    Returns updated pars.

    Model:
      model func form: 
        y = z - q * exp(-d*(t-t0))
      note y = ln (number of cases)
        z = asymptote
        d = exponential rate of approach to asmptote
        q*d*exp(d*t0) = early rate of exponential growth
        z - q = y(t0)

      linearization:
        ylin = z' - q' * E + q0*E*dd
        q' = q0 + dq
        d' = d0 + dd
        E = exp(-d0*(t-t0))

      weighting:
        For any function f over the input set, we use weighted averages for the optimization
        If weights ws are initially normalized,  sum(ws)=1.
        Then <f> = sum(f*ws)

      Fitting:
        For the fitting, we optimize F = < (ys - y(ts,pars'))^2 >
        dF/dpars:
          0 = dFdz = <(Y-y')>   = <Y>   - z'     + q'<E>    - q0*dd <Et>
          0 = dFdq = <(Y-y')E>  = <YE>  - z'<E>  + q'<E^2>  - q0*dd <E^2t> 
          0 = dFdd = <(Y-y')Et> = <YEt> - z'<Et> + q'<E^2t> - q0*dd <E^2t^2>
          denote (eg):
            <YE>   => wY
            <E^2t> => wE2t 
            dFdz   => wY - z' + q'*wE - q0*dd*wEt
          Thus       
            A = dFdq-wE*dFdz  =  wYE-wY*wE  + q'*(wE2-wE^2)    - q0*dd*(wE2t-wEt*wE)
            B = dFdd-wEt*dFdz = wYEt-wY*wEt + q'*(wE2t-wE*wEt) - q0*dd*(wE2t2-wEt^2))
          denote (eg)
            wYE-wY*wE   => wYExE
            wE2t2-wEt*wEt => wE2tvEt
            A =>  wYxE  + q'*wExE  - q0*dd*wEtxE
            B =>  wYxEt + q'*wExEt - q0*dd*wEtxEt
          Thus
            C = 0 = A*wExEt-B*wExE = wYxE*wExEt - wYxEt*wExE - q0*dd* (wEtxE*wExEt-wEtxEt*wExE)
          Solve:
            q0dd =  ( wYxE*wExEt - wYxEt*wExE ) /  ( wEtxE*wExEt - wEtxEt*wExE )
            d' = d0 + q0dd/q0 
            q' = (- wYxE  + q0dd*wEtxE ) / wExE
            z' = wY + q'*wE - q0dd*wEt
  '''
  #pars
  d0=pars0['d']
  q0=pars0['q']

  #normalize
  ws=ws/sum(ws)
  #print('ws',ws)

  #set t0 as mean of ts
  t0 = sum(ws*ts)
  dt = ts-t0
  Es = np.exp(-dt*d0)
  #print('t0,dt:',t0,dt)
  #print('Es',Es)

  #compute basic means
  wE    = sum(ws*Es)
  wE2   = sum(ws*Es*Es)
  wEt   = sum(ws*Es*dt)
  wE2t  = sum(ws*Es*Es*dt)
  wE2t2 = sum(ws*Es*Es*dt*dt)
  wY    = sum(ws*ys)
  wYE   = sum(ws*ys*Es)
  wYEt  = sum(ws*ys*Es*dt)
  #vprint('wE,wE2',wE,wE2)

  #secondary quantities
  wExE   = wE2  - wE*wE
  wEtxE  = wE2t - wEt*wE  # = wExEt 
  wEtxEt = wE2t2 - wEt*wEt
  wYxE   = wYE  - wY*wE
  wYxEt  = wYEt - wY*wEt
  #vprint('wExE:', wExE)

  #solve
  q0dd =  ( wYxE*wEtxE - wYxEt*wExE ) /  ( wEtxE*wEtxE - wEtxEt*wExE )
  dmin = max([0.0001,d0/10.0])
  dnew = d0 + q0dd/q0
  dnew = max([dnew,dmin])
  qmin = -1
  qnew = max([qmin,(- wYxE  + q0dd*wEtxE ) / wExE])
  znew = wY + qnew*wE - q0dd*wEt
  #vprint('q0,d0,q0dd',q0,d0,q0dd)

  pars={}
  pars['z']  = znew
  pars['q']  = qnew
  pars['d']  = dnew
  pars['t0'] = t0

  if stats:
    #print(ts,pars)
    ynew = model(ts,pars)
    #print(ynew)
    residual = ys - ynew
    #print(ws,residual)
    F = sum(ws*residual*residual)
    #print('F',F)
    pars['sigma']=np.sqrt(F)
    if 't0' in pars0:
      ynew = model(ts,pars0)
      residual = ys - ynew  
      F = sum(ws*residual*residual)
      #print('F0',F)
      vprint('sigma,sigma0:',pars['sigma'],np.sqrt(F))
     
  return pars

def model_lsf(ts,ys,ws,stats=True):
  pars={'q':0.9,'d':0.01}
  dp=1
  pars0 = dict(pars)
  while dp > 1e-6: 
    #print('p',pars)
    pars  =  model_llsf(ts,ys,ws,pars,stats)
    dq=pars['q']-pars0['q']
    dd=pars['d']-pars0['d']
    dp = (dq)**2/(pars['q']+pars0['q'])**2 + (dd)**2/(pars['d']+pars0['d'])**2
    pars0=dict(pars)
    vprint('dp,pars:',dp,pars)
    dstep=0.005
    pars['d']-=dd*dstep
    #if pars['q'] < dq*dstep**2:  pars['q'] = dq*dstep**2
    #else: 
    pars['q']-=dq*dstep
  return pars

def make_model(ts, ys, n, tend, wtpow=0,it=None,nsigma=2):
  #make model from partial series up to entry it using data of length n
  if it is not None:
    iend=it+1
  else:
    iend=len(ts)
  istart=max([0,iend-n])
  vprint('Fitting from ',ts[istart],'to',ts[iend-1])
  ws=np.exp(wtpow*ys)
  #vprint('ws:',ws)
  pars=model_lsf(ts[istart:iend],ys[istart:iend],ws[istart:iend],stats=True)
  dt=1
  t=np.arange(ts[0],tend)
  f=model(t,pars)
  fplus=model_err(t,pars,pars['sigma'],1*nsigma)
  fminus=model_err(t,pars,pars['sigma'],-1*nsigma)
  return [t,f,fplus,fminus]

def show_model(datafile='MoCoCovidData.csv',fitdays=None,fitwidth=30,nextrap=45,minday=10,delta=False,col="Moco"):
    coviddata=pd.read_csv(datafile)
    ts=coviddata.index.values+1
    ys=np.log(coviddata[col+' cases'].dropna().values)
    ndata=len(ys)
    ts=ts[:ndata]
    
    datebase = np.datetime64('2020-03-04')
    dfri=2

    if col=='20853':
        datebase = np.datetime64('2020-04-11')
        dfri=6 
        
    if fitdays is None:
        #print('computing fitdays. last day is',ts[-1])
        maxfits=6
        spacing=7
        fitdays=ts[(ts-dfri)%spacing==0]
        #print('fitdays=',fitdays)
        if len(fitdays)>maxfits: fitdays=fitdays[-maxfits:]
        #print('fitdays=',fitdays)
  
    #create fig
    colors=['b','g','r','c','m','y']
    fig,axs=plt.subplots(1,figsize=(13.5,9))
    #ax0=axs[0]
    ax1=axs
    #locator = mdates.AutoDateLocator(minticks=int(ndata/14), maxticks=20)
    locator=mdates.MonthLocator()
    minorloc=mdates.WeekdayLocator(mdates.FR)
    #formatter = mdates.ConciseDateFormatter(locator)
    formatter = mdates.DateFormatter('%b')
    minorform = mdates.DateFormatter('%d')
    #ax0.xaxis.set_major_locator(locator)
    #ax0.xaxis.set_minor_locator(minorloc)    
    #ax0.xaxis.set_major_formatter(formatter)
    #ax0.xaxis.set_minor_formatter(minorform)
    #ax0.xaxis.set_tick_params(which='major',pad=15)
    ax1.xaxis.set_major_locator(locator)
    ax1.xaxis.set_minor_locator(minorloc)    
    ax1.xaxis.set_major_formatter(formatter)
    ax1.xaxis.set_minor_formatter(minorform)
    ax1.xaxis.set_tick_params(which='major',pad=15)
    #ax0.grid(axis='x',which='minor')
    #ax0.grid(axis='y',which='major')
    ax1.grid(axis='x',which='minor')
    ax1.grid(axis='y',which='major')
    
    ymax=0    
    ic=0
    for fitday in fitdays:
        #prep for fit
        if fitday is None: fitday=ts[-1]
        tmax=ts[-1]+nextrap
        tmin=10
        ifit=len(ts[ts<=fitday+0.5])
        imin=len(ts[ts<minday])
        fw=fitwidth
        if ifit-fw<imin: fw=ifit-imin
        itmin=len(ts[ts<=tmin])
        fts=ts[itmin:ifit]
        fys=ys[itmin:ifit]
        #print('fitting over {} <= t <= {}'.format(fts[0],fts[-1]))
        
        c=colors[len(colors)-1-ic]
        ic+=1
        fade=0.6**(len(fitdays)-ic)
        #make_model (x weight)
        [t,f,fm,fp]=make_model(fts,fys,fw,tmax,1)
        #print('x weight',f3m,f3p)
        if delta: #plot day-to-day difference
            #ax0.plot(datebase+t[1:],f[1:]-f[:-1],c,label='x weight',alpha=fade)
            ax1.plot(datebase+t[1:],np.exp(f[1:])-np.exp(f[:-1]),c,label='x weight',alpha=fade)
            #ax0.plot(datebase+fts[-1:],fys[-1:]-fys[-2:-1],c+'.',ms=20)
            ax1.plot(datebase+fts[-1:],np.exp(fys[-1:])-np.exp(fys[-2:-1]),c+'.',ms=20)
            #ax0.fill_between(datebase+t[1:],
            #                 fm[1:]-fm[:-1],
            #                 fp[1:]-fp[:-1],
            #                 color=c,alpha=0.2*fade)
            ax1.fill_between(datebase+t[1:],
                             np.exp(fm[1:])-np.exp(fm[:-1]),
                             np.exp(fp[1:])-np.exp(fp[:-1]),
                             color=c,alpha=0.2*fade)
            ymax=max([ymax,max(np.exp(f[1:])-np.exp(f[:-1]))])
        else:
            #ax0.plot(datebase+t,[max([0,x])for x in f],c,label='x weight',alpha=fade)
            ax1.plot(datebase+t[0:],np.exp(f),c,label='x weight',alpha=fade)
            #ax0.plot(datebase+fts[-1:],fys[-1:],c+'.',ms=20)
            ax1.plot(datebase+fts[-1:],np.exp(fys[-1:]),c+'.',ms=20)
            #ax0.fill_between(datebase+t,[max([0,x])for x in fm],[max([0,x])for x in fp],color=c,alpha=0.2*fade)
            ax1.fill_between(datebase+t,np.exp(fm),np.exp(fp),color=c,alpha=0.2*fade)
            ymax=max([ymax,max(np.exp(f))])
    #plot data
    c='k'
    if delta:
        #ax0.plot(datebase+ts[1:],ys[1:]-ys[:-1],c+'.',ms=10)
        x=datebase+ts[1:]
        y=np.exp(ys[1:])-np.exp(ys[:-1])
        #ax1.plot(datebase+ts[1:],np.exp(ys[1:])-np.exp(ys[:-1]),c+'.',ms=10)
        ax1.plot(x,y,c+'.',ms=10)
        x=x[6:]
        y=[np.mean(y[ii-6:ii+1]) for ii in range(6,len(y))]
        ax1.plot(x,y,'k-')
    else:
        #ax0.plot(datebase+ts,ys,c+'.',ms=10)
        ax1.plot(datebase+ts,np.exp(ys),c+'.',ms=10)
    ymax*=1.5 
    if plt.ylim()[1]>ymax: plt.ylim(top=ymax)
    #return fig
