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
  print('dq=',q-q0)
  z = z0 + pm*sigma - (q-q0)
  print('dz',z-z0)
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
  #print('t0,dt,Es:',t0,dt,Es)

  #compute basic means
  wE    = sum(ws*Es)
  wE2   = sum(ws*Es*Es)
  wEt   = sum(ws*Es*dt)
  wE2t  = sum(ws*Es*Es*dt)
  wE2t2 = sum(ws*Es*Es*dt*dt)
  wY    = sum(ws*ys)
  wYE   = sum(ws*ys*Es)
  wYEt  = sum(ws*ys*Es*dt)
  #print('wE,wE2',wE,wE2)

  #secondary quantities
  wExE   = wE2  - wE*wE
  wEtxE  = wE2t - wEt*wE  # = wExEt 
  wEtxEt = wE2t2 - wEt*wEt
  wYxE   = wYE  - wY*wE
  wYxEt  = wYEt - wY*wEt
  #print('wExE:', wExE)

  #solve
  q0dd =  ( wYxE*wEtxE - wYxEt*wExE ) /  ( wEtxE*wEtxE - wEtxEt*wExE )
  dnew = d0 + q0dd/q0 
  qnew = (- wYxE  + q0dd*wEtxE ) / wExE
  znew = wY + qnew*wE - q0dd*wEt
  #print('q0,d0,q0dd',q0,d0,q0dd)

  pars={}
  pars['z']  = znew
  pars['q']  = qnew
  pars['d']  = dnew
  pars['t0'] = t0

  if stats:
    ynew = model(ts,pars)
    residual = ys - ynew
    F = sum(ws*residual*residual)
    pars['sigma']=np.sqrt(F)
    if 't0' in pars0:
      ynew = model(ts,pars0)
      residual = ys - ynew  
      F = sum(ws*residual*residual)
      print('sigma,sigma0:',pars['sigma'],np.sqrt(F))
     
  return pars

def model_lsf(ts,ys,ws,stats=True):
  pars={'q':2,'d':0.01}
  dp=1
  pars0 = dict(pars)
  while dp > 1e-6:  
    pars  =  model_llsf(ts,ys,ws,pars,stats)
    dq=pars['q']-pars0['q']
    dd=pars['d']-pars0['d']
    dp = (dq)**2/(pars['q']+pars0['q'])**2 + (dd)**2/(pars['d']+pars0['d'])**2
    pars0=dict(pars)
    print('dp,pars:',dp,pars)
    pars['d']-=dd*0.1
    pars['q']-=dq*0.1
  return pars

def make_model(ts, ys, n, tend, wtpow=0,it=None):
  #make model from partial series up to entry it using data of length n
  if it is not None:
    iend=it+1
  else:
    iend=len(ts)
  istart=max([0,iend-n])
  print('Fitting from ',ts[istart],'to',ts[iend-1])
  ws=np.exp(wtpow*ys)
  #print('ws:',ws)
  pars=model_lsf(ts[istart:iend],ys[istart:iend],ws[istart:iend],stats=True)
  dt=1
  t=np.arange(ts[0],tend)
  f=model(t,pars)
  fplus=model_err(t,pars,pars['sigma'],1)
  fminus=model_err(t,pars,pars['sigma'],-1)
  return [t,f,fplus,fminus]

