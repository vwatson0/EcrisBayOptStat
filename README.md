Bayesian optimization under control constraint with Gaussian process smoothness auto-tunning 

This pakage includes:
- 2 libraries necessary to run a bayasian optimizer on an ECR ion source like VENUS
- 1 library to simulate a 2dimensional input fake ion source
- 1 test file to test the kalman filter
- 1 test file to test the fake ion source
- 1 test file and jupyter-notebook file to test all the functionalities

  To use the kalman filter for time series tracking:
  
    import KalmanFilterStdEst

  Create an object:

    myObj = KalmanFilterStdEst.KFobject(FirstMeasure(scalar), FilterParameter(scalar))

  To estimate recusively the stats of the time series:

    myObj.EstimateState(CurrentMeasure(scalar), ElaspsedTimeSinceLastCall(scalar))

  To have the current statistics:
    MeanValue = myObj.X[0]
    StDev = myObj.Sig[0]
    slope = myObj.X[1]


  To use the optimizer:

    import EcrisBayOps

  Create an object:

    Opt = EcrisBayOps.Optimizer(Klen, alpha, exBias, expNoise, risk, thresh)

  The parameters are:

  - Klen: kernel length (vect)
  - alpha: possible values for alpha (vect)
  - exBias: exploration exploitation bias (scalar)
  - risk: acceptable rsik of evaluating an unstable area (scalar)
  - thresh: stability threshold (scalar)
  - expNoise: expected noise level (scalar)
  - limits: limits of the exploration space (2xn vect)
 
  Get the next settings for the system:

    nextSettings = EcrisBayOps.NextPointQuery(PreviousSettings, Y, S, Opt, limits)

  The parameters are:

    - nextSettings: Settings the Bayesian optimizer is suggesting (vect - len:d) - d: number of parameters
    - PreviousSettings: Series of settings previously measured used to build the gaussian processes (vect - nxd)
    - Y: series of output (beam current) associated with the previous settings
    - S: series of control (stability) associated with the prervious settings
    - Opt: optimizer object
    - limits: search boundaries for the parameters (2xd)
  
