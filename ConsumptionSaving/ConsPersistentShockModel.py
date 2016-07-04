'''
Classes to solve consumption-saving models with idiosyncratic shocks to income
in which shocks are not necessarily fully transitory or fully permanent.  Extends
ConsIndShockModel by explicitly tracking permanent income as a state variable.
'''

import sys 
import os
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('./'))

from copy import copy, deepcopy
import numpy as np
from HARKutilities import warnings  # Because of "patch" to warnings modules
from HARKinterpolation import LowerEnvelope2D, BilinearInterp, Curvilinear2DInterp,\
                              LinearInterpOnInterp1D, LinearInterp
from HARKutilities import CRRAutility, CRRAutilityP, CRRAutilityPP, CRRAutilityP_inv,\
                          CRRAutility_invP, CRRAutility_inv, CRRAutilityP_invP,\
                          approxLognormal
from ConsIndShockModel import ConsIndShockSetup, ConsumerSolution, IndShockConsumerType

utility       = CRRAutility
utilityP      = CRRAutilityP
utilityPP     = CRRAutilityPP
utilityP_inv  = CRRAutilityP_inv
utility_invP  = CRRAutility_invP
utility_inv   = CRRAutility_inv
utilityP_invP = CRRAutilityP_invP

class MargValueFunc2D():
    '''
    A class for representing a marginal value function in models where the
    standard envelope condition of V'(M,p) = u'(c(M,p)) holds (with CRRA utility).
    This is copied from ConsAggShockModel, with the second state variable re-
    labeled as permanent income p.    
    '''
    def __init__(self,cFunc,CRRA):
        '''
        Constructor for a new marginal value function object.
        
        Parameters
        ----------
        cFunc : function
            A real function representing the marginal value function composed
            with the inverse marginal utility function, defined on market
            resources and the level of permanent income: uP_inv(VPfunc(M,p)).
            Called cFunc because when standard envelope condition applies,
            uP_inv(VPfunc(M,p)) = cFunc(M,p).
        CRRA : float
            Coefficient of relative risk aversion.
            
        Returns
        -------
        new instance of MargValueFunc
        '''
        self.cFunc = deepcopy(cFunc)
        self.CRRA = CRRA
        
    def __call__(self,M,p):
        return utilityP(self.cFunc(M,p),gam=self.CRRA)
        
###############################################################################
        
class ConsIndShockSolverExplicitPermInc(ConsIndShockSetup):
    '''
    A class for solving the same one period "idiosyncratic shocks" problem as
    ConsIndShock, but with permanent income explicitly tracked as a state variable.
    Can't yet handle value function calculation, or cubic spline interpolation
    of the consumption function.
    '''
    def __init__(self,solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,Rfree,
                      PermGroFac,BoroCnstArt,aXtraGrid,pLvlGrid,vFuncBool,CubicBool):
        '''
        Constructor for a new solver-setup for problems with income subject to
        permanent and transitory shocks, with permanent income explicitly
        tracked as a state variable.
        
        Parameters
        ----------
        solution_next : ConsumerSolution
            The solution to next period's one period problem.
        IncomeDstn : [np.array]
            A list containing three arrays of floats, representing a discrete
            approximation to the income process between the period being solved
            and the one immediately following (in solution_next). Order: event
            probabilities, permanent shocks, transitory shocks.
        LivPrb : float
            Survival probability; likelihood of being alive at the beginning of
            the succeeding period.    
        DiscFac : float
            Intertemporal discount factor for future utility.        
        CRRA : float
            Coefficient of relative risk aversion.
        Rfree : float
            Risk free interest factor on end-of-period assets.
        PermGroGac : float
            Expected permanent income growth factor at the end of this period.
        BoroCnstArt: float or None
            Borrowing constraint for the minimum allowable assets to end the
            period with.
        aXtraGrid: np.array
            Array of "extra" end-of-period (normalized) asset values-- assets
            above the absolute minimum acceptable level.
        pLvlGrid: np.array
            Array of permanent income levels at which to solve the problem.
        vFuncBool: boolean
            An indicator for whether the value function should be computed and
            included in the reported solution.  Can't yet handle vFuncBool=True.
        CubicBool: boolean
            An indicator for whether the solver should use cubic or linear inter-
            polation.  Can't yet handle CubicBool=True.
                        
        Returns
        -------
        None
        '''
        self.assignParameters(solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,Rfree,
                                PermGroFac,BoroCnstArt,aXtraGrid,pLvlGrid,vFuncBool,CubicBool)
        self.defUtilityFuncs()
        
    def assignParameters(self,solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,Rfree,
                                PermGroFac,BoroCnstArt,aXtraGrid,pLvlGrid,vFuncBool,CubicBool):
        '''
        Assigns period parameters as attributes of self for use by other methods
        
        Parameters
        ----------
        solution_next : ConsumerSolution
            The solution to next period's one period problem.
        IncomeDstn : [np.array]
            A list containing three arrays of floats, representing a discrete
            approximation to the income process between the period being solved
            and the one immediately following (in solution_next). Order: event
            probabilities, permanent shocks, transitory shocks.
        LivPrb : float
            Survival probability; likelihood of being alive at the beginning of
            the succeeding period.    
        DiscFac : float
            Intertemporal discount factor for future utility.        
        CRRA : float
            Coefficient of relative risk aversion.
        Rfree : float
            Risk free interest factor on end-of-period assets.
        PermGroGac : float
            Expected permanent income growth factor at the end of this period.
        BoroCnstArt: float or None
            Borrowing constraint for the minimum allowable assets to end the
            period with.
        aXtraGrid: np.array
            Array of "extra" end-of-period (normalized) asset values-- assets
            above the absolute minimum acceptable level.
        pLvlGrid: np.array
            Array of permanent income levels at which to solve the problem.
        vFuncBool: boolean
            An indicator for whether the value function should be computed and
            included in the reported solution.  Can't yet handle vFuncBool=True.
        CubicBool: boolean
            An indicator for whether the solver should use cubic or linear inter-
            polation.  Can't yet handle CubicBool=True.
                        
        Returns
        -------
        none
        '''
        ConsIndShockSetup.assignParameters(self,solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,Rfree,
                                PermGroFac,BoroCnstArt,aXtraGrid,vFuncBool,CubicBool)
        self.pLvlGrid = pLvlGrid
        
    def defBoroCnst(self,BoroCnstArt):
        '''
        Defines the constrained portion of the consumption function as cFuncNowCnst,
        an attribute of self.
        
        Parameters
        ----------
        BoroCnstArt : float or None
            Borrowing constraint for the minimum allowable assets to end the
            period with.  If it is less than the natural borrowing constraint,
            then it is irrelevant; BoroCnstArt=None indicates no artificial bor-
            rowing constraint.
            
        Returns
        -------
        none
        '''
        # Everything is the same as base model except the constrained consumption function has to be 2D
        ConsIndShockSetup.defBoroCnst(self,BoroCnstArt)
        self.cFuncNowCnst = BilinearInterp(np.array([[0.0,-self.mNrmMinNow],[1.0,1.0-self.mNrmMinNow]]),
                                           np.array([0.0,1.0]),np.array([0.0,1.0]))
        #self.cFuncNowCnst = lambda mLvl,pLvl : mLvl - self.mNrmMinNow*pLvl # alternate version
                                         
    def prepareToCalcEndOfPrdvP(self):
        '''
        Prepare to calculate end-of-period marginal value by creating an array
        of market resources that the agent could have next period, considering
        the grid of end-of-period normalized assets, the grid of permanent income
        levels, and the distribution of shocks he might experience next period.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        aLvlNow : np.array
            2D array of end-of-period assets; also stored as attribute of self.
        pLvlNow : np.array
            2D array of permanent income levels this period.
        '''               
        ShkCount    = self.TranShkValsNext.size
        pLvlCount   = self.pLvlGrid.size
        aNrmCount   = self.aXtraGrid.size
        aNrmNow     = np.tile(np.asarray(self.aXtraGrid) + self.BoroCnstNat,(pLvlCount,1))
        pLvlNow     = np.tile(self.pLvlGrid,(aNrmCount,1)).transpose()
        aLvlNow     = aNrmNow*pLvlNow
        pLvlNow_tiled = np.tile(pLvlNow,(ShkCount,1,1))
        aLvlNow_tiled = np.tile(aLvlNow,(ShkCount,1,1)) # shape = (ShkCount,pLvlCount,aNrmCount)
        
        # Tile arrays of the income shocks and put them into useful shapes
        PermShkVals_tiled = np.transpose(np.tile(self.PermShkValsNext,(aNrmCount,pLvlCount,1)),(2,1,0))
        TranShkVals_tiled = np.transpose(np.tile(self.TranShkValsNext,(aNrmCount,pLvlCount,1)),(2,1,0))
        ShkPrbs_tiled     = np.transpose(np.tile(self.ShkPrbsNext,(aNrmCount,pLvlCount,1)),(2,1,0))
        
        # Get cash on hand next period
        pLvlNext = pLvlNow_tiled*PermShkVals_tiled*self.PermGroFac
        mLvlNext = self.Rfree*aLvlNow_tiled + pLvlNext*TranShkVals_tiled

        # Store and report the results
        self.ShkPrbs_temp      = ShkPrbs_tiled
        self.pLvlNext          = pLvlNext
        self.mLvlNext          = mLvlNext 
        self.aLvlNow           = aLvlNow               
        return aLvlNow, pLvlNow
        
    def calcEndOfPrdvP(self):
        '''
        Calculates end-of-period marginal value of assets at each state space
        point in ALvlNow x pLvlNow. Does so by taking a weighted sum of next
        period marginal values across income shocks (in preconstructed grids
        self.MLvlNext x self.pLvlNext).
        
        Parameters
        ----------
        none
        
        Returns
        -------
        EndOfPrdVP : np.array
            A 2D array of end-of-period marginal value of assets.
        '''
        EndOfPrdvP  = self.DiscFacEff*self.Rfree*np.sum(self.vPfuncNext(self.mLvlNext,self.pLvlNext)*self.ShkPrbs_temp,axis=0)  
        return EndOfPrdvP
    
    def getPointsForInterpolation(self,EndOfPrdvP,aLvlNow):
        '''
        Finds endogenous interpolation points (c,m) for the consumption function.
        
        Parameters
        ----------
        EndOfPrdvP : np.array
            Array of end-of-period marginal values.
        aLvlNow : np.array
            Array of end-of-period asset values that yield the marginal values
            in EndOfPrdvP.
            
        Returns
        -------
        c_for_interpolation : np.array
            Consumption points for interpolation.
        m_for_interpolation : np.array
            Corresponding market resource points for interpolation.
        '''
        cLvlNow = self.uPinv(EndOfPrdvP)
        mLvlNow = cLvlNow + aLvlNow

        # Limiting consumption is zero as m approaches mNrmMin
        c_for_interpolation = np.concatenate((np.zeros((self.pLvlGrid.size,1)),cLvlNow),axis=-1)
        m_for_interpolation = np.concatenate((self.BoroCnstNat*np.reshape(self.pLvlGrid,(self.pLvlGrid.size,1)),mLvlNow),axis=-1)
        
        # Limiting consumption is MPCmin*mLvl as p approaches 0
        m_temp = np.reshape(m_for_interpolation[0,:],(1,m_for_interpolation.shape[1]))
        m_for_interpolation = np.concatenate((m_temp,m_for_interpolation),axis=0)
        c_for_interpolation = np.concatenate((self.MPCminNow*m_temp,c_for_interpolation),axis=0)
        
        return c_for_interpolation, m_for_interpolation
        
    def usePointsForInterpolation(self,cLvl,mLvl,pLvl,interpolator):
        '''
        Constructs a basic solution for this period, including the consumption
        function and marginal value function.
        
        Parameters
        ----------
        cLvl : np.array
            Consumption points for interpolation.
        mLvl : np.array
            Corresponding market resource points for interpolation.
        pLvl : np.array
            Corresponding permanent income level points for interpolation.
        interpolator : function
            A function that constructs and returns a consumption function.
            
        Returns
        -------
        solution_now : ConsumerSolution
            The solution to this period's consumption-saving problem, with a
            consumption function, marginal value function, and minimum m.
        '''
        # Construct the unconstrained consumption function
        cFuncNowUnc = interpolator(mLvl,pLvl,cLvl)

        # Combine the constrained and unconstrained functions into the true consumption function
        cFuncNow = LowerEnvelope2D(cFuncNowUnc,self.cFuncNowCnst)

        # Make the marginal value function and the marginal marginal value function
        vPfuncNow = MargValueFunc2D(cFuncNow,self.CRRA)

        # Pack up the solution and return it
        solution_now = ConsumerSolution(cFunc=cFuncNow, vPfunc=vPfuncNow, mNrmMin=self.mNrmMinNow)
        return solution_now
        
    def makeBasicSolution(self,EndOfPrdvP,aLvl,pLvl,interpolator):
        '''
        Given end of period assets and end of period marginal value, construct
        the basic solution for this period.
        
        Parameters
        ----------
        EndOfPrdvP : np.array
            Array of end-of-period marginal values.
        aLvl : np.array
            Array of end-of-period asset values that yield the marginal values
            in EndOfPrdvP.
        pLvl : np.array
            Array of permanent income levels that yield the marginal values
            in EndOfPrdvP (corresponding pointwise to aLvl).            
        interpolator : function
            A function that constructs and returns a consumption function.
            
        Returns
        -------
        solution_now : ConsumerSolution
            The solution to this period's consumption-saving problem, with a
            consumption function, marginal value function, and minimum m.
        '''
        cLvl,mLvl    = self.getPointsForInterpolation(EndOfPrdvP,aLvl)
        pLvl_temp    = np.concatenate((np.reshape(self.pLvlGrid,(self.pLvlGrid.size,1)),pLvl),axis=-1)
        pLvl_temp    = np.concatenate((np.zeros((1,mLvl.shape[1])),pLvl_temp))
        solution_now = self.usePointsForInterpolation(cLvl,mLvl,pLvl_temp,interpolator)
        return solution_now
        
    def makeCurvilinearcFunc(self,mLvl,pLvl,cLvl):
        '''
        Makes a curvilinear interpolation to represent the (unconstrained)
        consumption function.
        
        Parameters
        ----------
        mLvl : np.array
            Market resource points for interpolation.
        cLvl : np.array
            Consumption points for interpolation.
        pLvl : np.array
            Permanent income level points for interpolation.
            
        Returns
        -------
        cFuncUnc : LinearInterp
            The unconstrained consumption function for this period.
        '''
        cFuncUnc = Curvilinear2DInterp(f_values=cLvl.transpose(),x_values=mLvl.transpose(),y_values=pLvl.transpose())
        return cFuncUnc
        
    def makeLinearcFunc(self,mLvl,pLvl,cLvl):
        '''
        Makes a quasi-bilinear interpolation to represent the (unconstrained)
        consumption function.
        
        Parameters
        ----------
        mLvl : np.array
            Market resource points for interpolation.
        cLvl : np.array
            Consumption points for interpolation.
        pLvl : np.array
            Permanent income level points for interpolation.
            
        Returns
        -------
        cFuncUnc : LinearInterp
            The unconstrained consumption function for this period.
        '''
        cFunc_by_pLvl_list = [] # list of consumption functions for each pLvl
        for j in range(pLvl.shape[0]):
            m_temp = mLvl[j,:]
            c_temp = cLvl[j,:] # Make a linear consumption function for this pLvl
            cFunc_by_pLvl_list.append(LinearInterp(m_temp,c_temp,lower_extrap=True))
        pLvl_list = pLvl[:,0]
        cFuncUnc = LinearInterpOnInterp1D(cFunc_by_pLvl_list,pLvl_list) # Combine all linear cFuncs
        return cFuncUnc
        
    def addMPCandHumanWealth(self,solution):
        '''
        Take a solution and add human wealth and the bounding MPCs to it.  This
        is identical to the version in ConsIndShockSolverBasic, but that version
        can't be called due to inheritance problems.
        
        Parameters
        ----------
        solution : ConsumerSolution
            The solution to this period's consumption-saving problem.
            
        Returns:
        ----------
        solution : ConsumerSolution
            The solution to this period's consumption-saving problem, but now
            with human wealth and the bounding MPCs.
        '''
        solution.hNrm   = self.hNrmNow
        solution.MPCmin = self.MPCminNow
        solution.MPCmax = self.MPCmaxEff
        return solution
        
    def solve(self):
        '''
        Solves a one period consumption saving problem with risky income, with
        permanent income explicitly tracked as a state variable.
        
        Parameters
        ----------
        None
            
        Returns
        -------
        solution : ConsumerSolution
            The solution to the one period problem, including a consumption
            function (defined over market resources and permanent income), a
            marginal value function, bounding MPCs, and normalized human wealth.
        '''
        aLvl,pLvl  = self.prepareToCalcEndOfPrdvP()           
        EndOfPrdvP = self.calcEndOfPrdvP()
        if self.mNrmMinNow == 0.0:   
            interpolator = self.makeLinearcFunc
        else: # Can use a faster solution method if lower bound of m is zero
            interpolator = self.makeCurvilinearcFunc
        solution   = self.makeBasicSolution(EndOfPrdvP,aLvl,pLvl,interpolator)
        solution   = self.addMPCandHumanWealth(solution)
        return solution
        
        
def solveConsIndShockExplicitPermInc(solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,Rfree,PermGroFac,
                                BoroCnstArt,aXtraGrid,pLvlGrid,vFuncBool,CubicBool):
    '''
    Solves the one period problem of a consumer who experiences permanent and
    transitory shocks to his income; the permanent income level is tracked as a
    state variable rather than normalized out as in ConsIndShock.
    
    Parameters
    ----------
    solution_next : ConsumerSolution
        The solution to next period's one period problem.
    IncomeDstn : [np.array]
        A list containing three arrays of floats, representing a discrete
        approximation to the income process between the period being solved
        and the one immediately following (in solution_next). Order: event
        probabilities, permanent shocks, transitory shocks.
    LivPrb : float
        Survival probability; likelihood of being alive at the beginning of
        the succeeding period.    
    DiscFac : float
        Intertemporal discount factor for future utility.        
    CRRA : float
        Coefficient of relative risk aversion.
    Rfree : float
        Risk free interest factor on end-of-period assets.
    PermGroGac : float
        Expected permanent income growth factor at the end of this period.
    BoroCnstArt: float or None
        Borrowing constraint for the minimum allowable assets to end the
        period with.  Currently ignored, with BoroCnstArt=0 used implicitly.
    aXtraGrid: np.array
        Array of "extra" end-of-period (normalized) asset values-- assets
        above the absolute minimum acceptable level.
    pLvlGrid: np.array
        Array of permanent income levels at which to solve the problem.
    vFuncBool: boolean
        An indicator for whether the value function should be computed and
        included in the reported solution.  Can't yet handle vFuncBool=True.
    CubicBool: boolean
        An indicator for whether the solver should use cubic or linear inter-
        polation.  Can't yet handle CubicBool=True.
                        
    Returns
    -------
    solution : ConsumerSolution
            The solution to the one period problem, including a consumption
            function (defined over market resources and permanent income), a
            marginal value function, bounding MPCs, and normalized human wealth.
    '''
    solver = ConsIndShockSolverExplicitPermInc(solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,Rfree,
                            PermGroFac,BoroCnstArt,aXtraGrid,pLvlGrid,vFuncBool,CubicBool)
    solver.prepareToSolve()       # Do some preparatory work
    solution_now = solver.solve() # Solve the period
    return solution_now
    
###############################################################################
    
class IndShockExplicitPermIncConsumerType(IndShockConsumerType):
    '''
    A consumer type with idiosyncratic shocks to permanent and transitory income.
    His problem is defined by a sequence of income distributions, survival prob-
    abilities, and permanent income growth rates, as well as time invariant values
    for risk aversion, discount factor, the interest rate, the grid of end-of-
    period assets, and an artificial borrowing constraint.  Identical to the
    IndShockConsumerType except that permanent income is tracked as a state
    variable rather than normalized out.
    '''
    cFunc_terminal_ = BilinearInterp(np.array([[0.0,0.0],[1.0,1.0]]),np.array([0.0,1.0]),np.array([0.0,1.0]))
    solution_terminal_ = ConsumerSolution(cFunc = cFunc_terminal_, mNrmMin=0.0, hNrm=0.0, MPCmin=1.0, MPCmax=1.0)
     
    def __init__(self,cycles=1,time_flow=True,**kwds):
        '''
        Instantiate a new ConsumerType with given data.
        See ConsumerParameters.make_this_dictionary for a dictionary of
        the keywords that should be passed to the constructor.
        
        Parameters
        ----------
        cycles : int
            Number of times the sequence of periods should be solved.
        time_flow : boolean
            Whether time is currently "flowing" forward for this instance.
        
        Returns
        -------
        None
        '''       
        # Initialize a basic ConsumerType
        IndShockConsumerType.__init__(self,cycles=cycles,time_flow=time_flow,**kwds)
        self.solveOnePeriod = solveConsIndShockExplicitPermInc # idiosyncratic shocks solver with explicit permanent income
        
    def update(self):
        '''
        Update the income process, the assets grid, the permanent income grid,
        and the terminal solution.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        IndShockConsumerType.update(self)
        self.updatePermIncGrid()
        
    def updateSolutionTerminal(self):
        '''
        Update the terminal period solution.  This method should be run when a
        new AgentType is created or when CRRA changes.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        self.solution_terminal.vPfunc = MargValueFunc2D(self.cFunc_terminal_,self.CRRA)
        
    def updatePermIncGrid(self):
        '''
        Update the grid of permanent income levels.  Currently only works for
        infinite horizon models (cycles=0) and lifecycle models (cycles=1).  Not
        clear what to do about cycles>1.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        if self.cycles == 1: 
            PermIncStdNow = self.PermIncStdInit # get initial distribution of permanent income
            PermIncAvgNow = self.PermIncAvgInit
            PermIncGrid = [] # empty list of time-varying permanent income grids
            # Calculate distribution of permanent income in each period of lifecycle
            for t in range(len(self.PermShkStd)):
                PermIncGrid.append(approxLognormal(mu=(np.log(PermIncAvgNow)-0.5*PermIncStdNow**2),
                                   sigma=PermIncStdNow, N=self.PermIncCount, tail_N=self.PermInc_tail_N, tail_bound=[0.05,0.95])[1])
                PermIncStdNow = np.sqrt(PermIncStdNow**2 + self.PermShkStd[t]**2)
                PermIncAvgNow = PermIncAvgNow*self.PermGroFac[t]
                
        # Calculate "stationary" distribution in infinite horizon (might vary across periods of cycle)
        elif self.cycles == 0:
            assert np.isclose(np.product(self.PermGroFac),1.0), "Long run permanent income growth not allowed!" 
            CumLivPrb     = np.product(self.LivPrb)
            CumDeathPrb   = 1.0 - CumLivPrb
            CumPermShkStd = np.sqrt(np.sum(np.array(self.PermShkStd)**2))
            ExPermShkSq   = np.exp(CumPermShkStd**2)
            ExPermIncSq   = CumDeathPrb/(1.0 - CumLivPrb*ExPermShkSq)
            PermIncStdNow = np.sqrt(np.log(ExPermIncSq))
            PermIncAvgNow = 1.0
            PermIncGrid = [] # empty list of time-varying permanent income grids
            # Calculate distribution of permanent income in each period of infinite cycle
            for t in range(len(self.PermShkStd)):
                PermIncGrid.append(approxLognormal(mu=(np.log(PermIncAvgNow)-0.5*PermIncStdNow**2),
                                   sigma=PermIncStdNow, N=self.PermIncCount, tail_N=self.PermInc_tail_N, tail_bound=[0.05,0.95])[1])
                PermIncStdNow = np.sqrt(PermIncStdNow**2 + self.PermShkStd[t]**2)
                PermIncAvgNow = PermIncAvgNow*self.PermGroFac[t]
        
        # Throw an error if cycles>1
        else:
            assert False, "Can only handle cycles=0 or cycles=1!"
            
        # Store the result and add attribute to time_vary
        orig_time = self.time_flow
        self.timeFwd()
        self.pLvlGrid = PermIncGrid
        self.addToTimeVary('pLvlGrid')
        if not orig_time:
            self.timeRev()
            
            
###############################################################################

if __name__ == '__main__':
    import ConsumerParameters as Params
    from time import clock
    import matplotlib.pyplot as plt
    mystr = lambda number : "{:.4f}".format(number)
    
    # Make and solve an example "explicit permanent income" consumer with idiosyncratic shocks
    ExplicitExample = IndShockExplicitPermIncConsumerType(**Params.init_explicit_perm_inc)
    
    t_start = clock()
    ExplicitExample.solve()
    t_end = clock()
    print('Solving an explicit permanent income consumer took ' + mystr(t_end-t_start) + ' seconds.')
    
    # Plot the consumption function at various permanent income levels
    pGrid = np.linspace(0.1,8,24)
    M = np.linspace(0,20,300)
    for p in pGrid:
        M_temp = M+p*ExplicitExample.solution[0].mNrmMin
        C = ExplicitExample.solution[0].cFunc(M_temp,p*np.ones_like(M_temp))
        plt.plot(M_temp,C)
    plt.show()
    