"""
Classes to solve consumption-savings model where the agent can invest on
the human capital formation of their child or save in a risky asset.
"""

import numpy as np

from HARK import MetricObject, NullFunc
from HARK.ConsumptionSaving.ConsGenIncProcessModel import (
    GenIncProcessConsumerType,
    ConsGenIncProcessSolver,
)
from HARK.ConsumptionSaving.ConsIndShockModel import ConsIndShockSolver
from HARK.distribution import calc_expectation
from HARK.interpolation import (
    MargValueFuncCRRA,
    TrilinearInterp,
    LinearInterp,
    BilinearInterp,
    LinearInterpOnInterp1D,
)
from HARK.utilities import (
    CRRAutility,
    CRRAutilityP,
    CRRAutilityP_inv,
    CRRAutility_inv,
    CRRAutility_invP,
)


class HumanCapitalProductionFunctionCES(MetricObject):
    """
    A class for representing the human capital production function of children,
    which takes as input a level of investment, the child's current level of
    skills, and the parent's current level of permanent income, which is a
    proxy of their own human capital
    """

    def __init__(self, tfp, i_share, k_share, p_share, ces):
        self.tfp = tfp
        self.i_share = i_share
        self.k_share = k_share
        self.p_share = p_share
        self.shares = np.array([i_share, k_share, p_share])
        self.ces = ces

    def _dprod(self, i_lvl, k_lvl, p_lvl):
        return np.dot(np.array(i_lvl, k_lvl, p_lvl).T ** self.ces, self.shares)

    def __call__(self, i_lvl, k_lvl, p_lvl):
        return self.tfp * self._dprod(i_lvl, k_lvl, p_lvl) ** (1.0 / self.ces)

    def derivativeI(self, i_lvl, k_lvl, p_lvl):
        return (
            self.tfp
            * self._dprod(i_lvl, k_lvl, p_lvl) ** (1.0 / self.ces - 1.0)
            * self.i_share
            * i_lvl ** (self.ces - 1.0)
        )

    def derivativeK(self, i_lvl, k_lvl, p_lvl):
        return (
            self.tfp
            * self._dprod(i_lvl, k_lvl, p_lvl) ** (1.0 / self.ces - 1.0)
            * self.k_share
            * k_lvl ** (self.ces - 1.0)
        )

    def derivatives(self, i_lvl, k_lvl, p_lvl):
        dprod = self._dprod(i_lvl, k_lvl, p_lvl)

        derivI = (
            self.tfp
            * dprod ** (1.0 / self.ces - 1.0)
            * self.i_share
            * i_lvl ** (self.ces - 1.0)
        )

        derivK = (
            self.tfp
            * dprod ** (1.0 / self.ces - 1.0)
            * self.k_share
            * k_lvl ** (self.ces - 1.0)
        )

        return derivI, derivK

    def eval_and_deriv(self, i_lvl, k_lvl, p_lvl):
        dprod = self._dprod(i_lvl, k_lvl, p_lvl)

        evaluate = self.tfp * dprod ** (1.0 / self.ces)

        derivativeI = (
            self.tfp
            * dprod ** (1.0 / self.ces - 1.0)
            * self.i_share
            * i_lvl ** (self.ces - 1.0)
        )

        derivativeK = (
            self.tfp
            * dprod ** (1.0 / self.ces - 1.0)
            * self.k_share
            * k_lvl ** (self.ces - 1.0)
        )

        return evaluate, derivativeI, derivativeK


class HumanCapitalProductionFunctionCD(MetricObject):
    def __init__(self, tfp, i_share, k_share):
        self.tfp = tfp
        self.i_share = i_share
        self.k_share = k_share


class ParentalSolution(MetricObject):
    """
    A class for representing the single period solution of the parental-child skills model.
    """

    def __init__(
        self,
        cFunc=NullFunc(),
        shareFunc=NullFunc(),
        vFunc=NullFunc(),
        vPfunc=NullFunc(),
        dvdmFunc=NullFunc(),
        dvdkFunc=NullFunc(),
    ):
        # set attributes of self
        self.cFunc = cFunc
        self.shareFunc = shareFunc
        self.vFunc = vFunc
        self.vPfunc = vPfunc
        self.dvdmFunc = dvdmFunc
        self.dvdkFunc = dvdkFunc


class ParentConsumerType(GenIncProcessConsumerType):
    pass


class ConsNrmIncParentSolver(ConsIndShockSolver):
    def __init__(
        self,
        solution_next,
        IncShkDstn,
        RiskyDstn,
        ShkDstn,
        DiscFac,
        CRRA,
        Rfree,
        BoroCnstArt,
        aXtraGrid,
        kNrmGrid,
        ShareGrid,
        vFuncBool,
        CubicBool,
    ):
        self.solution_next = solution_next
        self.IncShkDstn = IncShkDstn
        self.RiskyDstn = RiskyDstn
        self.ShkDstn = ShkDstn
        self.DiscFac = DiscFac
        self.CRRA = CRRA
        self.Rfree = Rfree
        self.BoroCnstArt = BoroCnstArt
        self.aXtraGrid = aXtraGrid
        self.kNrmGrid = kNrmGrid
        self.ShareGrid = ShareGrid
        self.vFuncBool = vFuncBool
        self.CubicBool = CubicBool

        self.def_utility_funcs()

    def def_utility_funcs(self):
        """
        Define temporary functions for utility and its derivative and inverse
        """

        self.u = lambda x: CRRAutility(x, self.CRRA)
        self.uP = lambda x: CRRAutilityP(x, self.CRRA)
        self.uPinv = lambda x: CRRAutilityP_inv(x, self.CRRA)
        self.uinv = lambda x: CRRAutility_inv(x, self.CRRA)
        self.uinvP = lambda x: CRRAutility_invP(x, self.CRRA)

    def set_and_update_values(self):
        """
        Unpacks some of the inputs (and calculates simple objects based on them),
        storing the results in self for use by other methods.
        """

        # Unpack next period's solution
        self.vPfunc_next = self.solution_next.vPfunc
        self.dvdmFunc_next = self.solution_next.dvdmFunc
        self.dvdkFunc_next = self.solution_next.dvdkFunc

        if self.vFuncBool:
            self.vFunc_next = self.solution_next.vFunc

        # Unpack the shock distribution
        TranShks = self.IncShkDstn.X[1]
        RiskyShks = self.RiskyDstn.X

        # Flag for whether the natural borrowing constraint is zero
        self.zero_bound = np.min(TranShks) == 0.0
        self.RiskyMax = np.max(RiskyShks)
        self.RiskyMin = np.min(RiskyShks)

        self.PermShkIdx = 0
        self.TranShkIdx = 1
        self.RiskyShkIdx = 0
        self.HumanShkIdx = 1

    def prepare_to_solve(self):
        """
        Perform preparatory work.
        """

        self.set_and_update_values()

    def prepare_to_calc_EndOfPrdvP(self):
        """
        Prepare to calculate end-of-period marginal values by creating an array
        of market resources that the agent could have next period, considering
        the grid of end-of-period assets and the distribution of shocks he might
        experience next period.
        """

        # bNrm represents R*a, balances after asset return shocks but before income.
        # This just uses the highest risky return as a rough shifter for the aXtraGrid.
        if self.zero_bound:
            self.aNrmGrid = self.aXtraGrid
            self.bNrmGrid = np.insert(
                self.RiskyMax * self.aXtraGrid, 0, self.RiskyMin * self.aXtraGrid[0]
            )
        else:
            # Add an asset point at exactly zero
            self.aNrmGrid = np.insert(self.aXtraGrid, 0, 0.0)
            self.bNrmGrid = self.RiskyMax * np.insert(self.aXtraGrid, 0, 0.0)

        # Get grid and shock sizes, for easier indexing
        self.aNrmCount = self.aNrmGrid.size
        self.kNrmCount = self.kNrmGrid.size
        self.ShareCount = self.ShareGrid.size

        # Make tiled arrays to calculate future realizations of mNrm and Share when integrating over IncShkDstn
        self.bNrmNext, self.kNrmNext = np.meshgrid(
            self.bNrmGrid, self.kNrmGrid, indexing="ij"
        )

        # Make tiled arrays to calculate future realizations of bNrm and Share when integrating over RiskyDstn
        self.aNrmNow, self.kNrmNow, self.shareNow = np.meshgrid(
            self.aNrmGrid, self.kNrmGrid, self.ShareGrid, indexing="ij"
        )

    def m_nrm_next(self, shocks, b_nrm_next):
        """
        Calculate future realizations of market resources
        """

        return (
            b_nrm_next / (shocks[self.PermShkIdx] * self.PermGroFac)
            + shocks[self.TranShkIdx]
        )

    def b_nrm_next(self, shocks, b_nrm):
        """
        Calculate future realizations of bank balances
        """

        return shocks[self.RiskyShkIdx] * b_nrm

    def k_nrm_next(self, shocks, i_nrm, k_nrm):
        """
        Calculate future realizations of human capital
        """

        return shocks[self.HumanShkIdx] * self.kNrmNextFunc(i_nrm, k_nrm)

    def dvdbFunc(self, shocks, b_nrm_next, k_nrm_next):
        """
        Evaluate realizations of marginal value of market resources next period
        """

        mNrm_next = self.m_nrm_next(shocks, b_nrm_next)
        dvdm = self.dvdmFunc_next(mNrm_next, k_nrm_next / shocks[self.PermShkIdx])

        return (shocks[self.PermShkIdx] * self.PermGroFac) ** (-self.CRRA) * dvdm

    def dvdkFunc(self, shocks, b_nrm_next, k_nrm_next):
        """
        Evaluate realizations of marginal value of risky share next period
        """

        mNrm_next = self.m_nrm_next(shocks, b_nrm_next)
        dvdk = self.dvdkFunc_next(mNrm_next, k_nrm_next / shocks[self.PermShkIdx])

        return (shocks[self.PermShkIdx] * self.PermGroFac) ** (-self.CRRA) * dvdk

    def EndOfPrddvdaFunc(self, shocks, a_nrm, k_nrm, share):

        b_nrm = a_nrm * (1.0 - share)
        i_nrm = a_nrm * share

        b_nrm_next = self.b_nrm_next(shocks, b_nrm)
        k_nrm_next = self.k_nrm_next(shocks, i_nrm, k_nrm)

        partial_b = (
            shocks[self.RiskyShkIdx]
            * (1.0 - share)
            * self.dvdbFunc_intermed(b_nrm_next, k_nrm_next)
        )

        partial_k = (
            shocks[self.HumanShkIdx]
            * share
            * self.kLvlNextFunc.derivativeI(i_nrm, k_nrm)
            * self.dvdkFunc_intermed(b_nrm_next, k_nrm_next)
        )

        return partial_b + partial_k

    def EndOfPrddvdsFunc(self, shocks, a_nrm, k_nrm, share):

        b_nrm = a_nrm * (1.0 - share)
        i_nrm = a_nrm * share

        b_nrm_next = self.b_nrm_next(shocks, b_nrm)
        k_nrm_next = self.k_nrm_next(shocks, i_nrm, k_nrm)

        partial_b = shocks[self.RiskyShkIdx] * self.dvdbFunc_intermed(
            b_nrm_next, k_nrm_next
        )

        partial_k = (
            shocks[self.HumanShkIdx]
            * self.kLvlNextFunc.derivativeI(i_nrm, k_nrm)
            * self.dvdkFunc_intermed(b_nrm_next, k_nrm_next)
        )

        return partial_b + partial_k

    def calc_EndOfPrdvP(self):
        """
        Calculate end-of-period marginal value of assets and shares at each point
        in aNrm and ShareGrid. Does so by taking expectation of next period marginal
        values across income and risky return shocks.
        """

        # Calculate intermediate marginal value of bank balances by taking expectations over income shocks
        dvdb_intermed = calc_expectation(
            self.IncShkDstn, self.dvdbFunc, self.bNrmNext, self.kNrmNext
        )
        # calc_expectation returns one additional "empty" dimension, remove it
        # this line can be deleted when calc_expectation is fixed
        dvdb_intermed = dvdb_intermed[:, :, 0]
        dvdbNvrs_intermed = self.uPinv(dvdb_intermed)
        dvdbNvrsFunc_intermed = BilinearInterp(
            dvdbNvrs_intermed, self.bNrmGrid, self.kNrmGrid
        )
        self.dvdbFunc_intermed = MargValueFuncCRRA(dvdbNvrsFunc_intermed, self.CRRA)

        # Calculate intermediate marginal value of risky portfolio share by taking expectations
        dvdk_intermed = calc_expectation(
            self.IncShkDstn, self.dvdkFunc, self.bNrmNext, self.kNrmNext
        )
        # calc_expectation returns one additional "empty" dimension, remove it
        # this line can be deleted when calc_expectation is fixed
        dvdk_intermed = dvdk_intermed[:, :, 0]
        self.dvdkFunc_intermed = BilinearInterp(
            dvdk_intermed, self.bNrmGrid, self.kNrmGrid
        )

        # Evaluate realizations of value and marginal value after asset returns are realized

        # Calculate end-of-period marginal value of assets by taking expectations
        self.EndOfPrddvda = self.DiscFac * calc_expectation(
            self.ShkDstn,  # includes RiskyShk and HumanShk
            self.EndOfPrddvdaFunc,  # constructed from intermed marginal value funcs
            self.aNrmNow,  # exogenous grid
            self.kNrmNow,  # grid of human capital
            self.shareNow,  # discrete shares
        )
        # calc_expectation returns one additional "empty" dimension, remove it
        # this line can be deleted when calc_expectation is fixed
        self.EndOfPrddvda = self.EndOfPrddvda[:, :, :, 0]
        self.cEGM = self.uPinv(self.EndOfPrddvda)

        # Calculate end-of-period marginal value of risky portfolio share by taking expectations
        self.EndOfPrddvds = calc_expectation(
            self.ShkDstn,
            self.EndOfPrddvdsFunc,
            self.aNrmNow,
            self.kNrmNow,
            self.shareNow,
        )
        # calc_expectation returns one additional "empty" dimension, remove it
        # this line can be deleted when calc_expectation is fixed
        self.EndOfPrddvds = self.EndOfPrddvds[:, :, :, 0]

    def optimize_share(self):
        """
        Optimization of Share on continuous interval [0,1]
        """

        # Initialize to putting everything in safe asset
        self.shareSoln = np.zeros((self.aNrmCount, self.kNrmCount))
        self.cNrmNow = np.zeros((self.aNrmCount, self.kNrmCount))

        # For values of aNrm at which the agent wants to put more than 100% into risky asset, constrain them
        FOCs = self.EndOfPrddvds

        # If agent wants to put more than 100% into risky asset, he is constrained
        constrained_top = FOCs[:, :, -1] > 0.0
        # Likewise if he wants to put less than 0% into risky asset
        constrained_bot = FOCs[:, :, 0] < 0.0
        self.shareSoln[constrained_top] = 1.0
        if not self.zero_bound:
            # aNrm=0, so there's no way to "optimize" the portfolio
            self.shareSoln[0] = 1.0
            # Consumption when aNrm=0 does not depend on Share
            self.cNrmNow[0] = self.EndOfPrddvdaNvrs[0, :, -1]
            # Mark as constrained so that there is no attempt at optimization
            constrained_top[0] = True

        # Get consumption when share-constrained
        self.cNrmNow[constrained_top] = self.cEGM[constrained_top, -1]
        self.cNrmNow[constrained_bot] = self.cEGM[constrained_bot, 0]
        # For each value of aNrm, find the value of Share such that FOC-Share == 0.
        # This loop can probably be eliminated, but it's such a small step that it won't speed things up much.
        crossing = np.logical_and(FOCs[:, :, 1:] <= 0.0, FOCs[:, :, :-1] >= 0.0)
        for i in range(self.aNrmCount):
            for j in range(self.kNrmCount):
                if not (constrained_top[i, j] or constrained_bot[i, j]):
                    idx = np.argwhere(crossing[i, j, :])[0][0]
                    bot_s = self.ShareGrid[idx]
                    top_s = self.ShareGrid[idx + 1]
                    bot_f = FOCs[i, j, idx]
                    top_f = FOCs[i, j, idx + 1]
                    bot_c = self.cEGM[i, j, idx]
                    top_c = self.cEGM[i, j, idx + 1]
                    alpha = 1.0 - top_f / (top_f - bot_f)
                    self.shareSoln[i, j] = (1.0 - alpha) * bot_s + alpha * top_s
                    self.cNrmNow[i, j] = (1.0 - alpha) * bot_c + alpha * top_c

    def dvdkNowFunc(self, shocks, a_nrm, k_nrm, share):

        b_nrm = a_nrm * (1.0 - share)
        i_nrm = a_nrm * share

        b_nrm_next = self.b_nrm_next(shocks, b_nrm)
        k_nrm_next = self.k_nrm_next(shocks, i_nrm, k_nrm)

        return (
            shocks[self.HumanShkIdx]
            * self.kLvlNextFunc.derivativeK(i_nrm, k_nrm)
            * self.dvdkFunc_intermed(b_nrm_next, k_nrm_next)
        )

    def make_basic_solution(self):
        """
        Given end of period assets and end of period marginal values, construct
        the basic solution for this period.
        """

        aNrmNow, kNrmNow = np.meshgrid(self.aNrmGrid, self.kNrmGrid, indexing="ij")

        # Calculate the endogenous mNrm gridpoints when the agent adjusts his portfolio
        mNrmNow = aNrmNow + self.cNrmNow

        dvdkNow = self.DiscFac * calc_expectation(
            self.ShkDstn, self.dvdkNowFunc, aNrmNow, kNrmNow, self.shareSoln
        )

        cFunc_by_k = []
        shareFunc_by_k = []
        dvdk_by_k = []
        for i in range(self.kNrmCount):
            cFunc_by_k.append(
                LinearInterp(
                    np.insert(mNrmNow[:, i], 0, 0.0),
                    np.insert(self.cNrmNow[:, i], 0, 0.0),
                )
            )

            shareFunc_by_k.append(LinearInterp(mNrmNow[:, i], self.shareNow[:, i]))

            dvdk_by_k.append(LinearInterp(mNrmNow[:, i], dvdkNow[:, i]))

        self.cFuncNow = LinearInterpOnInterp1D(cFunc_by_k, self.kNrmGrid)

        # Construct the marginal value (of mNrm) function when the agent can adjust
        self.vPfuncNow = MargValueFuncCRRA(self.cFuncNow, self.CRRA)

        self.dvdkFuncNow = LinearInterpOnInterp1D(dvdk_by_k, self.kNrmGrid)


class ConsGenIncParentSolver(ConsGenIncProcessSolver):
    def __init__(
        self,
        solution_next,
        IncShkDstn,
        RiskyDstn,
        ShkDstn,
        DiscFac,
        CRRA,
        Rfree,
        BoroCnstArt,
        aXtraGrid,
        kLvlGrid,
        pLvlGrid,
        vFuncBool,
        CubicBool,
    ):
        self.solution_next = solution_next
        self.IncShkDstn = IncShkDstn
        self.RiskyDstn = RiskyDstn
        self.ShkDstn = ShkDstn
        self.DiscFac = DiscFac
        self.CRRA = CRRA
        self.Rfree = Rfree
        self.BoroCnstArt = BoroCnstArt
        self.aXtraGrid = aXtraGrid
        self.kLvlGrid = kLvlGrid
        self.pLvlGrid = pLvlGrid
        self.vFuncBool = vFuncBool
        self.CubicBool = CubicBool

        self.def_utility_funcs()

    def def_utility_funcs(self):
        """
        Define tmporary functions for utility and its derivative and inverse
        """

        self.u = lambda x: CRRAutility(x, self.CRRA)
        self.uP = lambda x: CRRAutilityP(x, self.CRRA)
        self.uPinv = lambda x: CRRAutilityP_inv(x, self.CRRA)
        self.uinv = lambda x: CRRAutility_inv(x, self.CRRA)
        self.uinvP = lambda x: CRRAutility_invP(x, self.CRRA)

    def set_and_update_values(self):
        """
        Unpacks some of the inputs (and calculates simple objects based on them),
        storing the results in self for use by other methods.
        """

        # Unpack next period's solution
        self.vPfunc_next = self.solution_next.vPfunc
        self.dvdmFunc_next = self.solution_next.dvdmFunc
        self.dvdkFunc_next = self.solution_next.dvdkFunc

        if self.vFuncBool:
            self.vFunc_next = self.solution_next.vFunc

        self.RiskyMax = self.RiskyDstn.X.max()

        self.PermShkIdx = 0
        self.TranShkIdx = 1
        self.RiskyShkIdx = 0
        self.HumanShkIdx = 1

    def prepare_to_solve(self):
        """
        Perform preparatory work.
        """

        self.set_and_update_values()

    def p_lvl_next(self, shocks, p_lvl):
        """
        Calculates future realizations of permanent income.
        """
        return shocks[self.PermShkIdx] * self.pLvlNextFunc(p_lvl)

    def m_lvl_next(self, shocks, b_lvl_next, p_lvl_next):
        """
        Calculates future realizations of market resources.
        """
        return b_lvl_next + shocks[self.TranShkIdx] * p_lvl_next

    def b_lvl_next(self, shocks, b_lvl):
        """
        Calculates future realizations of bank balances.
        """
        return shocks[self.RiskyShkIdx] * b_lvl

    def k_lvl_next(self, shocks, i_lvl, k_lvl, p_lvl):
        """
        Calculates future realizations of human capital.
        """
        return shocks[self.HumanShkIdx] * self.kLvlNextFunc(i_lvl, k_lvl, p_lvl)

    def dvdbFunc(self, shocks, b_lvl_next, k_lvl_next, p_lvl):
        """
        Evaluate realization of marginal value of market resources next
        period with respect to choice of asset level and risky share.
        """
        p_lvl_next = self.p_lvl_next(shocks, p_lvl)
        m_lvl_next = self.m_lvl_next(shocks, b_lvl_next, p_lvl_next)

        return self.dvdmFunc_next(m_lvl_next, k_lvl_next, p_lvl_next)

    def dvdkFunc(self, shocks, b_lvl_next, k_lvl_next, p_lvl):
        """
        Evaluate realization of marginal value of market resources next
        period with respect to choice of asset level and risky share.
        """
        p_lvl_next = self.p_lvl_next(shocks, p_lvl)
        m_lvl_next = self.m_lvl_next(shocks, b_lvl_next, p_lvl_next)

        return self.dvdkFunc_next(m_lvl_next, k_lvl_next, p_lvl_next)

    def IntermedFuncs(self, shocks, b_lvl_next, k_lvl_next, p_lvl):

        p_lvl_next = self.p_lvl_next(shocks, p_lvl)
        m_lvl_next = self.m_lvl_next(shocks, b_lvl_next, p_lvl_next)

        dvdb = self.dvdmFunc_next(m_lvl_next, k_lvl_next, p_lvl_next)
        dvdk = self.dvdkFunc_next(m_lvl_next, k_lvl_next, p_lvl_next)

        return dvdb, dvdk

    def EndOfPrddvdaFunc(self, shocks, a_lvl, share, k_lvl, p_lvl):

        b_lvl = a_lvl * (1.0 - share)
        i_lvl = a_lvl * share

        b_lvl_next = self.b_lvl_next(shocks, b_lvl)
        k_lvl_next = self.k_lvl_next(shocks, i_lvl, k_lvl, p_lvl)

        dvda_partial_b = (
            shocks[self.RiskyShkIdx]
            * (1.0 - share)
            * self.dvdbFunc_intermed(b_lvl_next, k_lvl_next, p_lvl)
        )

        dvda_partial_k = (
            shocks[self.HumanShkIdx]
            * share
            * self.kLvlNextFunc.derivativeI(i_lvl, k_lvl, p_lvl)
            * self.dvdkFunc_intermed(b_lvl_next, k_lvl_next, p_lvl)
        )

        return dvda_partial_b + dvda_partial_k

    def EndOfPrddvdsFunc(self, shocks, a_lvl, share, k_lvl, p_lvl):

        b_lvl = a_lvl * (1.0 - share)
        i_lvl = a_lvl * share

        b_lvl_next = self.b_lvl_next(shocks, b_lvl)
        k_lvl_next = self.k_lvl_next(shocks, i_lvl, k_lvl, p_lvl)

        dvds_partial_b = shocks[self.RiskyShkIdx] * self.dvdbFunc_intermed(
            b_lvl_next, k_lvl_next, p_lvl
        )

        dvds_partial_k = (
            shocks[self.HumanShkIdx]
            * self.kLvlNextFunc.derivativeI(i_lvl, k_lvl, p_lvl)
            * self.dvdkFunc_intermed(b_lvl_next, k_lvl_next, p_lvl)
        )

        return dvds_partial_b + dvds_partial_k

    def EndOfPrdMarginalFuncs(self, shocks, a_lvl, share, k_lvl, p_lvl):

        b_lvl = a_lvl * (1.0 - share)
        i_lvl = a_lvl * share

        b_lvl_next = self.b_lvl_next(shocks, b_lvl)
        k_lvl_next = self.k_lvl_next(shocks, i_lvl, k_lvl, p_lvl)

        partial_b = shocks[self.RiskyShkIdx] * self.dvdbFunc_intermed(
            b_lvl_next, k_lvl_next, p_lvl
        )

        partial_k = (
            shocks[self.HumanShkIdx]
            * self.kLvlNextFunc.derivativeI(i_lvl, k_lvl, p_lvl)
            * self.dvdkFunc_intermed(b_lvl_next, k_lvl_next, p_lvl)
        )

        dvda = (1.0 - share) * partial_b + share * partial_k

        dvds = partial_b + partial_k

        return dvda, dvds

    def prepare_to_calc_EndOfPrdvP(self):

        # make sure to add limit when aXtraGrid = 0
        self.aLvlGrid = np.insert(self.aXtraGrid, 0, 0.0)
        # shift by RiskyMax assuming agent invests min share of 0
        self.bLvlGrid = self.aLvlGrid * self.RiskyMax

        self.aLvlCount = self.aXtraGrid + 1

        # 3D matrices shape (aLvlCount, kLvlCount, pLvlCount)
        self.aLvlNow3d, self.kLvlNow3d, self.pLvlNow3d = np.meshgrid(
            self.aLvlGrid, self.kLvlGrid, self.pLvlGrid
        )

        # shift by RiskyMax assuming agent invests min share of 0
        self.bLvlNext3d = self.aLvlNow3d * self.RiskyMax
        # kLvlNext is same grid as kLvlNow
        self.kLvlNext3d = self.kLvlNow3d

        # 4D matrices shape (aLvlCount, shareCount, kLvlCount, pLvlCount)
        self.aLvlNow4d, self.shareNow4d, self.kLvlNow4d, self.pLvlNow4d = np.meshgrid(
            self.aXtraGrid, self.shareGrid, self.kLvlGrid, self.pLvlGrid
        )

    def calc_EndOfPrdvP(self):
        # 2nd stage: Taking control variables as given, take expectations over
        # income distribution to obtain intermediate marginal value functions

        # dvdb is the intermediate marginal value function with respect to bank balances
        dvdb_intermed = calc_expectation(
            self.IncShkDstn,
            self.dvdbFunc,
            self.bLvlNext3d,
            self.kLvlNext3d,
            self.pLvlNow3d,
        )
        # calc_expectation returns one additional "empty" dimension, remove it
        # this line can be deleted when calc_expectation is fixed
        dvdb_intermed = dvdb_intermed[:, :, :, 0]
        dvdbNvrs_intermed = self.uPinv(dvdb_intermed)
        dvdbNvrsFunc_intermed = TrilinearInterp(
            dvdbNvrs_intermed, self.bLvlGrid, self.kLvlGrid, self.pLvlGrid
        )

        self.dvdbFunc_intermed = MargValueFuncCRRA(dvdbNvrsFunc_intermed, self.CRRA)

        # dvdk is the intermediate marginal value function with respect to human capital
        dvdk_intermed = calc_expectation(
            self.IncShkDstn, self.dvdkFunc, self.bLvlNext, self.kLvlNext, self.pLvlNow3d
        )
        # calc_expectation returns one additional "empty" dimension, remove it
        # this line can be deleted when calc_expectation is fixed
        dvdk_intermed = dvdk_intermed[:, :, :, 0]
        dvdkNvrs_intermed = self.uPinv(dvdk_intermed)
        dvdkNvrsFunc_intermed = TrilinearInterp(
            dvdkNvrs_intermed, self.bLvlGrid, self.kLvlGrid, self.pLvlGrid
        )

        self.dvdbFunc_intermed = MargValueFuncCRRA(dvdkNvrsFunc_intermed, self.CRRA)

        # Stage 1

        # Calculate end-of-period marginal value of assets by taking expectations
        self.EndOfPrddvda = self.DiscFac * calc_expectation(
            self.RiskyDstn,
            self.EndOfPrddvdaFunc,
            self.aLvlNow,
            self.shareNow,
            self.kLvlNow,
            self.pLvlNow4d,
        )
        # calc_expectation returns one additional "empty" dimension, remove it
        # this line can be deleted when calc_expectation is fixed
        self.EndOfPrddvda = self.EndOfPrddvda[:, :, :, 0]
        self.cEGM = self.uPinv(self.EndOfPrddvda)

        # Calculate end-of-period marginal value of risky portfolio share by taking expectations
        self.EndOfPrddvds = self.DiscFac * calc_expectation(
            self.RiskyDstn,
            self.EndOfPrddvdsFunc,
            self.aLvlNow,
            self.shareNow,
            self.kLvlNow,
            self.pLvlNow4d,
        )
        # calc_expectation returns one additional "empty" dimension, remove it
        # this line can be deleted when calc_expectation is fixed
        self.EndOfPrddvds = self.EndOfPrddvds[:, :, :, 0]

    def optimize_share(self):
        """
        Optimization of Share on continuous interval [0,1]
        """

        # For values of aNrm at which the agent wants to put more than 100% into risky asset, constrain them
        FOCs = self.EndOfPrddvds
        # Initialize to putting everything in safe asset
        self.shareNow = np.zeros((self.aXtraCount, self.kLvlCount, self.pLvlCount))
        self.cNow = np.zeros((self.aXtraCount, self.kLvlCount, self.pLvlCount))

        for k in self.kLvlCount:
            for p in self.pLvlCount:

                FOCs_tmp = FOCs[:, :, k, p]
                cEGM_tmp = self.cEGM[:, :, k, p]

                # If agent wants to put more than 100% into risky asset, he is constrained
                constrained_top = FOCs_tmp[:, -1] > 0.0
                # Likewise if he wants to put less than 0% into risky asset
                constrained_bot = FOCs_tmp[:, 0] < 0.0
                self.shareNow[constrained_top, k, p] = 1.0
                if not self.zero_bound:
                    # aNrm=0, so there's no way to "optimize" the portfolio
                    self.shareNow[0, k, p] = 1.0
                    # Consumption when aNrm=0 does not depend on Share
                    self.cNow[0, k, p] = cEGM_tmp[0, -1]
                    # Mark as constrained so that there is no attmpt at optimization
                    constrained_top[0] = True

                # Get consumption when share-constrained
                self.cNow[constrained_top, k, p] = cEGM_tmp[constrained_top, -1]
                self.cNow[constrained_bot, k, p] = cEGM_tmp[constrained_bot, 0]
                # For each value of aNrm, find the value of Share such that FOC-Share == 0.
                # This loop can probably be eliminated, but it's such a small step that it won't speed things up much.
                crossing = np.logical_and(
                    FOCs_tmp[:, 1:] <= 0.0, FOCs_tmp[:, :-1] >= 0.0
                )
                for j in range(self.aXtraCount):
                    if not (constrained_top[j] or constrained_bot[j]):
                        idx = np.argwhere(crossing[j, :])[0][0]
                        bot_s = self.shareGrid[idx]
                        top_s = self.shareGrid[idx + 1]
                        bot_f = FOCs_tmp[j, idx]
                        top_f = FOCs_tmp[j, idx + 1]
                        bot_c = cEGM_tmp[j, idx]
                        top_c = cEGM_tmp[j, idx + 1]
                        alpha = 1.0 - top_f / (top_f - bot_f)
                        self.shareNow[j, k, p] = (1.0 - alpha) * bot_s + alpha * top_s
                        self.cNow[j, k, p] = (1.0 - alpha) * bot_c + alpha * top_c

    def make_basic_solution(self):
        """
        Given end of period assets and end of period marginal values, construct
        the basic solution for this period.
        """

        # Calculate the endogenous mNrm gridprodoints when the agent adjusts his portfolio
        self.mLvlNow = self.aLvlNow + self.cNow

        # Construct the consumption function when the agent can adjust
        cNow = np.insert(self.cNow, 0, 0.0)
        self.cFunc_now = TrilinearInterp(
            cNow, self.mLvlNow, self.kLvlGrid, self.pLvlGrid
        )

        # Construct the marginal value (of mNrm) function when the agent can adjust
        self.vPfunc_now = MargValueFuncCRRA(self.cFunc_now, self.CRRA)

    def make_ShareFunc(self):
        """
        Construct the risky share function when the agent can adjust
        """

        if self.zero_bound:
            Share_lower_bound = self.shareLimit
        else:
            Share_lower_bound = 1.0
        Share_now = np.insert(self.shareNow, 0, Share_lower_bound)
        self.shareFunc_now = LinearInterp(
            np.insert(self.mLvlNow, 0, 0.0),
            Share_now,
            intercept_limit=self.shareLimit,
            slope_limit=0.0,
        )
