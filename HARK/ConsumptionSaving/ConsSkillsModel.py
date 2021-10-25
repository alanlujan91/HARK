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
from HARK.distribution import calc_expectation
from HARK.interpolation import MargValueFuncCRRA, TrilinearInterp
from HARK.utilities import (
    CRRAutility,
    CRRAutilityP,
    CRRAutilityP_inv,
    CRRAutility_inv,
    CRRAutility_invP,
)


class HumanCapitalProductionFuncCES(MetricObject):
    def __init__(self, constant, iLvlFac, kLvlFac, pLvlFac, cesFac):
        self.constant = constant
        self.iLvlFac = iLvlFac
        self.kLvlFac = kLvlFac
        self.pLvlFac = pLvlFac
        self.Factors = np.array([iLvlFac, kLvlFac, pLvlFac])
        self.cesFac = cesFac

    def __call__(self, i_lvl, k_lvl, p_lvl):
        return self.constant * np.dot(
            np.array(i_lvl, k_lvl, p_lvl).T, self.Factors
        ) ** (1 / self.cesFac)

    def derivativeI(self, i_lvl, k_lvl, p_lvl):
        prod = np.dot(np.array(i_lvl, k_lvl, p_lvl).T, self.Factors)
        return (
            self.constant
            * prod ** (1 / self.cesFac - 1)
            * self.iLvlFac
            * i_lvl ** (self.cesFac - 1)
        )

    def derivativeK(self, i_lvl, k_lvl, p_lvl):
        prod = np.dot(np.array(i_lvl, k_lvl, p_lvl).T, self.Factors)
        return (
            self.constant
            * prod ** (1 / self.cesFac - 1)
            * self.kLvlFac
            * k_lvl ** (self.cesFac - 1)
        )

    def derivatives(self, i_lvl, k_lvl, p_lvl):
        prod = np.dot(np.array(i_lvl, k_lvl, p_lvl).T, self.Factors)
        derivI = (
            self.constant
            * prod ** (1 / self.cesFac - 1)
            * self.iLvlFac
            * i_lvl ** (self.cesFac - 1)
        )

        derivK = (
            self.constant
            * prod ** (1 / self.cesFac - 1)
            * self.kLvlFac
            * k_lvl ** (self.cesFac - 1)
        )

        return derivI, derivK

    def eval_and_deriv(self, i_lvl, k_lvl, p_lvl):
        prod = np.dot(np.array(i_lvl, k_lvl, p_lvl).T, self.Factors)
        eval = self.constant * prod ** (1 / self.cesFac)
        derivI = (
            self.constant
            * prod ** (1 / self.cesFac - 1)
            * self.iLvlFac
            * i_lvl ** (self.cesFac - 1)
        )
        derivK = (
            self.constant
            * prod ** (1 / self.cesFac - 1)
            * self.kLvlFac
            * k_lvl ** (self.cesFac - 1)
        )

        return eval, derivI, derivK


class HumanCapitalProductionFuncCD(MetricObject):
    def __init__(self, constant, iLvlFac, kLvlFac, pLvlFac):
        self.constant = constant
        self.iLvlFac = iLvlFac
        self.kLvlFac = kLvlFac
        self.pLvlFac = pLvlFac


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


class ConsParentSolver(ConsGenIncProcessSolver):
    def __init__(
        self,
        solution_next,
        IncShkDstn,
        RiskyDstn,
        DiscFac,
        CRRA,
        Rfree,
        PermGroFac,
        BoroCnstArt,
        aXtraGrid,
        vFuncBool,
        CubicBool,
    ):
        self.solution_next = solution_next
        self.IncShkDstn = IncShkDstn
        self.RiskyDstn = RiskyDstn
        self.DiscFac = DiscFac
        self.CRRA = CRRA
        self.Rfree = Rfree
        self.PermGroFac = PermGroFac
        self.BoroCnstArt = BoroCnstArt
        self.aXtraGrid = aXtraGrid
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

        self.IncShkDstn = self.IncShkDstn
        self.ShkPrbsNext = self.IncShkDstn.pmf
        self.PermShkValsNext = self.IncShkDstn.X[0]
        self.TranShkValsNext = self.IncShkDstn.X[1]
        self.PermShkMinNext = np.min(self.PermShkValsNext)
        self.TranShkMinNext = np.min(self.TranShkValsNext)
        self.WorstIncPrb = np.sum(
            self.ShkPrbsNext[
                (self.PermShkValsNext * self.TranShkValsNext)
                == (self.PermShkMinNext * self.TranShkMinNext)
            ]
        )

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
        The 0th shock is psi permanent shock to income.
        """
        return self.PermGroFac * shocks[self.PermShkIdx] * self.pLvlNextFunc(p_lvl)

    def m_lvl_next(self, shocks, b_lvl, p_lvl_next):
        """
        Calculates future realizations of market resources.
        The 1st shock is tsi temporary shock to income.
        """

        return b_lvl + shocks[self.TranShkIdx] * p_lvl_next

    def b_lvl_next(self, shocks, b_lvl):
        return shocks[self.RiskyShkIdx] * b_lvl

    def k_lvl_next(self, shocks, i_lvl):
        """
        Calculates future realizations of human capital.
        """
        return shocks[self.HumanShkIdx] * self.kLvlNextFunc(
            i_lvl, self.pLvlNow, self.kLvlNow
        )

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

    def EndOfPrddvdaFunc(self, shocks, a_lvl, share, p_lvl):
        b_lvl_next = a_lvl * (1 - share)
        i_lvl = a_lvl * share
        k_lvl_next = self.k_lvl_next(shocks, i_lvl)

        dvda_partial_b = shocks[self.RiskyShkIdx] * self.dvdbFunc_intermed(
            b_lvl_next, k_lvl_next, p_lvl
        )

        dvda_partial_k = (
            shocks[self.HumanShkIdx]
            * self.kLvlNextFunc.derivativeK(i_lvl, k_lvl, p_lvl)
            * self.dvdkFunc_intermed(b_lvl_next, k_lvl_next, p_lvl)
        )

        return (1 - share) * dvda_partial_b + share * dvda_partial_k

    def EndOfPrddvdsFunc(self, shocks, a_lvl, share):
        b_lvl_next = a_lvl * (1 - share)
        i_lvl = a_lvl * share
        k_lvl_next = self.k_lvl_next(shocks, i_lvl)

        dvds_partial_b = shocks[self.RiskyShkIdx] * self.dvdbFunc_intermed(
            b_lvl_next, k_lvl_next, p_lvl
        )

        dvds_partial_k = (
            shocks[self.HumanShkIdx]
            * self.kLvlNextFunc.derivativeK(i_lvl, k_lvl, p_lvl)
            * self.dvdkFunc_intermed(b_lvl_next, k_lvl_next, p_lvl)
        )

        return dvds_partial_b + share * dvds_partial_k

    def EndOfPrdPartialsFunc(self, shocks, a_lvl, share):
        b_lvl_next = a_lvl * (1 - share)
        i_lvl = a_lvl * share
        k_lvl_next = self.k_lvl_next(shocks, i_lvl)

        partial_b = shocks[self.RiskyShkIdx] * self.dvdbFunc_intermed(
            b_lvl_next, k_lvl_next, p_lvl
        )

        partial_k = (
            shocks[self.HumanShkIdx]
            * self.kLvlNextFunc.derivativeK(i_lvl, k_lvl, p_lvl)
            * self.dvdkFunc_intermed(b_lvl_next, k_lvl_next, p_lvl)
        )

        dvda = (1 - share) * partial_b + share * partial_k

        dvds = partial_b + partial_k

        return dvda, dvds

    def prepare_to_calc_EndOfPrdvP(self):
        # shift by Rfree assuming agent invests min share of 0
        # make sure to add limit when aXtraGrid = 0
        self.bLvlGrid = self.aXtraGrid * self.Rfree

        bLvlNow, kLvlNow, pLvlNow = np.meshgrid(
            self.bLvlGrid, self.kLvlGrid, self.pLvlGrid
        )

        self.bLvlNext = bLvlNow
        # for kLvl, assume same grid for kLvlNow and kLvlNext
        self.kLvlNext = kLvlNow
        self.pLvlNow = pLvlNow

    def calc_EndOfPrdvP(self):
        # 2nd stage: Taking control variables as given, take expectations over
        # income distribution to obtain intermediate marginal value functions

        # dvdb is the intermediate marginal value function with respect to bank balances
        dvdb_intermed = calc_expectation(
            self.IncShkDstn, self.dvdbFunc, self.bLvlNext, self.kLvlNext
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
            self.IncShkDstn, self.dvdkFunc, self.bLvlNext, self.kLvlNext
        )
        # calc_expectation returns one additional "empty" dimension, remove it
        # this line can be deleted when calc_expectation is fixed
        dvdk_intermed = dvdk_intermed[:, :, :, 0]
        dvdbNvrs_intermed = self.uPinv(dvdb_intermed)
        dvdbNvrsFunc_intermed = TrilinearInterp(
            dvdbNvrs_intermed, self.bLvlGrid, self.kLvlGrid, self.pLvlGrid
        )

        self.dvdbFunc_intermed = MargValueFuncCRRA(dvdbNvrsFunc_intermed, self.CRRA)

        # Stage 1

        # Calculate end-of-period marginal value of assets by taking expectations
        self.EndOfPrddvda = self.DiscFac * calc_expectation(
            self.RiskyDstn, self.EndOfPrddvdaFunc, self.aNrm_tiled, self.Share_tiled
        )
        # calc_expectation returns one additional "empty" dimension, remove it
        # this line can be deleted when calc_expectation is fixed
        self.EndOfPrddvda = self.EndOfPrddvda[:, :, :, 0]
        self.EndOfPrddvdaNvrs = self.uPinv(self.EndOfPrddvda)

        # Calculate end-of-period marginal value of risky portfolio share by taking expectations
        self.EndOfPrddvds = self.DiscFac * calc_expectation(
            self.RiskyDstn, self.EndOfPrddvdsFunc, self.aNrm_tiled, self.Share_tiled
        )
        # calc_expectation returns one additional "empty" dimension, remove it
        # this line can be deleted when calc_expectation is fixed
        self.EndOfPrddvds = self.EndOfPrddvds[:, :, :, 0]

    def optimize_share(self):
        """
        Optimization of Share on continuous interval [0,1]
        """

        # For values of aNrm at which the agent wants to put more than 100% into risky asset, constrain them
        FOC_s = self.EndOfPrddvds
        # Initialize to putting everything in safe asset
        self.Share_now = np.zeros_like(self.aNrmGrid)
        self.cNrmAdj_now = np.zeros_like(self.aNrmGrid)
        # If agent wants to put more than 100% into risky asset, he is constrained
        constrained_top = FOC_s[:, -1] > 0.0
        # Likewise if he wants to put less than 0% into risky asset
        constrained_bot = FOC_s[:, 0] < 0.0
        self.Share_now[constrained_top] = 1.0
        if not self.zero_bound:
            # aNrm=0, so there's no way to "optimize" the portfolio
            self.Share_now[0] = 1.0
            # Consumption when aNrm=0 does not depend on Share
            self.cNrmAdj_now[0] = self.EndOfPrddvdaNvrs[0, -1]
            # Mark as constrained so that there is no attempt at optimization
            constrained_top[0] = True

        # Get consumption when share-constrained
        self.cNrmAdj_now[constrained_top] = self.EndOfPrddvdaNvrs[constrained_top, -1]
        self.cNrmAdj_now[constrained_bot] = self.EndOfPrddvdaNvrs[constrained_bot, 0]
        # For each value of aNrm, find the value of Share such that FOC-Share == 0.
        # This loop can probably be eliminated, but it's such a small step that it won't speed things up much.
        crossing = np.logical_and(FOC_s[:, 1:] <= 0.0, FOC_s[:, :-1] >= 0.0)
        for j in range(self.aNrm_N):
            if not (constrained_top[j] or constrained_bot[j]):
                idx = np.argwhere(crossing[j, :])[0][0]
                bot_s = self.ShareGrid[idx]
                top_s = self.ShareGrid[idx + 1]
                bot_f = FOC_s[j, idx]
                top_f = FOC_s[j, idx + 1]
                bot_c = self.EndOfPrddvdaNvrs[j, idx]
                top_c = self.EndOfPrddvdaNvrs[j, idx + 1]
                alpha = 1.0 - top_f / (top_f - bot_f)
                self.Share_now[j] = (1.0 - alpha) * bot_s + alpha * top_s
                self.cNrmAdj_now[j] = (1.0 - alpha) * bot_c + alpha * top_c

    def make_basic_solution(self):
        """
        Given end of period assets and end of period marginal values, construct
        the basic solution for this period.
        """

        # Calculate the endogenous mNrm gridpoints when the agent adjusts his portfolio
        self.mNrmAdj_now = self.aNrmGrid + self.cNrmAdj_now

        # Construct the consumption function when the agent can adjust
        cNrmAdj_now = np.insert(self.cNrmAdj_now, 0, 0.0)
        self.cFuncAdj_now = LinearInterp(
            np.insert(self.mNrmAdj_now, 0, 0.0), cNrmAdj_now
        )

        # Construct the marginal value (of mNrm) function when the agent can adjust
        self.vPfuncAdj_now = MargValueFuncCRRA(self.cFuncAdj_now, self.CRRA)

        # Construct the consumption function when the agent *can't* adjust the risky share, as well
        # as the marginal value of Share function
        cFuncFxd_by_Share = []
        dvdsFuncFxd_by_Share = []
        for j in range(self.Share_N):
            cNrmFxd_temp = self.EndOfPrddvdaNvrs[:, j]
            mNrmFxd_temp = self.aNrmGrid + cNrmFxd_temp
            cFuncFxd_by_Share.append(
                LinearInterp(
                    np.insert(mNrmFxd_temp, 0, 0.0), np.insert(cNrmFxd_temp, 0, 0.0)
                )
            )
            dvdsFuncFxd_by_Share.append(
                LinearInterp(
                    np.insert(mNrmFxd_temp, 0, 0.0),
                    np.insert(self.EndOfPrddvds[:, j], 0, self.EndOfPrddvds[0, j]),
                )
            )
        self.cFuncFxd_now = LinearInterpOnInterp1D(cFuncFxd_by_Share, self.ShareGrid)
        self.dvdsFuncFxd_now = LinearInterpOnInterp1D(
            dvdsFuncFxd_by_Share, self.ShareGrid
        )

        # The share function when the agent can't adjust his portfolio is trivial
        self.ShareFuncFxd_now = IdentityFunction(i_dim=1, n_dims=2)

        # Construct the marginal value of mNrm function when the agent can't adjust his share
        self.dvdmFuncFxd_now = MargValueFuncCRRA(self.cFuncFxd_now, self.CRRA)

    def make_ShareFuncAdj(self):
        """
        Construct the risky share function when the agent can adjust
        """

        if self.zero_bound:
            Share_lower_bound = self.ShareLimit
        else:
            Share_lower_bound = 1.0
        Share_now = np.insert(self.Share_now, 0, Share_lower_bound)
        self.ShareFuncAdj_now = LinearInterp(
            np.insert(self.mNrmAdj_now, 0, 0.0),
            Share_now,
            intercept_limit=self.ShareLimit,
            slope_limit=0.0,
        )
