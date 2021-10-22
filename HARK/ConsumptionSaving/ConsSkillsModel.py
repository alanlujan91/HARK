"""
Classes to solve consumption-savings model where the agent can invest on
the human capital formation of their offspring.
Extends ConsGenIncProcessModel.py by adding
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


class HumanCapitalProdFunc(MetricObject):
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

    def prepare_to_solve(self):
        """
        Perform preparatory work.
        """

        self.set_and_update_values()

    def p_lvl_next(self, psi, p_lvl):
        """
        Calculates future realizations of permanent income.
        The 0th shock is psi permanent shock to income.
        """
        return self.PermGroFac * psi[0] * self.pLvlNextFunc(p_lvl)

    def m_lvl_next(self, tsi, b_lvl, p_lvl_next):
        """
        Calculates future realizations of market resources.
        The 1st shock is tsi temporary shock to income.
        """

        return b_lvl + tsi[1] * p_lvl_next

    def k_lvl_next(self, shk, i_lvl):
        """
        Calculates future realizations of human capital.
        """
        return shk * self.kLvlFuncNext(i_lvl, self.pLvlNow, self.kLvlNow)

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

    def prepare_to_calc_EndOfPrdvP(self):
        # shift by Rfree assuming agent invests min share of 0
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

        dvdbFunc_intermed = MargValueFuncCRRA(dvdbNvrsFunc_intermed, self.CRRA)

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

        dvdbFunc_intermed = MargValueFuncCRRA(dvdbNvrsFunc_intermed, self.CRRA)
