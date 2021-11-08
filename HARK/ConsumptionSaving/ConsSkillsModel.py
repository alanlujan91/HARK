"""
Classes to solve consumption-savings model where the agent can invest on
the human capital formation of their child or save in a risky asset.
"""

import numpy as np

from HARK import MetricObject, NullFunc
from HARK.ConsumptionSaving.ConsGenIncProcessModel import (
    ConsGenIncProcessSolver,
)
from HARK.ConsumptionSaving.ConsIndShockModel import ConsIndShockSolver
from HARK.ConsumptionSaving.ConsPortfolioModel import PortfolioConsumerType
from HARK.distribution import (
    calc_expectation,
    IndexDistribution,
    Lognormal,
    combine_indep_dstns,
)
from HARK.interpolation import (
    MargValueFuncCRRA,
    TrilinearInterp,
    LinearInterp,
    BilinearInterp,
    LinearInterpOnInterp1D,
    ValueFuncCRRA,
    CubicInterp,
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


class ParentConsumerType(PortfolioConsumerType):
    """
    A consumer type with an investment choice. This agent can either invest on a risky asset
    or invest on the human capital production of their children.
    """

    state_vars = PortfolioConsumerType.state_vars + ["kLvl"]

    def __init__(self, verbose=False, quiet=False, **kwds):
        params = init_parent.copy()
        params.update(kwds)
        kwds = params

        # Initialize a basic consumer type
        PortfolioConsumerType.__init__(self, verbose=verbose, quiet=quiet, **kwds)

        if self.SubsParam == 0.0:
            solver = ConsNrmIncParentSolver
        else:
            solver = ConsGenIncParentSolver

        self.solve_one_period = solver

    def update_KaptlNextFunc(self):

        if (
            (
                type(self.TotlFactrProd) is list
                and len(self.TotlFactrProd) == self.T_cycle
            )
            and (type(self.InvstShare) is list and len(self.InvstShare) == self.T_cycle)
            and (type(self.KaptlShare) is list and len(self.KaptlShare) == self.T_cycle)
        ):
            self.add_to_time_vary("kLvlNextFunc")
            self.kLvlNextFunc = [
                HumanCapitalProductionFunctionCD(
                    self.TotlFactrProd[t], self.InvstShare[t], self.KaptlShare[t]
                )
                for t in range(self.T_cycle)
            ]
        elif (
            type(self.TotlFactrProd) is list
            or type(self.InvstShare) is list
            or type(self.KaptlShare) is list
        ):
            raise AttributeError(
                "If Human Capital production function is time-varying, then TotlFactrProd, InvstShare, and KaptlShare must be as well, and they must all have length of T_cycle!"
            )

    def update_kLvlGrid(self):
        pass

    def update_HumanDstn(self):
        """
        Creates the attributes HumanDstn from the primitive attributes HumanAvg,
        HumanStd, and HumanCount, approximating the (perceived) distribution of
        shocks to human capital.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Determine whether this instance has time-varying risk perceptions
        if (
            (type(self.HumanAvg) is list)
            and (type(self.HumanStd) is list)
            and (len(self.HumanAvg) == len(self.HumanStd))
            and (len(self.HumanAvg) == self.T_cycle)
        ):
            self.add_to_time_vary("HumanAvg", "HumanStd")
        elif (type(self.HumanStd) is list) or (type(self.HumanAvg) is list):
            raise AttributeError(
                "If HumanAvg is time-varying, then HumanStd must be as well, and they must both have length of T_cycle!"
            )
        else:
            self.add_to_time_inv("HumanAvg", "HumanStd")

        # Generate a discrete approximation to the Human return distribution if the
        # agent has age-varying beliefs about the Human asset
        if "HumanAvg" in self.time_vary:
            self.HumanDstn = IndexDistribution(
                Lognormal.from_mean_std,
                {"mean": self.HumanAvg, "std": self.HumanStd},
                seed=self.RNG.randint(0, 2 ** 31 - 1),
            ).approx(self.HumanCount)

            self.add_to_time_vary("HumanDstn")

        # Generate a discrete approximation to the Human return distribution if the
        # agent does *not* have age-varying beliefs about the Human asset (base case)
        else:
            self.HumanDstn = Lognormal.from_mean_std(
                self.HumanAvg,
                self.HumanStd,
            ).approx(self.HumanCount)
            self.add_to_time_inv("HumanDstn")

    def update_ShockDstn(self):
        """
        Combine the Risky return distribution (RiskyDstn) with the
        Human shock distribution (HumanDstn) to make a new attribute called ShockDstn.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if "HumanDstn" and "RiskyDstn" in self.time_vary:
            self.ShockDstn = [
                combine_indep_dstns(self.RiskyDstn[t], self.HumanDstn[t])
                for t in range(self.T_cycle)
            ]
        elif "RiskyDstn" in self.time_vary:
            self.ShockDstn = [
                combine_indep_dstns(self.RiskyDstn[t], self.HumanDstn)
                for t in range(self.T_cycle)
            ]
        elif "HumanDstn" in self.time_vary:
            self.ShockDstn = [
                combine_indep_dstns(self.RiskyDstn, self.HumanDstn[t])
                for t in range(self.T_cycle)
            ]
        else:
            self.ShockDstn = combine_indep_dstns(self.RiskyDstn, self.HumanDstn)
        self.add_to_time_vary("ShockDstn")


class ConsNrmIncParentSolver(ConsIndShockSolver):
    """
    Solver class for ParentConsumerType when solution can be normalized by permanent income.
    """

    def __init__(
        self,
        solution_next,
        IncShkDstn,
        ShkDstn,
        DiscFac,
        CRRA,
        Rfree,
        BoroCnstArt,
        kNrmNextFunc,
        aXtraGrid,
        kNrmGrid,
        ShareGrid,
        vFuncBool,
        CubicBool,
    ):
        self.solution_next = solution_next
        self.IncShkDstn = IncShkDstn
        self.ShkDstn = ShkDstn
        self.DiscFac = DiscFac
        self.CRRA = CRRA
        self.Rfree = Rfree
        self.BoroCnstArt = BoroCnstArt
        self.kNrmNextFunc = kNrmNextFunc
        self.aXtraGrid = aXtraGrid
        self.kNrmGrid = kNrmGrid
        self.ShareGrid = ShareGrid
        self.vFuncBool = vFuncBool
        self.CubicBool = CubicBool

        # Make sure the individual is liquidity constrained.  Allowing a consumer to
        # borrow *and* invest in an asset with unbounded (negative) returns is a bad mix.
        if BoroCnstArt != 0.0:
            raise ValueError("ParentConsumerType must have BoroCnstArt=0.0!")

        # Make sure that if risky portfolio share is optimized only discretely, then
        # the value function is also constructed (else this task would be impossible).
        if not vFuncBool:
            raise ValueError("ParentConsumerType requires vFuncBool to be True!")

        if CubicBool:
            raise NotImplementedError(
                "ParentConsumerType does not have a cubic cFunc option yet!"
            )

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

        self.PermShkIdx = 0
        self.TranShkIdx = 1
        self.RiskyShkIdx = 0
        self.HumanShkIdx = 1

        # Unpack the shock distribution
        TranShks = self.IncShkDstn.X[self.TranShkIdx]
        RiskyShks = self.ShkDstn.X[self.RiskyShkIdx]

        # Flag for whether the natural borrowing constraint is zero
        self.zero_bound = np.min(TranShks) == 0.0
        self.RiskyMax = np.max(RiskyShks)
        self.RiskyMin = np.min(RiskyShks)

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
        self.aNrmCount = self.aNrmGrid.size  # same as bNrmGrid size
        self.kNrmCount = self.kNrmGrid.size
        self.ShareCount = self.ShareGrid.size

        # Make tiled arrays to calculate future realizations of mNrm and kNrm when integrating over IncShkDstn
        # shape of these is (aNrmCount, kNrmCount)
        self.bNrmNext, self.kNrmNext = np.meshgrid(
            self.bNrmGrid, self.kNrmGrid, indexing="ij"
        )

        # Make tiled arrays to calculate future realizations of bNrm and kNrm when integrating over ShkDstn
        # shape of these is (aNrmCount, kNrmCount, ShareCount)
        self.aNrmNow, self.kNrmNow, self.shareNow = np.meshgrid(
            self.aNrmGrid, self.kNrmGrid, self.ShareGrid, indexing="ij"
        )

        # for solution objects we only need arrays of shape (aNrmCount, kNrmCount)
        # so we can remove the last dimension since share is a control not a state
        self.aNrmSoln = self.aNrmNow[:, :, 0]
        self.kNrmSoln = self.kNrmNow[:, :, 0]

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

    def dvFrak_db_helper(self, shocks, b_nrm_next, k_nrm_next):
        """
        Evaluate realizations of marginal value of market resources next period
        """

        mNrm_next = self.m_nrm_next(shocks, b_nrm_next)
        dvdm = self.dvdmFunc_next(mNrm_next, k_nrm_next / shocks[self.PermShkIdx])

        return (shocks[self.PermShkIdx] * self.PermGroFac) ** (-self.CRRA) * dvdm

    def dvFrak_dk_helper(self, shocks, b_nrm_next, k_nrm_next):
        """
        Evaluate realizations of marginal value of risky share next period
        """

        mNrm_next = self.m_nrm_next(shocks, b_nrm_next)
        dvdk = self.dvdkFunc_next(mNrm_next, k_nrm_next / shocks[self.PermShkIdx])

        return (shocks[self.PermShkIdx] * self.PermGroFac) ** (-self.CRRA) * dvdk

    def EndOfPrddvda_helper(self, shocks, a_nrm, k_nrm, share):

        b_nrm = a_nrm * (1.0 - share)
        i_nrm = a_nrm * share

        b_nrm_next = self.b_nrm_next(shocks, b_nrm)
        k_nrm_next = self.k_nrm_next(shocks, i_nrm, k_nrm)

        partial_b = (
            shocks[self.RiskyShkIdx]
            * (1.0 - share)
            * self.dvFrak_dbFunc(b_nrm_next, k_nrm_next)
        )

        partial_k = (
            shocks[self.HumanShkIdx]
            * share
            * self.kLvlNextFunc.derivativeI(i_nrm, k_nrm)
            * self.dvFrak_dkFunc(b_nrm_next, k_nrm_next)
        )

        return partial_b + partial_k

    def EndOfPrddvds_helper(self, shocks, a_nrm, k_nrm, share):

        b_nrm = a_nrm * (1.0 - share)
        i_nrm = a_nrm * share

        b_nrm_next = self.b_nrm_next(shocks, b_nrm)
        k_nrm_next = self.k_nrm_next(shocks, i_nrm, k_nrm)

        partial_b = shocks[self.RiskyShkIdx] * self.dvFrak_dbFunc(
            b_nrm_next, k_nrm_next
        )

        partial_k = (
            shocks[self.HumanShkIdx]
            * self.kLvlNextFunc.derivativeI(i_nrm, k_nrm)
            * self.dvFrak_dkFunc(b_nrm_next, k_nrm_next)
        )

        return partial_b + partial_k

    def calc_EndOfPrdvP(self):
        """
        Calculate end-of-period marginal value of assets and shares at each point
        in aNrm and ShareGrid. Does so by taking expectation of next period marginal
        values across income and risky return shocks.
        """

        # Calculate intermediate marginal value of bank balances by taking expectations over income shocks
        dvFrak_db = calc_expectation(
            self.IncShkDstn, self.dvFrak_db_helper, self.bNrmNext, self.kNrmNext
        )

        dvFrak_db = dvFrak_db[:, :, 0]
        dvFrak_dbNvrs = self.uPinv(dvFrak_db)
        dvFrak_dbNvrsFunc = BilinearInterp(dvFrak_dbNvrs, self.bNrmGrid, self.kNrmGrid)
        self.dvFrak_dbFunc = MargValueFuncCRRA(dvFrak_dbNvrsFunc, self.CRRA)

        # Calculate intermediate marginal value of risky portfolio share by taking expectations
        dvFrak_dk = calc_expectation(
            self.IncShkDstn, self.dvFrak_dk_helper, self.bNrmNext, self.kNrmNext
        )

        dvFrak_dk = dvFrak_dk[:, :, 0]
        dvFrak_dkNvrs = self.uPinv(dvFrak_dk)
        dvFrak_dkFunc = BilinearInterp(dvFrak_dkNvrs, self.bNrmGrid, self.kNrmGrid)
        self.dvFrak_dkFunc = MargValueFuncCRRA(dvFrak_dkFunc, self.CRRA)

        # Evaluate realizations of value and marginal value after asset returns are realized

        # Calculate end-of-period marginal value of assets by taking expectations
        self.EndOfPrddvda = self.DiscFac * calc_expectation(
            self.ShkDstn,  # includes RiskyShk and HumanShk
            self.EndOfPrddvda_helper,  # constructed from intermed marginal value funcs
            self.aNrmNow,  # exogenous grid
            self.kNrmNow,  # grid of human capital
            self.shareNow,  # discrete shares
        )

        self.EndOfPrddvda = self.EndOfPrddvda[:, :, :, 0]
        self.cEGM = self.uPinv(self.EndOfPrddvda)

        # Calculate end-of-period marginal value of risky portfolio share by taking expectations
        self.EndOfPrddvds = calc_expectation(
            self.ShkDstn,
            self.EndOfPrddvds_helper,
            self.aNrmNow,
            self.kNrmNow,
            self.shareNow,
        )

        self.EndOfPrddvds = self.EndOfPrddvds[:, :, :, 0]

    def optimize_share(self):
        """
        Optimization of Share on continuous interval [0,1]
        """

        # Initialize to putting everything in safe asset
        self.shareSoln = np.zeros_like(self.aNrmSoln)
        self.cNrmSoln = np.zeros_like(self.aNrmSoln)

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
            # Consumption when aNrm=0 does not depend on Share; pick any?
            self.cNrmSoln[0] = self.EndOfPrddvdaNvrs[0, :, -1]
            # Mark as constrained so that there is no attempt at optimization
            constrained_top[0] = True

        # Get consumption when share-constrained
        self.cNrmSoln[constrained_top] = self.cEGM[constrained_top, -1]
        self.cNrmSoln[constrained_bot] = self.cEGM[constrained_bot, 0]

        # For each value of aNrm, find the value of Share such that FOC-Share == 0.
        # This loop can probably be eliminated, but it's such a small step that it won't speed things up much.
        # This loop can be optimized with numba parallelization #TODO

        # find points such that 1 after is less than 0 and one before is greater than 0
        crossing = np.logical_and(FOCs[:, :, 1:] <= 0.0, FOCs[:, :, :-1] >= 0.0)
        for i in range(self.aNrmCount):
            for j in range(self.kNrmCount):
                if not (constrained_top[i, j] or constrained_bot[i, j]):
                    idx = np.argwhere(crossing[i, j])[0, 0]
                    bot_s = self.ShareGrid[idx]
                    top_s = self.ShareGrid[idx + 1]
                    bot_f = FOCs[i, j, idx]
                    top_f = FOCs[i, j, idx + 1]
                    bot_c = self.cEGM[i, j, idx]
                    top_c = self.cEGM[i, j, idx + 1]
                    alpha = 1.0 - top_f / (top_f - bot_f)
                    self.shareSoln[i, j] = (1.0 - alpha) * bot_s + alpha * top_s
                    self.cNrmSoln[i, j] = (1.0 - alpha) * bot_c + alpha * top_c

    def dvdkSoln_helper(self, shocks, a_nrm, k_nrm, share):

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

        # Calculate the endogenous mNrm gridpoints when the agent adjusts his portfolio
        self.mNrmSoln = self.aNrmSoln + self.cNrmSoln

        dvdkSoln = self.DiscFac * calc_expectation(
            self.ShkDstn,
            self.dvdkSoln_helper,
            self.aNrmSoln,
            self.kNrmSoln,
            self.shareSoln,
        )

        dvdkSoln = dvdkSoln[:, :, 0]

        cFunc_by_k = []
        shareFunc_by_k = []
        dvdk_by_k = []

        mNrm_temp = np.insert(self.mNrmSoln, 0, 0.0, axis=0)
        cNrm_temp = np.insert(self.cNrmSoln, 0, 0.0, axis=0)
        # is this acurate? maybe it's the opposite; also what is the share limit?
        Share_lower_bound = 1.0
        share_temp = np.insert(self.shareNow, 0, Share_lower_bound, axis=0)
        # why do we repeat the last point?
        dvdk_temp = np.insert(dvdkSoln, 0, dvdkSoln[0], axis=0)

        # still need a fix for when k is low, below min(kNrmGrid)
        kNrm_temp = np.insert(self.kNrmGrid, 0, 0.0)
        # for now, assume if k is 0; agent puts all on children
        # repeat lowest m; it doesn't matter what it is
        mNrm_temp = np.insert(mNrm_temp, 0, mNrm_temp[:, 0], axis=1)
        # consume 0 when kNrm is 0; all in on child
        cNrm_temp = np.insert(cNrm_temp, 0, 0.0, axis=1)
        # invest all on child at kNrm = 0
        share_temp = np.insert(share_temp, 0, Share_lower_bound, axis=1)

        for i in range(self.kNrmCount + 1):
            cFunc_by_k.append(LinearInterp(mNrm_temp[:, i], cNrm_temp[:, i]))
            shareFunc_by_k.append(LinearInterp(mNrm_temp[:, i], share_temp[:, i]))
            dvdk_by_k.append(LinearInterp(mNrm_temp[:, i], dvdk_temp[:, i]))

        self.cFuncSoln = LinearInterpOnInterp1D(cFunc_by_k, kNrm_temp)

        # Construct the marginal value (of mNrm) function when the agent can adjust
        self.vPfuncSoln = MargValueFuncCRRA(self.cFuncNow, self.CRRA)

        self.dvdkFuncSoln = LinearInterpOnInterp1D(dvdk_by_k, kNrm_temp)

    def add_vFunc(self):
        """
        Creates the value function for this period and adds it to the solution.
        """

        self.make_EndOfPrdvFunc()
        self.make_vFunc()

    def vFrak_helper(self, shocks, b_nrm_next, k_nrm_next):
        mNrm_next = self.m_nrm_next(shocks, b_nrm_next)

        v_next = self.vFuncNext(mNrm_next, k_nrm_next / shocks[self.PermShkIdx])

        return (shocks[self.PermShkIdx] * self.PermGroFac) ** (1.0 - self.CRRA) * v_next

    def EndOfPrdv_helper(self, shocks, a_nrm, k_nrm, share):

        b_nrm = a_nrm * (1 - share)
        i_nrm = a_nrm * share

        b_nrm_next = self.b_nrm_next(shocks, b_nrm)
        k_nrm_next = self.k_nrm_next(shocks, i_nrm, k_nrm)

        return self.vFrakFunc(b_nrm_next, k_nrm_next)

    def make_EndOfPrdvFunc(self):
        """
        Construct the end-of-period value function for this period, storing it
        as an attribute of self for use by other methods.
        """

        # Calculate intermediate value by taking expectations over income shocks
        vFrak = calc_expectation(
            self.IncShkDstn, self.vFrak_helper, self.bNrmNext, self.kNrmNext
        )
        vFrak = vFrak[:, :, 0]
        vFrakNvrs = self.uinv(vFrak)
        vFrakNvrsFunc = BilinearInterp(vFrakNvrs, self.bNrmGrid, self.kNrmGrid)
        self.vFrakFunc = ValueFuncCRRA(vFrakNvrsFunc, self.CRRA)

        # Calculate end-of-period value by taking expectations
        self.EndOfPrdv = self.DiscFac * calc_expectation(
            self.ShkDstn,
            self.EndOfPrdv_helper,
            self.aNrmSoln,
            self.kNrmSoln,
            self.shareSoln,
        )
        self.EndOfPrdv = self.EndOfPrdv[:, :, :, 0]
        self.EndOfPrdvNvrs = self.uinv(self.EndOfPrdv)

    def make_vFunc(self):
        """
        Creates the value functions for this period, defined over market
        resources m when agent can adjust his portfolio, and over market
        resources and fixed share when agent can not adjust his portfolio.
        self must have the attribute EndOfPrdvFunc in order to execute.
        """

        # First, make an end-of-period value function over aNrm and Share
        EndOfPrdvNvrsFunc = TrilinearInterp(
            self.EndOfPrdvNvrs, self.aNrmGrid, self.kNrmGrid, self.ShareGrid
        )
        EndOfPrdvFunc = ValueFuncCRRA(EndOfPrdvNvrsFunc, self.CRRA)

        # Construct the value function when the agent can adjust his portfolio
        mNrm_temp = self.aNrmSoln  # Just use aXtraGrid as our grid of mNrm values
        cNrm_temp = self.cFuncAdj_now(mNrm_temp, self.kNrmSoln)
        aNrm_temp = mNrm_temp - cNrm_temp
        Share_temp = self.ShareFuncAdj_now(mNrm_temp, self.kNrmSoln)
        v_temp = self.u(cNrm_temp) + EndOfPrdvFunc(aNrm_temp, self.kNrmSoln, Share_temp)
        vNvrs_temp = self.uinv(v_temp)
        vNvrsP_temp = self.uP(cNrm_temp) * self.uinvP(v_temp)

        vNvrs_by_k = []
        for i in range(self.kNrmCount):
            vNvrs_by_k.append(
                CubicInterp(
                    np.insert(mNrm_temp, 0, 0.0, axis=0),  # x_list
                    np.insert(vNvrs_temp, 0, 0.0, axis=0),  # f_list
                    np.insert(vNvrsP_temp, 0, vNvrsP_temp[0], axis=0),  # dfdx_list
                )
            )

        vNvrsFunc = LinearInterpOnInterp1D(vNvrs_by_k, self.kNrmGrid)
        # Re-curve the pseudo-inverse value function
        self.vFuncSoln = ValueFuncCRRA(vNvrsFunc, self.CRRA)

    def make_solution(self):

        self.solution = ParentalSolution(
            cFunc=self.cFuncSoln,
            shareFunc=self.shareFuncSoln,
            vPfunc=self.vPfuncSoln,
            vFunc=self.vFuncASoln,
            dvdkFunc=self.dvdkFuncNow,
            dvdmFunc=self.vPfuncSoln,
        )

    def solve(self):
        """
        Solve the one period problem for a portfolio-choice consumer.

        Returns
        -------
        solution_now : PortfolioSolution
        The solution to the single period consumption-saving with portfolio choice
        problem.  Includes two consumption and risky share functions: one for when
        the agent can adjust his portfolio share (Adj) and when he can't (Fxd).
        """

        # Make arrays of end-of-period assets and end-of-period marginal values
        self.prepare_to_calc_EndOfPrdvP()
        self.calc_EndOfPrdvP()

        # Construct a basic solution for this period
        self.optimize_share()
        self.make_basic_solution()

        # Add the value function
        self.add_vFunc()

        self.make_solution()

        return self.solution


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


init_parent = {
    "TotlFactrProd": [1.0],
    "InvstShare": [1 / 3],
    "KaptlShare": [1 / 3],
    "IncomShare": [1 / 3],
}
