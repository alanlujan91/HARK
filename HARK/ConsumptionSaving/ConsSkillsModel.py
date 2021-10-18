"""
Classes to solve consumption-savings model where the agent can invest on
the human capital formation of their offspring.
Extends ConsGenIncProcessModel.py by adding
"""
from HARK import MetricObject, NullFunc
from HARK.ConsumptionSaving.ConsGenIncProcessModel import (
    GenIncProcessConsumerType,
    ConsGenIncProcessSolver,
)


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
        dvdhFunc=NullFunc(),
    ):

        # set attributes of self
        self.cFunc = cFunc
        self.shareFunc = shareFunc
        self.vFunc = vFunc
        self.vPfunc = vPfunc
        self.dvdmFunc = dvdmFunc
        self.dvdhFunc = dvdhFunc


class ParentConsumerType(GenIncProcessConsumerType):
    pass


class ConsParentSolver(ConsGenIncProcessSolver):
    pass
