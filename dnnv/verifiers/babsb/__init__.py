from dnnv.properties import Expression
from dnnv.verifiers.bab import BaB
from dnnv.verifiers.common.base import Parameter


class BaBSB(BaB):
    EXE = "bab"
    parameters = {
        "reluify_maxpools": Parameter(bool, default=False),
    }

    def __init__(self, dnn_property: Expression, **kwargs):
        super().__init__(dnn_property, **kwargs)
        self.parameter_values["smart_branching"] = True
