from dnnv.errors import DNNVError


class VerifierError(DNNVError):
    pass


class VerifierTranslatorError(DNNVError):
    pass


__all__ = ["VerifierError", "VerifierTranslatorError"]
