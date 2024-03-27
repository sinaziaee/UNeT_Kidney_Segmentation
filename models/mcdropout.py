import torch
from torch import nn
from torch.nn import functional as F

class MCDropoutBase(nn.Module):
    """
    Base class for MC Dropout.
    """

    # Class attribute shared by all instances of MCDropoutBase.
    # If active is True, dropout is applied during inference.
    # Note that the dropout is always applied during training.
    active = True

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    @classmethod
    def activate(cls):
        """
        Class method to activate dropout during inference.
        Sets the class attribute 'active' to True.
        """
        cls.active = True

    @classmethod
    def deactivate(cls):
        """
        Class method to deactivate dropout during inference.
        Sets the class attribute 'active' to False.
        """
        cls.active = False

    @classmethod
    def is_active(cls):
        """
        Class method to check the current status of dropout.
        :return: bool, True if dropout is active, False otherwise.
        """
        return cls.active


class MCDropout1D(MCDropoutBase):
    """
    Class implementing Monte Carlo Dropout for 1D inputs.
    Inherits from the MCDropoutBase class.
    """
    def forward(self, x):
        if self.__class__.active or self.training:
            return F.dropout(x, self.p, self.training or self.__class__.active)
        return x


class MCDropout2D(MCDropoutBase):
    """
    Class implementing Monte Carlo Dropout for 2D inputs.
    Inherits from the MCDropoutBase class.
    """
    def forward(self, x):
        if self.__class__.active or self.training:
            return F.dropout2d(x, self.p, self.training or self.__class__.active)
        return x


class MCDropout3D(MCDropoutBase):
    """
    Class implementing Monte Carlo Dropout for 3D inputs.
    Inherits from the MCDropoutBase class.
    """
    def forward(self, x):
        if self.__class__.active or self.training:
            return F.dropout3d(x, self.p, self.training or self.__class__.active)
        return x