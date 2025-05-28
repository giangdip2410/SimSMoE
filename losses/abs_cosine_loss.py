
""" Symmetrized Cosine Similarity Loss Functions """


import warnings

import torch


class AbsCosineSimilarityLoss(torch.nn.Module):
    """Implementation of the Symmetrized Loss used in the SimSiam[0] paper.

    [0] SimSiam, 2020, https://arxiv.org/abs/2011.10566

    Examples:

        >>> # initialize loss function
        >>> loss_fn = CosineSimilarityLoss()
        >>>
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(out0, out1)

    """

    def __init__(self) -> None:
        super().__init__()
        

    def _cosine_simililarity(self, x, y):
        v = torch.nn.functional.cosine_similarity(x, y, dim=-1).abs().mean()
        return v

    def forward(self, out0: torch.Tensor, out1: torch.Tensor):
        """Forward pass through Symmetric Loss.

        Args:
            out0:
                Output projections of the first set of transformed images.
                Expects the tuple to be of the form (z0, p0), where z0 is
                the output of the backbone and projection mlp, and p0 is the
                output of the prediction head.
            out1:
                Output projections of the second set of transformed images.
                Expects the tuple to be of the form (z1, p1), where z1 is
                the output of the backbone and projection mlp, and p1 is the
                output of the prediction head.

        Returns:
            Contrastive Cross Entropy Loss value.

        Raises:
            ValueError if shape of output is not multiple of batch_size.
        """
    

        return self._cosine_simililarity(out0, out1)
