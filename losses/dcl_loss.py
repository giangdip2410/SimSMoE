from functools import partial
from typing import Callable, Optional

import torch
from torch import Tensor
from torch import distributed as torch_dist
from torch import nn

# from lightly.utils import dist

from typing import Optional, Tuple

import torch
import torch.distributed as dist


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all processes, supporting backward propagation.

    This code was taken and adapted from here:
    https://github.com/Spijkervet/SimCLR

    """

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        ctx.save_for_backward(input)
        output = [torch.empty_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads: torch.Tensor) -> torch.Tensor:
        (input,) = ctx.saved_tensors
        grad_out = torch.empty_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


def rank() -> int:
    """Returns the rank of the current process."""
    return dist.get_rank() if dist.is_initialized() else 0


def world_size() -> int:
    """Returns the current world size (number of distributed processes)."""
    return dist.get_world_size() if dist.is_initialized() else 1


def gather(input: torch.Tensor) -> Tuple[torch.Tensor]:
    """Gathers this tensor from all processes. Supports backprop."""
    return GatherLayer.apply(input)


def eye_rank(n: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Returns an (n, n * world_size) zero matrix with the diagonal for the rank
    of this process set to 1.

    Example output where n=3, the current process has rank 1, and there are
    4 processes in total:

        rank0   rank1   rank2   rank3
        0 0 0 | 1 0 0 | 0 0 0 | 0 0 0
        0 0 0 | 0 1 0 | 0 0 0 | 0 0 0
        0 0 0 | 0 0 1 | 0 0 0 | 0 0 0

    Equivalent to torch.eye for undistributed settings or if world size == 1.

    Args:
        n:
            Size of the square matrix on a single process.
        device:
            Device on which the matrix should be created.

    """
    rows = torch.arange(n, device=device, dtype=torch.long)
    cols = rows + rank() * n
    diag_mask = torch.zeros((n, n * world_size()), dtype=torch.bool)
    diag_mask[(rows, cols)] = True
    return diag_mask


def rank_zero_only(fn):
    """Decorator that only runs the function on the process with rank 0.

    Example:
        >>> @rank_zero_only
        >>> def print_rank_zero(message: str):
        >>>     print(message)
        >>>
        >>> print_rank_zero("Hello from rank 0!")

    """

    def wrapped(*args, **kwargs):
        if rank() == 0:
            return fn(*args, **kwargs)

    return wrapped


@rank_zero_only
def print_rank_zero(*args, **kwargs) -> None:
    """Equivalent to print, but only runs on the process with rank 0."""
    print(*args, **kwargs)


def negative_mises_fisher_weights(
    out0: Tensor, out1: Tensor, sigma: float = 0.5
) -> torch.Tensor:
    """Negative Mises-Fisher weighting function as presented in Decoupled
    Contrastive Learning [0].

    The implementation was inspired by [1].

    - [0] Chun-Hsiao Y. et. al., 2021, Decoupled Contrastive Learning https://arxiv.org/abs/2110.06848
    - [1] https://github.com/raminnakhli/Decoupled-Contrastive-Learning

    Args:
        out0:
            Output projections of the first set of transformed images.
            Shape: (batch_size, embedding_size)
        out1:
            Output projections of the second set of transformed images.
            Shape: (batch_size, embedding_size)
        sigma:
            Similarities are scaled by inverse sigma.
    Returns:
        A tensor with shape (batch_size,) where each entry is the weight for one
        of the input images.

    """
    similarity = torch.einsum("nm,nm->n", out0.detach(), out1.detach()) / sigma
    return 2 - out0.shape[0] * nn.functional.softmax(similarity, dim=0)


class DCLLoss(nn.Module):
    """Implementation of the Decoupled Contrastive Learning Loss from
    Decoupled Contrastive Learning [0].

    This code implements Equation 6 in [0], including the sum over all images `i`
    and views `k`. The loss is reduced to a mean loss over the mini-batch.
    The implementation was inspired by [1].

    - [0] Chun-Hsiao Y. et. al., 2021, Decoupled Contrastive Learning https://arxiv.org/abs/2110.06848
    - [1] https://github.com/raminnakhli/Decoupled-Contrastive-Learning

    Attributes:
        temperature:
            Similarities are scaled by inverse temperature.
        weight_fn:
            Weighting function `w` from the paper. Scales the loss between the
            positive views (views from the same image). No weighting is performed
            if weight_fn is None. The function must take the two input tensors
            passed to the forward call as input and return a weight tensor. The
            returned weight tensor must have the same length as the input tensors.
        gather_distributed:
            If True then negatives from all gpus are gathered before the
            loss calculation.

    Examples:

        >>> loss_fn = DCLLoss(temperature=0.07)
        >>>
        >>> # generate two random transforms of images
        >>> t0 = transforms(images)
        >>> t1 = transforms(images)
        >>>
        >>> # embed images using some model, for example SimCLR
        >>> out0 = model(t0)
        >>> out1 = model(t1)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(out0, out1)
        >>>
        >>> # you can also add a custom weighting function
        >>> weight_fn = lambda out0, out1: torch.sum((out0 - out1) ** 2, dim=1)
        >>> loss_fn = DCLLoss(weight_fn=weight_fn)

    """

    def __init__(
        self,
        temperature: float = 0.1,
        weight_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
        gather_distributed: bool = False,
    ):
        super().__init__()
        self.temperature = temperature
        self.weight_fn = weight_fn
        self.gather_distributed = gather_distributed

        if gather_distributed and not torch_dist.is_available():
            raise ValueError(
                "gather_distributed is True but torch.distributed is not available. "
                "Please set gather_distributed=False or install a torch version with "
                "distributed support."
            )

    def forward(
        self,
        out0: Tensor,
        out1: Tensor,
    ) -> Tensor:
        """Forward pass of the DCL loss.

        Args:
            out0:
                Output projections of the first set of transformed images.
                Shape: (batch_size, embedding_size)
            out1:
                Output projections of the second set of transformed images.
                Shape: (batch_size, embedding_size)

        Returns:
            Mean loss over the mini-batch.
        """
        # normalize the output to length 1
        out0 = nn.functional.normalize(out0, dim=1)
        out1 = nn.functional.normalize(out1, dim=1)

        if self.gather_distributed and dist.world_size() > 1:
            # gather representations from other processes if necessary
            out0_all = torch.cat(dist.gather(out0), 0)
            out1_all = torch.cat(dist.gather(out1), 0)
        else:
            out0_all = out0
            out1_all = out1

        # calculate symmetric loss
        loss0 = self._loss(out0, out1, out0_all, out1_all)
        loss1 = self._loss(out1, out0, out1_all, out0_all)
        return (-0.5) * (loss0 + loss1)

    def _loss(self, out0, out1, out0_all, out1_all):
        """Calculates DCL loss for out0 with respect to its positives in out1
        and the negatives in out1, out0_all, and out1_all.

        This code implements Equation 6 in [0], including the sum over all images `i`
        but with `k` fixed at 0.

        Args:
            out0:
                Output projections of the first set of transformed images.
                Shape: (batch_size, embedding_size)
            out1:
                Output projections of the second set of transformed images.
                Shape: (batch_size, embedding_size)
            out0_all:
                Output projections of the first set of transformed images from
                all distributed processes/gpus. Should be equal to out0 in an
                undistributed setting.
                Shape (batch_size * world_size, embedding_size)
            out1_all:
                Output projections of the second set of transformed images from
                all distributed processes/gpus. Should be equal to out1 in an
                undistributed setting.
                Shape (batch_size * world_size, embedding_size)

        Returns:
            Mean loss over the mini-batch.
        """
        # create diagonal mask that only selects similarities between
        # representations of the same images
        batch_size = out0.shape[0]
        if self.gather_distributed and dist.world_size() > 1:
            diag_mask = dist.eye_rank(batch_size, device=out0.device)
        else:
            diag_mask = torch.eye(batch_size, device=out0.device, dtype=torch.bool)

        # calculate similarities
        # here n = batch_size and m = batch_size * world_size.
        sim_00 = torch.einsum("nc,mc->nm", out0, out0_all) / self.temperature
        sim_01 = torch.einsum("nc,mc->nm", out0, out1_all) / self.temperature

        positive_loss = -sim_01[diag_mask]
        if self.weight_fn:
            positive_loss = positive_loss * self.weight_fn(out0, out1)

        # remove simliarities between same views of the same image
        sim_00 = sim_00[~diag_mask].view(batch_size, -1)
        # remove similarities between different views of the same images
        # this is the key difference compared to NTXentLoss
        sim_01 = sim_01[~diag_mask].view(batch_size, -1)

        negative_loss_00 = torch.logsumexp(sim_00, dim=1)
        negative_loss_01 = torch.logsumexp(sim_01, dim=1)
        return (positive_loss + negative_loss_00 + negative_loss_01).mean()


class DCLWLoss(DCLLoss):
    """Implementation of the Weighted Decoupled Contrastive Learning Loss from
    Decoupled Contrastive Learning [0].

    This code implements Equation 6 in [0] with a negative Mises-Fisher
    weighting function. The loss returns the mean over all images `i` and
    views `k` in the mini-batch. The implementation was inspired by [1].

    - [0] Chun-Hsiao Y. et. al., 2021, Decoupled Contrastive Learning https://arxiv.org/abs/2110.06848
    - [1] https://github.com/raminnakhli/Decoupled-Contrastive-Learning

    Attributes:
        temperature:
            Similarities are scaled by inverse temperature.
        sigma:
            Similar to temperature but applies the inverse scaling in the
            weighting function.
        gather_distributed:
            If True then negatives from all gpus are gathered before the
            loss calculation.

    Examples:

        >>> loss_fn = DCLWLoss(temperature=0.07)
        >>>
        >>> # generate two random transforms of images
        >>> t0 = transforms(images)
        >>> t1 = transforms(images)
        >>>
        >>> # embed images using some model, for example SimCLR
        >>> out0 = model(t0)
        >>> out1 = model(t1)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(out0, out1)

    """

    def __init__(
        self,
        temperature: float = 0.1,
        sigma: float = 0.5,
        gather_distributed: bool = False,
    ):
        super().__init__(
            temperature=temperature,
            weight_fn=partial(negative_mises_fisher_weights, sigma=sigma),
            gather_distributed=gather_distributed,
        )