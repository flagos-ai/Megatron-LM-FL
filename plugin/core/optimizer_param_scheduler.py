"""Learning rate decay and weight decay incr functions."""
import logging
import math

from megatron.core.utils import log_single_rank

logger = logging.getLogger(__name__)

from plugin.decorators import plugin_implementation


@plugin_implementation("OptimizerParamScheduler", "get_lr")
def get_lr(self, param_group: dict) -> float:
    """Learning rate decay functions from:
    https://openreview.net/pdf?id=BJYwwY9ll pg. 4

    Args:
        param_group (dict): parameter group from the optimizer.
    """
    print(f"Megatron-LM-FL Plugins: get_lr")
    max_lr = param_group.get('max_lr', self.max_lr)
    min_lr = param_group.get('min_lr', self.min_lr)

    # Use linear warmup for the initial part.
    if self.lr_warmup_steps > 0 and self.num_steps <= self.lr_warmup_steps:
        return self.init_lr + (
            (max_lr - self.init_lr) * float(self.num_steps) / float(self.lr_warmup_steps)
        )

    # If the learning rate is constant, just return the initial value.
    if self.lr_decay_style == 'constant':
        return max_lr

    # For any steps larger than `self.lr_decay_steps`, use `min_lr`.
    if self.num_steps > self.lr_decay_steps:
        return min_lr

    # If we are done with the warmup period, use the decay style.
    if self.lr_decay_style == 'inverse-square-root':
        warmup_steps = max(self.lr_warmup_steps, 1)
        num_steps = max(self.num_steps, 1)
        lr = max_lr * warmup_steps**0.5 / (num_steps**0.5)
        return max(min_lr, lr)

    # stablelm2 scheduler of multiple stages
    if self.stablelm2_scheduler_config is not None:
        log_single_rank(logger, logging.INFO, f"> stablelm2_scheduler_config: {self.stablelm2_scheduler_config}")
        if self.num_steps <= self.stablelm2_scheduler_config.cosine_samples:
            ## cosine phase
            # decay_ratio = float(self.num_steps) / float(self.lr_decay_steps)
            # TODO
            decay_ratio = float(self.num_steps) / float(self.stablelm2_scheduler_config.cosine_period_samples)
            cosine_min_lr = self.stablelm2_scheduler_config.cosine_max_lr * 0.1
            delta_lr = self.stablelm2_scheduler_config.cosine_max_lr - cosine_min_lr
            coeff = 0.5 * (math.cos(2 * math.pi * decay_ratio) + 1.0)
            self.stablelm2_scheduler_config.cosine_lr = cosine_min_lr + coeff * delta_lr
            return self.stablelm2_scheduler_config.cosine_lr
        elif self.num_steps <= self.stablelm2_scheduler_config.rsqrt_samples:
            ## rsqrt phase
            alpha = self.stablelm2_scheduler_config.alpha
            beta = self.stablelm2_scheduler_config.beta
            gbs = self.stablelm2_scheduler_config.global_batch_size * 1.0
            self.stablelm2_scheduler_config.rsqrt_lr = alpha / ((self.num_steps / gbs + beta) ** 0.5)
            return self.stablelm2_scheduler_config.rsqrt_lr
        elif self.stablelm2_scheduler_config.decay_samples <= 0:
            ## optional linear phase
            decay_steps_ = self.lr_decay_steps - self.stablelm2_scheduler_config.rsqrt_samples
            num_steps_ = self.num_steps - self.stablelm2_scheduler_config.rsqrt_samples
            decay_ratio = float(num_steps_) / float(decay_steps_)
            coeff = (1.0 - decay_ratio)
            return coeff * self.stablelm2_scheduler_config.rsqrt_lr
        else:
            ## optional linear phase
            valid_lr_decay_steps_ = min(
                self.lr_decay_steps,
                self.stablelm2_scheduler_config.rsqrt_samples + self.stablelm2_scheduler_config.decay_samples)
            if self.num_steps <= valid_lr_decay_steps_:
                decay_steps_ = valid_lr_decay_steps_ - self.stablelm2_scheduler_config.rsqrt_samples
                num_steps_ = self.num_steps - self.stablelm2_scheduler_config.rsqrt_samples
                decay_ratio = float(num_steps_) / float(decay_steps_)
                coeff = (1.0 - decay_ratio)
                delta_lr = self.stablelm2_scheduler_config.rsqrt_lr - self.min_lr
                assert decay_ratio >= 0.0
                return coeff * delta_lr + self.min_lr
            else:
                return self.min_lr

    # Warmup-Stable-Decay(WSD)
    if self.lr_decay_style == 'warmup-stable-decay':
        W = self.lr_warmup_steps
        S = round((self.lr_decay_steps - W) * 10. / 11.)
        ## D is 10% of S.
        T = self.lr_decay_steps - W - S
        ## Warmup Phase, see above
        ## Stable Phase
        if self.num_steps < S:
            return self.max_lr
        else: # Decay Phase
            return self.max_lr * 0.5 ** ((self.num_steps - S) / T)

    num_steps_ = self.num_steps - self.lr_warmup_steps
    decay_steps_ = self.lr_decay_steps - self.lr_warmup_steps
    decay_ratio = float(num_steps_) / float(decay_steps_)
    assert decay_ratio >= 0.0
    assert decay_ratio <= 1.0
    delta_lr = max_lr - min_lr

    coeff = None
    if self.lr_decay_style == 'linear':
        coeff = 1.0 - decay_ratio
    elif self.lr_decay_style == 'cosine':
        coeff = 0.5 * (math.cos(math.pi * decay_ratio) + 1.0)
    elif self.lr_decay_style == 'WSD':
        wsd_anneal_start_ = self.lr_decay_steps - self.wsd_decay_steps
        if self.num_steps <= wsd_anneal_start_:
            coeff = 1.0
        else:
            wsd_steps = self.num_steps - wsd_anneal_start_
            wsd_decay_ratio = float(wsd_steps) / float(self.wsd_decay_steps)
            if self.lr_wsd_decay_style == "linear":
                coeff = 1.0 - wsd_decay_ratio
            elif self.lr_wsd_decay_style == "cosine":
                coeff = 0.5 * (math.cos(math.pi * wsd_decay_ratio) + 1.0)
            elif self.lr_wsd_decay_style == "exponential":
                coeff = (2.0 * math.pow(0.5, wsd_decay_ratio)) - 1.0
            elif self.lr_wsd_decay_style == "minus_sqrt":
                coeff = 1.0 - math.sqrt(wsd_decay_ratio)

    else:
        raise Exception(f'{self.lr_decay_style} decay style is not supported.')
    assert coeff is not None

    return min_lr + coeff * delta_lr

