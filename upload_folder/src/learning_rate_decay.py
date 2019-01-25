import mxnet as mx
import logging
import math
import numpy as np


class ExponentialScheduler(mx.lr_scheduler.LRScheduler):

    """Applies exponential decay to the learning rate.
    decayed_learning_rate = learning_rate *
                          decay_rate ^ (global_step / decay_steps)
    """

    def __init__(self,
                 decay_step,
                 factor=0.94,
                 staircase=False,
                 stop_factor_lr=1e-8):
        super(ExponentialScheduler, self).__init__()
        self.decay_step=decay_step
        self.factor=factor
        self.staircase=staircase
        self.stop_factor_lr=stop_factor_lr
        self.start_lr=-1

        if factor > 1.0:
            raise ValueError("Factor must be no more than 1 to make lr reduce")
        assert isinstance(decay_step, int), "decay_step must be int type"

    def __call__(self, num_update):

        if self.start_lr < 0:
            self.start_lr = self.base_lr
        if self.staircase:
            self.base_lr = self.start_lr * pow(self.factor,
                                               num_update / self.decay_step)
        else:
            self.base_lr = self.start_lr * pow(self.factor,
                                    float(num_update) / self.decay_step)
        if self.base_lr <= self.stop_factor_lr:
            self.base_lr = self.stop_factor_lr
        return self.base_lr


class PiecewiseConstantScheduler(mx.lr_scheduler.LRScheduler):

    """
    Piecewise constant from boundaries and interval values.
      Example: use a learning rate that's 1.0 for the first 100001 steps, 0.5
        for the next 10000 steps, and 0.1 for any additional steps.
    """

    def __init__(self,
                 boundary_step,
                 lr_values):
        super(PiecewiseConstantScheduler, self).__init__()
        assert isinstance(boundary_step, list) and len(boundary_step) >= 1
        assert isinstance(lr_values, list) and len(lr_values) >= 1
        assert len(lr_values) == (len(boundary_step)+1)


        for i, _step in enumerate(boundary_step):
            if i != 0 and boundary_step[i] <= boundary_step[i-1]:
                raise ValueError("Schedule step must be an increasing integer list")
            if _step < 1:
                raise ValueError("Schedule step must be greater or equal than 1 round")

        for i, _values in enumerate(lr_values):
            # if i != 0 and lr_values[i] >= lr_values[i-1]:
            #     raise ValueError("Schedule step must be an decreasing list")
            if _values < 0:
                raise ValueError(" learning rate values must be greater or equal than 0")

        self.boundary_step = boundary_step
        self.lr_values = lr_values
        self.cur_step_ind = -1

    def __call__(self, num_update):
        if self.cur_step_ind < 0:
            self.cur_step_ind = 0
            self.base_lr = self.lr_values[self.cur_step_ind]


        while self.cur_step_ind <= len(self.boundary_step)-1:
            if num_update > self.boundary_step[self.cur_step_ind]:
                self.cur_step_ind += 1
                self.base_lr = self.lr_values[self.cur_step_ind]
                logging.info("Update[%d]: Change learning rate to %0.5e",
                             num_update, self.base_lr)
            else:
                return self.base_lr
        return self.base_lr


class PolynomialScheduler(mx.lr_scheduler.LRScheduler):
    """Reduce learning rate in factor

    Assume the weight has been updated by n times, then the learning rate will
    be

     (start_lr-end_lr) * (1 - num_update/max_num_update) ^ (power)+end_lr

    Parameters
    ----------
    max_num_update : int default = 300000
        max num_update of training.
    power : float  default = 1.4
        The power to change lr.
    stop_factor_lr : float, default = 1e-8
        Stop updating the learning rate if it is less than this value.
    """

    def __init__(self,
                 decay_steps=300000,
                 power=1.0,
                 end_lr=0.0001,
                 cycle=False):
        super(PolynomialScheduler, self).__init__()
        if decay_steps < 0:
            raise ValueError("Schedule max_num_update must be greater than 0")
        if power < 0:
            raise ValueError("power must be no more than 0 to make lr reduce")
        self.decay_steps = decay_steps
        self.power = power
        self.end_lr = end_lr
        self.cycle=cycle
        self.start_lr = -1
        self.update_steps=0

    def __call__(self, num_update):
        """
        Call to schedule current learning rate

        Parameters
        ----------
        num_update: int
            the maximal number of updates applied to a weight.
        """
        if self.start_lr < 0:
            self.start_lr = self.base_lr
        if self.cycle:
            if num_update==0:
                 multiplier=1
            else:
                 multiplier=math.ceil(float(num_update)/self.decay_steps)
            decay_steps = self.decay_steps * multiplier
        else:
            num_update=min(num_update, self.decay_steps)
            decay_steps = self.decay_steps

        self.base_lr = (self.start_lr-self.end_lr)*pow(1-float(num_update)/decay_steps,
                                                     self.power) + self.end_lr
        return self.base_lr

class NaturalexpScheduler(mx.lr_scheduler.LRScheduler):

    """Applies natural exponential decay to the initial learning rate.
      When training a model, it is often recommended to lower the learning rate as
      the training progresses.  This function applies an exponential decay function
      to a provided initial learning rate.  It requires an `global_step` value to
      compute the decayed learning rate.  You can just pass a TensorFlow variable
      that you increment at each training step.
      The function returns the decayed learning rate.  It is computed as:
      ```python
      decayed_learning_rate = learning_rate * exp(-decay_rate * global_step)
      ```
      """
    def __init__(self,decay_steps,
                 decay_factor,
                 staircase=False):

        super(NaturalexpScheduler, self).__init__()
        if decay_steps < 0:
            raise ValueError("Schedule decay_steps must be greater than 0")

        if decay_factor < 0:
            raise ValueError("Schedule decay_factor must be greater than 0")

        self.decay_steps=decay_steps
        self.decay_factor=decay_factor
        self.staircase=staircase
        self.start_lr=-1


    def __call__(self,num_update):
        if self.start_lr < 0:
            self.start_lr = self.base_lr
        if self.staircase:
            fraction=num_update/self.decay_steps
        else:
            fraction=float(num_update)/self.decay_steps

        decayed=math.exp(-self.decay_factor*fraction)

        self.base_lr=self.start_lr*decayed
        return self.base_lr




class InversetimeScheduler(mx.lr_scheduler.LRScheduler):
    """Applies inverse time decay to the initial learning rate.
      When training a model, it is often recommended to lower the learning rate as
      the training progresses.  This function applies an inverse decay function
      to a provided initial learning rate.  It requires an `global_step` value to
      compute the decayed learning rate.  You can just pass a TensorFlow variable
      that you increment at each training step.
      The function returns the decayed learning rate.  It is computed as:
      ```python
      decayed_learning_rate = learning_rate / (1 + decay_rate * global_step / decay_step)
      ```
      or, if `staircase` is `True`, as:
      ```python
      decayed_learning_rate = learning_rate / (1 + decay_rate * floor(global_step / decay_step))
      """


    def __init__(self,decay_steps,
                 decay_factor,
                 staircase=False):

        super(InversetimeScheduler, self).__init__()
        if decay_steps < 0:
            raise ValueError("Schedule decay_steps must be greater than 0")

        if decay_factor < 0:
            raise ValueError("Schedule decay_factor must be greater than 0")

        self.decay_steps=decay_steps
        self.decay_factor=decay_factor
        self.staircase=staircase
        self.start_lr=-1

    def __call__(self,num_update):
        if self.start_lr < 0:
            self.start_lr = self.base_lr
        if self.staircase:
            fraction=num_update/self.decay_steps
        else:
            fraction=float(num_update)/self.decay_steps

        decayed=1.0+self.decay_factor*fraction

        self.base_lr=float(self.start_lr)/decayed
        return self.base_lr








class CosineScheduler(mx.lr_scheduler.LRScheduler):
    """Applies cosine decay to the learning rate.
      See [Loshchilov & Hutter, ICLR2016], SGDR: Stochastic Gradient Descent
      with Warm Restarts. https://arxiv.org/abs/1608.03983
      When training a model, it is often recommended to lower the learning rate as
      the training progresses.  This function applies a cosine decay function
      to a provided initial learning rate.  It requires a `global_step` value to
      compute the decayed learning rate.  You can just pass a TensorFlow variable
      that you increment at each training step.
      The function returns the decayed learning rate.  It is computed as:
      ```python
      global_step = min(global_step, decay_steps)
      cosine_decay = 0.5 * (1 + cos(pi * global_step / decay_steps))
      decayed = (1 - alpha) * cosine_decay + alpha
      decayed_learning_rate = learning_rate * decayed
      ```
      """

    def __init__(self, decay_steps=300000,
                 stop_factor_lr=1e-8,
                 alpha=0.0,
                 cycle=False):
        super(CosineScheduler, self).__init__()
        if decay_steps < 0:
            raise ValueError("Schedule max_num_update must be greater than 0")
        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must be larger than 0 or smaller than 1 to make lr reduce")
        self.decay_steps = decay_steps
        self.alpha = alpha
        self.stop_factor_lr = stop_factor_lr
        self.cycle = cycle
        self.start_lr = -1

    def __call__(self, num_update):

        if self.start_lr < 0:
            self.start_lr = self.base_lr

        if self.cycle:
          global_steps = num_update % (self.decay_steps) + 1
        else:
          global_steps = min(num_update, self.decay_steps)
        completed_fraction = float(global_steps) / self.decay_steps
        cosine_decayed = 0.5*(1.0 + math.cos(math.pi * completed_fraction))
        decayed = (1 - self.alpha) * cosine_decayed + self.alpha
        self.base_lr = self.start_lr * decayed

        if self.base_lr < self.stop_factor_lr:
            self.base_lr = self.stop_factor_lr

        return self.base_lr



class LinearcosineScheduler(mx.lr_scheduler.LRScheduler):
    """Applies linear cosine decay to the learning rate.
      See [Bello et al., ICML2017] Neural Optimizer Search with RL.
      https://arxiv.org/abs/1709.07417
      For the idea of warm starts here controlled by `num_periods`,
      see [Loshchilov & Hutter, ICLR2016] SGDR: Stochastic Gradient Descent
      with Warm Restarts. https://arxiv.org/abs/1608.03983
      Note that linear cosine decay is more aggressive than cosine decay and
      larger initial learning rates can typically be used.
      When training a model, it is often recommended to lower the learning rate as
      the training progresses.  This function applies a linear cosine decay function
      to a provided initial learning rate.  It requires a `global_step` value to
      compute the decayed learning rate.  You can just pass a TensorFlow variable
      that you increment at each training step.
      The function returns the decayed learning rate.  It is computed as:
      ```python
      global_step = min(global_step, decay_steps)
      linear_decay = (decay_steps - global_step) / decay_steps)
      cosine_decay = 0.5 * (
          1 + cos(pi * 2 * num_periods * global_step / decay_steps))
      decayed = (alpha + linear_decay) * cosine_decay + beta
      decayed_learning_rate = learning_rate * decayed
      """

    def __init__(self, decay_steps=10000,
                 num_periods=0.5,
                 alpha=0.0,
                 beta=0.001,
                 stop_factor_lr=1e-8):
        super(LinearcosineScheduler, self).__init__()
        if decay_steps < 0:
            raise ValueError("Schedule max_num_update must be greater than 0")
        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must be larger than 0 or smaller than 1 to make lr reduce")
        if beta < 0:
            raise ValueError("beta must be larger than  0 to make lr reduce")
        if num_periods < 0:
            raise ValueError("num_periods must be larger than  0 to make lr reduce")

        self.decay_steps=decay_steps
        self.num_periods=num_periods
        self.alpha=alpha
        self.beta=beta
        self.stop_factor_lr = stop_factor_lr
        self.start_lr=-1

    def __call__(self,num_update):

        if self.start_lr < 0:
            self.start_lr=self.base_lr

        global_steps=min(self.decay_steps,num_update)
        linear_decayed=float(self.decay_steps-global_steps)/self.decay_steps
        completed_fraction=float(global_steps)/self.decay_steps
        fraction=2.0*self.num_periods*completed_fraction
        cosine_decayed = 0.5 * (1.0 + math.cos(math.pi * fraction))
        linear_cosine_decayed=(self.alpha+linear_decayed)*cosine_decayed+self.beta
        self.base_lr = self.start_lr * linear_cosine_decayed

        if self.base_lr < self.stop_factor_lr:
          self.base_lr = self.stop_factor_lr

        return self.base_lr

class NoisylinearcosineScheduler(mx.lr_scheduler.LRScheduler):
    """Applies noisy linear cosine decay to the learning rate.
      See [Bello et al., ICML2017] Neural Optimizer Search with RL.
      https://arxiv.org/abs/1709.07417
      For the idea of warm starts here controlled by `num_periods`,
      see [Loshchilov & Hutter, ICLR2016] SGDR: Stochastic Gradient Descent
      with Warm Restarts. https://arxiv.org/abs/1608.03983
      Note that linear cosine decay is more aggressive than cosine decay and
      larger initial learning rates can typically be used.
      When training a model, it is often recommended to lower the learning rate as
      the training progresses.  This function applies a noisy linear
      cosine decay function to a provided initial learning rate.
      It requires a `global_step` value to compute the decayed learning rate.
      You can just pass a TensorFlow variable that you increment at each
      training step.
      The function returns the decayed learning rate.  It is computed as:
      ```python
      global_step = min(global_step, decay_steps)
      linear_decay = (decay_steps - global_step) / decay_steps)
      cosine_decay = 0.5 * (
          1 + cos(pi * 2 * num_periods * global_step / decay_steps))
      decayed = (alpha + linear_decay + eps_t) * cosine_decay + beta
      decayed_learning_rate = learning_rate * decayed
      ```
      where eps_t is 0-centered gaussian noise with variance
      initial_variance / (1 + global_step) ** variance_decay
      """

    def __init__(self, decay_steps=10000,
                 initial_variance=1.0,
                 variance_decay=0.55,
                 num_periods=0.5,
                 alpha=0.0,
                 beta=0.001):

        super(NoisylinearcosineScheduler, self).__init__()
        if decay_steps < 0:
            raise ValueError("Schedule max_num_update must be greater than 0")
        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must be larger than 0 or smaller than 1 to make lr reduce")
        if beta < 0:
            raise ValueError("beta must be larger than  0 to make lr reduce")
        if num_periods < 0:
            raise ValueError("num_periods must be larger than  0 to make lr reduce")

        self.decay_steps = decay_steps
        self.num_periods = num_periods
        self.initial_variance=initial_variance
        self.variance_decay=variance_decay
        self.alpha = alpha
        self.beta = beta
        self.start_lr = -1


    def __call__(self,num_update):

        if self.start_lr < 0:
            self.start_lr=self.base_lr

        global_steps=min(self.decay_steps,num_update)
        variance=self.initial_variance/pow(1.0+global_steps, self.variance_decay)
        std=math.sqrt(variance)
        linear_decayed=float(self.decay_steps-global_steps)/self.decay_steps
        noisy_linear_decayed=(linear_decayed+np.random.normal(scale=std))

        completed_fraction=float(global_steps)/self.decay_steps
        fraction=2.0*self.num_periods*completed_fraction
        cosine_decayed = 0.5 * (1.0 + math.cos(math.pi * fraction))
        noisy_linear_cosine_decayed=(self.alpha+noisy_linear_decayed)*cosine_decayed+self.beta
        self.base_lr = self.start_lr * noisy_linear_cosine_decayed

        return self.base_lr


class CLRScheduler(mx.lr_scheduler.LRScheduler):
    """Applies CLR decay to the learning rate.
      See [Super-Convergence:Very Fast Training of Neural
       Networks Using Large Learning Rates
      . https://arxiv.org/pdf/1708.07120v3.pdf]
      The function returns the cyclical learning rate.  It is computed as:
      ```python
      stage1:
      when  num_update <=2*step_sizes
      x=num_update-step_sizes
      x=x/step_sizes
      rate= start_lr + (max_lr-start_lr)*max(0,1-fab(x))
      stage2:
      when 2*step_size<num_update<=max_iter_steps

      x=num_update-2*step_sizes
      x=x/(max_iter_steps-2*step_sizes)
      rate= target_lr + (start_lr-target_lr)*pow(1-x,power)

      when num_update > max_iter_steps
      rate=target_lr

      ```
      """

    def __init__(self, step_sizes=50000,
                 max_iter_steps=500000,
                 start_lr=0.1,
                 max_lr=1,
                 target_lr=0.0001,
                 power=1):
        super(CLRScheduler, self).__init__()
        if step_sizes < 0:
            raise ValueError("Schedule step_sizes must be greater than 0")
        if max_iter_steps < 0:
            raise ValueError("Schedule max_num_update must be greater than 0")
        assert max_iter_steps >= 2*step_sizes

        self.step_sizes = step_sizes
        self.max_iter_steps = max_iter_steps
        self.max_lr = max_lr
        self.target_lr = target_lr
        self.power=power
        self.start_lr = start_lr

    def __call__(self, num_update):
        if self.start_lr < 0:
            self.start_lr = self.base_lr

        if num_update <= 2*self.step_sizes:
            x = float(num_update-self.step_sizes)
            x = x/self.step_sizes
            self.base_lr = self.start_lr\
                         + (self.max_lr-self.start_lr)*max(0,1.0-math.fabs(x))
        elif num_update <= self.max_iter_steps:
            x = float(num_update-2*self.step_sizes)
            x = x/(self.max_iter_steps-2*self.step_sizes)
            self.base_lr = self.target_lr \
                         + (self.start_lr - self.target_lr)*pow(1-x, self.power)
        else:
            self.base_lr = self.target_lr

        if self.base_lr < self.target_lr:
            self.base_lr = self.target_lr

        return self.base_lr
