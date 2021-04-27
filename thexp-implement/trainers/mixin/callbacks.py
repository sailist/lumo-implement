from thexp import Trainer, Params, Meter
from thexp.calculate.schedule import Schedule
from thexp.frame import callbacks

from .. import GlobalParams
from thextra.hold_memory import memory


class __CB(callbacks.TrainCallback):
    pass


class BaseCBMixin(Trainer):
    def callbacks(self, params: GlobalParams):
        from thexp import callbacks
        callbacks.LoggerCallback().hook(self)  # auto log in screen and file
        callbacks.EvalCallback(*params.eval_test_per_epoch).hook(self)  # auto eval/test per 5/10 epoch
        callbacks.AutoRecord().hook(self)  # auto record meter by SummaryWritter

        callbacks.ReportSche().hook(self)
        callbacks.LRSchedule().hook(self)  # auto get params.lr_sche to apply lr rate
        if params.ema:
            callbacks.EMAUpdate().hook(self)  # auto update module named with prefix `ema`


from torch.optim import Optimizer


class EachLRSchedule(callbacks.TrainCallback):
    def __init__(self, pairs: list, apply=True, use_eidx=True):
        for pair in pairs:
            assert len(pair) == 2
            optim, sche = pair
            assert isinstance(optim, Optimizer) and isinstance(sche, Schedule), str(type(optim)) + '/' + str(type(sche))
        self.pairs = pairs
        self.apply = apply
        self.use_eidx = use_eidx

    def on_train_epoch_end(self, trainer: Trainer, func, params: Params, meter: Meter, *args, **kwargs):
        super().on_train_epoch_end(trainer, func, params, meter, *args, **kwargs)
        for optim, sche in self.pairs:
            if self.use_eidx:
                step = params.eidx
            else:
                step = params.global_step

            if self.apply:
                new_lr = sche.apply(optim, step)
            else:
                ratio = sche.scale(optim, step)


class MemHoldCallback(callbacks.TrainCallback):
    priority = -1

    def __init__(self, memory, hold=False) -> None:
        super().__init__()
        self.memory = memory
        self.hold = hold

    def on_hooked(self, trainer: Trainer, params: Params):
        super().on_hooked(trainer, params)
        memory(self.memory, device=params.device, hold=self.hold).start()

    def on_train_end(self, trainer: Trainer, func, params: Params, meter: Meter, *args, **kwargs):
        super().on_train_end(trainer, func, params, meter, *args, **kwargs)

        memory.hold_current()


class ExceptionCallback(callbacks.TrainCallback):
    priority = -1

    def on_exception(self, trainer: Trainer, func, params: Params, e: BaseException, *args, **kwargs):
        import traceback, sys
        trainer.logger.raw('\n'.format(traceback.format_exception(*sys.exc_info())))
        trainer.logger.inline('continue?')
        return True
