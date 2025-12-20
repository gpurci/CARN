#!/usr/bin/python

import time
import warnings

from sys_function import * # este in root
sys_remove_modules("callback.callback")

from callback.callback import *

class RunConfigs():
    def __init__(self, model, trainer, epochs, train_dl, val_dl, history_path):
        self.model   = model
        self.trainer = trainer
        self.epochs  = epochs
        self.train_dl = train_dl
        self.val_dl   = val_dl
        self.callback = Callback(filename=history_path, freq=1)

    def run(self, name, **conf):
        print("Start running name: '{}', conf {}".format(name, conf))
        # reseteaza parametrii din model
        self.model.reset_parameters()
        # despacheteaza parametriii din config
        optimizer           = conf.get("optimizer", None)
        opt_hyperparameters = conf.get("opt_hyperparameters", None)
        lr                  = conf.get("lr", 0.001)
        lr_scheduler        = conf.get("lr_scheduler", None)
        lr_scheduler_hyperparameters = conf.get("lr_scheduler_hyperparameters", None)

        if (optimizer is None):
            warnings.warn("\n\nEmpty 'optimizer', name '{}'\n\n".format(name))
            return
        if (opt_hyperparameters is None):
            opt_hyperparameters = {}
        # seteaza optimizatorul
        optimizer = optimizer(self.model.parameters(), lr=lr, **opt_hyperparameters)
        # seteaza scheduler-ul
        if (lr_scheduler is not None):
            lr_scheduler(optimizer, **lr_scheduler_hyperparameters)
        # seteaza noul optimizator din config
        self.trainer.setOptimizer(optimizer)
        # antreneaza modelul
        start_time = time.time()
        logs = self.trainer.run(self.train_dl, self.val_dl, self.epochs, name)
        logs["name"] = name
        logs["time"] = time.time()-start_time
        # salveaza lgurile intr-un csv file
        self.callback(epoch=self.epochs, logs=logs)

    def __call__(self, **confs):
        for key_conf in confs.keys():
            conf = confs.get(key_conf)
            if (conf is None):
                conf = {}
            self.run(key_conf, **conf)

"""
name:(optimizer, (lr_scheduler=..., opt_hyperparameters=..., lr=float))
"""
