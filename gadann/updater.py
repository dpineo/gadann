#
# GADANN - GPU Accelerated Deep Artificial Neural Network
#
# Copyright (C) 2014 Daniel Pineo (daniel@pineo.net)
# 
# Permission is hereby granted, free of charge, to any person obtaining a 
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, 
# and/or sell copies of the Software, and to permit persons to whom the 
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included 
# in all copies or substantial portions of the Software
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE.

import logging

from . import kernels

logger = logging.getLogger(__name__)


class Updater(object):
    def __init__(self):
        pass

    def update(self, args):
        for key in params.iterkeys():
            params[key] = params[key] + grads[key]*learning_rate


class SgdUpdater(Updater):
    def __init__(self, learning_rate=0.1, weight_cost=0.01):
        self.weight_cost = weight_cost
        self.learning_rate = learning_rate

    def update(self, params, grads):
        for k, v in grads.items():
            params[k] = params[k] - self.learning_rate*v - params[k]*self.weight_cost

    def status(self):
        return ''


class MomentumUpdater(Updater):
    def __init__(self, learning_rate=0.1, inertia=0.9, weight_cost=0.00):
        self.inertia = inertia
        self.weight_cost = weight_cost
        self.learning_rate = learning_rate

    def update(self, params, grads):
        try:
            self.velocities = [self.inertia*v + (1-self.inertia)*g for (v,g) in zip(self.velocities, grads)]
        except:
            self.velocities = [(1-self.inertia)*g for g in grads]

        for i in range(len(params)):
            params[i] = params[i] - self.learning_rate*self.velocities[i] - params[i]*self.weight_cost
        self.inertia += .001*(1-self.inertia)

    def status(self):
        return 'inertia:' + str(self.inertia)


class RmspropUpdater(Updater):
    def __init__(self, learning_rate=0.1, inertia=0.0, weight_cost=0.00):
        self.epsilon = 0.000001
        self.inertia = inertia
        self.weight_cost = weight_cost
        self.learning_rate = learning_rate

    def update(self, params, grads):
        try:
            self.accum = [self.inertia*a + (1-self.inertia)*(g**2) for (a,g) in zip(self.accum, grads)]
        except:
            self.accum = [(1-self.inertia)*(g**2) for g in grads]

        for i in range(len(params)):
            params[i] = params[i] - self.learning_rate * grads[i] / (kernels.sqrt(self.accum[i]) + self.epsilon) - params[i]*self.weight_cost
        self.inertia += .001*(1-self.inertia)

    def status(self):
        return 'inertia:' + str(self.inertia)
