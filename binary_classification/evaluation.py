#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 06:11:43 2020

@author: ibrahima
"""


plt.figure()
plt.plot(evaluation_us, label ="us")
plt.plot(evaluation_ls, label="ls")
plt.plot(evaluation_ms, label="ms")
plt.plot(evaluation_rs, label="rs")
plt.legend()

