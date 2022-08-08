#!/bin/bash
###
 # @Author: Egoist
 # @Date: 2022-08-08 09:26:44
 # @LastEditors: Egoist
 # @LastEditTime: 2022-08-08 09:34:23
 # @FilePath: /smp/check.sh
 # @Description: 
 #    if you are using conda to manage package 
 #    run 'source check.sh' to list dependency package and version
### 

conda list | grep -E '^(python|pytorch|cudatoolkit|numpy|pandas|matplotlib|tensorboard|requests|xlrd) '