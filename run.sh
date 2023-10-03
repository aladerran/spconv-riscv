#!/bin/bash

echo 
echo ============ running gemmini conv ============
spike --extension=gemmini build/test/baremetalC/gemmini_conv
echo
# echo ============ running torchsparse conv ============
