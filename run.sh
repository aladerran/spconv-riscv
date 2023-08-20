#!/bin/bash

echo ============ running gemmini conv ============
spike --extension=gemmini build/test/baremetalC/conv
echo
echo ============ running sparse conv ============
spike --extension=gemmini pk build/test/pkCXX/sparse_conv
