#!/bin/bash
set -e

# echo 
# echo ============ running gemmini conv ============
# spike --extension=gemmini build/test/baremetalC/gemmini_conv
echo
echo ================ running torchsparse conv ================
spike --extension=gemmini pk build/test/pkCXX/torchsparse/torchsparse_conv
