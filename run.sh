#!/bin/bash

echo ============ running gemmini conv ============
spike --extension=gemmini build/test/baremetalC/gemmini_conv
echo
echo ============ running rpg_asynet conv ============
spike --extension=gemmini pk build/test/pkCXX/rpg_asynet/rpg_asynet_conv
echo
echo ============ running torchsparse conv ============
