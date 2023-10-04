#!/bin/bash
set -e

mv build/_deps .
cd build
rm -rf *
mv ../_deps .
