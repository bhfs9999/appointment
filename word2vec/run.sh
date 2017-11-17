#!/bin/bash

BIN_DIR=./bin
TEXT_DATA=../../word2vec/word_bag.txt
VECTOR_DATA=../../word2vec/model/appointment-vector.bin

echo -- Training vectors...
time $BIN_DIR/word2vec -train $TEXT_DATA -output $VECTOR_DATA -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -sample 1e-3 -threads 12 -binary 1


echo -----------------------------------------------------------------------------------------------------
echo -- distance...

$BIN_DIR/distance $VECTOR_DATA
