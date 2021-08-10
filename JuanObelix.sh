#!/bin/sh
rsync -r -v  "$PWD/$1/"$2 "jdarias@obelix-h0.science.uva.nl:Code/Julia/IonTweezers/$1"
