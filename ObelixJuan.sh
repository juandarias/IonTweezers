#!/bin/sh
rsync -r -v  "jdarias@obelix-h0.science.uva.nl:Code/Julia/IonTweezers/$1/"$2 "$PWD/$1"
