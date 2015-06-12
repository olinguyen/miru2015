#!/bin/bash
# Run OB on all folders in dirlist

NUMVIDEOS=6766
NUMCORES=19

# Run object bank on a video folder
#./ClusterExmaple.sh dirlist ../Indoor_Images/ outputs/ 17
run_objectbank() {
  CTR=0
  DIRFILE="hmdb51-dirlist.txt";
#  INPUTBASEDIR="/home/syq/olivier/dataset/ucf50-frames-1fps/";
  INPUTBASEDIR="/home/syq/fudan/hmdb51/frame/";
  OUTPUTBASEDIR="ob_hmdb51_feats/";
  OBJECTBANK_BIN="/home/syq/research_final/bin/OBmain"
  while read dir; do
    if [ $CTR -eq $1 ];
    then
      CATEGORYDIR=$(echo "$dir" | cut -d "/" -f1)
      # If were using the CCV dataset, CATEGORYDIR=$dir
      mkdir -p $OUTPUTBASEDIR$CATEGORYDIR
      "$OBJECTBANK_BIN" "$INPUTBASEDIR$dir/" "$OUTPUTBASEDIR$CATEGORYDIR/"
    fi
  CTR=$(($CTR+1))
  done<"$DIRFILE"
}

echo "Running Object Bank on dataset..."
export -f run_objectbank
parallel -j $NUMCORES run_objectbank ::: $(seq 666 $NUMVIDEOS)
