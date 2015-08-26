#!/usr/bin/python
# Extract frames from videos in a folder

import os, argparse
import os.path as path
import time as t
import subprocess as subp
import pandas as pd
from shutil import copytree, ignore_patterns


def extract_frames(indir, outdir, fps):
  ''' Loop through input directory's and extract  '''
  for root, dirs, files in os.walk(indir, topdown = False):
    for file in files:
      outfilename = file[:len(file)-4]
      print indir, outdir, outfilename
      outpath = path.join(root.replace(indir, outdir) + '/', outfilename)

      if not os.path.exists(outpath):
        os.makedirs(outpath)
      print outpath
      ffmpeg_options = ['ffmpeg', '-i', root + '/' + file,\
                                '-vf', 'fps=%s'%fps,\
                                '%s/%s-%%d.jpg'%(outpath,outfilename)]
      print ' '.join(ffmpeg_options)
      raw_input()
      fpipe = subp.Popen(ffmpeg_options,stdout=subp.PIPE,stderr=subp.PIPE)
      fpipe.communicate()


def extract_med(outdir, fps):
    csv_file = "/home/syq/trecvid/EVENTS-PS-100Ex_20140513_JudgementMD.csv"
    video_list = pd.read_csv(csv_file, dtype=str)
    for video in video_list.iterrows():
        event_id = video[1]['EventID']
        clip_id = video[1]['ClipID']
        filename = "HVC" + clip_id + ".mp4"
        outfilename = filename[:len(filename)-4]
        if event_id > 'E030':
            root = "/media/syq/LDC2014E26/LDCDIST/LDC2013E115/events"
        else:
            root = "/media/syq/LDC2014E26/LDCDIST/LDC2012E01/events"
        video_path = path.join(root, event_id, filename)
        #print video_path
        outpath = path.join(outdir, event_id, outfilename)
        if not os.path.exists(outpath):
            os.makedirs(outpath)

        ffmpeg_options = ['ffmpeg', '-i', video_path,\
                                    '-vf', 'fps=%s'%fps,\
                                    '-s', '341x256',\
                                    '%s/%%d.jpg'%(outpath)]

        fpipe = subp.Popen(ffmpeg_options,stdout=subp.PIPE,stderr=subp.PIPE)
        fpipe.communicate()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Loops through all videos in the input folder and extract their key frames')
  parser.add_argument("-fps", "--frame", nargs="?", default="1/5",
                      help="rate to extract frames, e.g. 1/5 = 1 frame every 5 seconds, default=1/5")
  parser.add_argument("input", nargs="?", default="input/",
                      help="path to the input folder of videos")
  parser.add_argument("output", nargs="?", default="output-frames/",
                      help="path to the output folder")

  args = parser.parse_args()

  indir = args.input
  outdir = args.output
  fps = args.frame

  ts = t.time()

  """
  if not os.path.exists(outdir):
    copytree(indir, outdir, ignore=ignore_patterns('*.flv', '*.avi', '*.mp4'))
  """
  #extract_frames(indir, outdir, fps)
  extract_med(outdir, fps)

  te = t.time()
  print 'Total time for frame extraction' , str(te - ts) , 'seconds'
