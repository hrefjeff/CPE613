To view your images without downloading,
  gio open <your-image-here>

Make sure you have X11 forwarding working.
If you are in Moba, this should be the default.
If you are in a terminal, make sure you ssh'ed with
  ssh -X <username>@<ASA DMC address>

Finally, I build OpenCV to be compatible with the 
cuda compiler, so make sure to run 
  module load cuda
before getting started. This will add some dependencies
to the appropriate paths.
