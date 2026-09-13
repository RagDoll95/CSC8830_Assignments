The synthetic_board_*.jpg files here are RENDERED, not photographed.

They were produced by tools/make_synthetic_data.py from a known ground-truth K
and distortion vector, so that calibration/calibrate.py can be scored against
the true values rather than merely run. See the warning at the top of
module2/README.md.

FOR SUBMISSION: delete the synthetic_* files and put your own 15-20 checkerboard
photos here, all at one resolution -- the same resolution used in Steps 2 and 3.
Then run:
    python calibration/capture_check.py
    python calibration/calibrate.py --square-size <your board's mm> \
        --notes "<phone model>, <capture app>, <resolution>, AE/AF locked"
