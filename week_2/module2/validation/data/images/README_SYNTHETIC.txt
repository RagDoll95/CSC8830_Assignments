The synthetic_meas_*.jpg files here are RENDERED, not photographed.

They were produced by tools/make_synthetic_data.py as flat rectangular targets
of exactly known size at exactly known distance, so that the Step 3 statistics
pipeline can be scored against the true values. The accompanying
validation/data/measurements.csv was produced by tools/simulate_measurements.py,
which stands in for 20 manual clicks. See the warning at the top of
module2/README.md.

FOR SUBMISSION: delete the synthetic_* files and put your own 20 measurement
photos here -- each object imaged from beyond 2 m, centred in frame, at a
tape-measured distance, with ground truth taken by ruler BEFORE imaging. Log one
row per measurement in validation/data/measurements.csv:

    id,image_file,object,dimension,Z_mm,ground_truth_mm,measured_mm,u1,v1,u2,v2

Then run:
    python validation/analyze.py
