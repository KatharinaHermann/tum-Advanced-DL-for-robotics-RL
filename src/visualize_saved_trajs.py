import os
import fnmatch

from hwr.utils import visualize_trajectory


# trajectory specification:
outcome = 'successful'  #['successful', 'unsuccessful', 'unfinished']
assert outcome in ['successful', 'unsuccessful', 'unfinished']
step = 320000
epi = 2

dirname = os.path.join('results', outcome + '_trajs')
prefix = "step_{0:08d}_epi_{1:02d}*".format(step, epi)

filename = None
for file in os.listdir(dirname):
    if fnmatch.fnmatch(file, prefix):
        filename = os.path.join(dirname, file)

assert filename is not None, 'No filename has been found in {} that contains the string{}'.format(
    dirname, prefix)

visualize_trajectory(filename)


