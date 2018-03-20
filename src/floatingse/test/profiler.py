import cProfile
import pstats

import floatingse.floating as ff

cProfile.run('ff.semiExample()','profout')
p = pstats.Stats('profout')
n = 40
# Clean up filenames for the report
p.strip_dirs()

p.sort_stats('cumulative').print_stats(n)
p.sort_stats('time').print_stats(n)


