import os
import fnmatch
import deep_learning

tests = [file for file in os.listdir(os.getcwd()) if fnmatch.fnmatch(file, 'test_*.py')]
tests.remove('test_all.py')

for test in tests:
	print '---------- '+test+' ----------'
	execfile(test)
