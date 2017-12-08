import glob, os

def countIters(file):
	f = open(file, 'r')
	count = 0
	for line in f.readlines():
		if "STATE explore" in line:
			count += 1
	return count

if __name__ == "__main__":
	iters = 0
	for file in glob.glob("*.txt"):
		train_steps = countIters(file)
		iters += train_steps
		print("Found file: %s, has %d train steps" % (file, train_steps))
	print("Total training: %d steps" % (iters))
