import numpy as np

def balance_data(data):
	left = []
	right = []
	straight = []
	reverse = []
	no_action = []

	w = [1,0,0,0,0,0,0,0,0]
	a = [0,1,0,0,0,0,0,0,0]
	s = [0,0,1,0,0,0,0,0,0]
	d = [0,0,0,1,0,0,0,0,0]
	wa = [0,0,0,0,1,0,0,0,0]
	wd = [0,0,0,0,0,1,0,0,0]
	sa = [0,0,0,0,0,0,1,0,0]
	sd = [0,0,0,0,0,0,0,1,0]
	nk = [0,0,0,0,0,0,0,0,1]

	
	for d in data:
		if d[1] == w:
			straight.append(d)
		elif d[1] == wa or d[1] == a:
			left.append(d)
		elif d[1] == wd or d[1] == d:
			right.append(d)
		elif d[1] == nk:
			no_action.append(d)
		else:
			reverse.append(d)

	straight = straight[:len(left)][:len(right)]
	left = left[:len(straight)]
	right = right[:len(straight)]
	balanced_data = straight + left + right + reverse + no_action
	
	stats = {
	"straight" : len(straight),
	"left" : len(left),
	"right" : len(right),
	"reverse" : len(reverse),
	"no_action" : len(no_action)
	}

	return stats, balanced_data
