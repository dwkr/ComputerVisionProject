import numpy as np
from game_controls import *

def balance_data(data):
	left = []
	right = []
	straight = []
	reverse = []
	no_action = []

	for d in data:
		if d[1] == keysW:
			straight.append(d)
		elif d[1] == keysWA or d[1] == keysA:
			left.append(d)
		elif d[1] == keysWD or d[1] == keysD:
			right.append(d)
		elif d[1] == keysNK:
			no_action.append(d)
		else:
			reverse.append(d)

	straight = straight[:2 * len(left)][:2 * len(right)]
	left = left[:2 * len(straight)]
	right = right[:2 * len(straight)]
	no_action = no_action[:2 * len(straight)]
	balanced_data = straight + left + right + reverse + no_action
	
	stats = {
	"straight" : len(straight),
	"left" : len(left),
	"right" : len(right),
	"reverse" : len(reverse),
	"no_action" : len(no_action)
	}

	return stats, balanced_data
