import numpy as np
from game_controls import *


def distribute_data(data):
	left = []
	right = []
	straight = []
	reduce = []
	no_action = []

	for d in data:
		if d[1] == keysW:
			straight.append(d)
		elif d[] == keysWA or d[1] == keysA:
			left.append(d)
		elif d[1] == keysWD or d[1] == keysD:
			right.append(d)
		elif d[1] == keysR:
			no_action.append(d)
		else:
			reduce.append(d)

	return left, right, straight, reduce, no_action

def gen_stats(data):
	left, right, straight, reduce, no_action = distribute_data(data)
	stats = {
	"straight" : len(straight),
	"left" : len(left),
	"right" : len(right),
	"reduce" : len(reduce),
	"no_action" : len(no_action)
	}
	return stats

def balance_data(data):
	left, right, straight, reduce, no_action = distribute_data(data)

	straight = straight[:len(left)][:len(right)]
	left = left[:len(straight)]
	right = right[:len(straight)]
	no_action = no_action[:len(straight)]
	reduce = reduce[:len(straight)]
	balanced_data = straight + left + right + reduce + no_action
	
	stats = {
	"straight" : len(straight),
	"left" : len(left),
	"right" : len(right),
	"reduce" : len(reduce),
	"no_action" : len(no_action)
	}

	return stats, balanced_data
