from typing import List, Dict


def pl_samples_to_counts(samples: List[List[int]]) -> Dict[str, int]:
	"""
	Converts samples from Pennylane to the counts format from qiskit.

	:param samples: samples from the execution of a Pennylane circuit with qml.sample(qml.PauliZ(wires=i)) measurements
	:return: counts dictionary in the same format as get_counts() from qiskit returns
	"""
	counts = {}

	for shot in zip(*samples):
		shot_str = ""

		for v in reversed(shot):  # needs to be reversed, because that is the order that qiskit uses
			if v == 1:
				shot_str += "0"
			else:
				shot_str += "1"

		if shot_str in counts:
			counts[shot_str] = counts[shot_str] + 1
		else:
			counts[shot_str] = 1

	return counts
