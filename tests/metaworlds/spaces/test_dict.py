import pickle
import unittest

from metaworlds.spaces.dict import Dict
from metaworlds.spaces.discrete import Discrete


class TestDict(unittest.TestCase):
    def test_pickleable(self):
        motion_dict = {"position": Discrete(2), "velocity": Discrete(3)}
        obj = Dict(motion_dict)
        round_trip = pickle.loads(pickle.dumps(obj))
        assert round_trip
        assert round_trip.contains(motion_dict)
