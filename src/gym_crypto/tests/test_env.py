import unittest
from ..env import Env
import os

package_path = os.path.abspath('./')
TEST_DATA = package_path + '/gym_crypto/tests/test_data/'


class GymTest(unittest.TestCase):
    def setUp(self):
        self.env = Env(test=TEST_DATA)
        self.env.make(['zilliqa'])

    def test_step(self):
        env = self.env
        start_state= env.reset()
        action = 1
        _, start_reward, _, _, = env.step(action)
        for i in range(5):
            new_state, reward, fin, options = env.step(action)
            if fin:
                break
        last_state, final_reward, fin, _ = env.step(0)
        self.assertNotEqual(start_reward, final_reward)


