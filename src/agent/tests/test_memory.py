import unittest
from ..memory import RingMemory

class TestMemory(unittest.TestCase):
    def test_append(self):
        memory = RingMemory(10, 2)
        memory.append([1,2])
        self.assertEqual(memory.data[0][0], 1)

    def test_sample(self):
        memory = RingMemory(10, 2)
        for i in range(1, 5):
            memory.append([i, i+10])
        sample = memory.sample(4)
        self.assertEqual(len(sample), 4)
        self.assertEqual(len(sample[0]), 2)
        self.assertEqual(len(sample[sample != 0]), len(sample.flatten()))


    def test_full_append(self):
        memory = RingMemory(10, 2)
        memory = self._fill_memory(memory)
        self.assertEqual(len(memory.data), 10)
        self.assertTrue(memory.data[0][0] < memory.data[0][1])


    def test_full_sample(self):
        memory = RingMemory(10, 2)
        memory = self._fill_memory(memory)
        sample = memory.sample(10)
        self.assertEqual(len(sample[sample != 0]), len(sample.flatten()))

    def _fill_memory(self, memory):
        for i in range(12):
            memory.append([i+10, i+11])
        return memory




