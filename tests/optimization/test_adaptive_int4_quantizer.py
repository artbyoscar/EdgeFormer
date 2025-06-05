import unittest
import torch

from src.optimization.dynamic_quantization import Int4Quantizer

# Alias for clarity: in future AdaptiveInt4Quantizer may extend Int4Quantizer
AdaptiveInt4Quantizer = Int4Quantizer

class TestAdaptiveInt4Quantizer(unittest.TestCase):
    def setUp(self):
        self.quantizer = AdaptiveInt4Quantizer()

    def test_pack_unpack_roundtrip(self):
        values = torch.tensor([-8, -4, 0, 3, 7, 1], dtype=torch.float32)
        packed = self.quantizer.pack_int4(values)
        unpacked = self.quantizer.unpack_int4(packed)
        self.assertTrue(torch.allclose(values, unpacked))

    def test_quantize_dequantize(self):
        tensor = torch.randn(4, 4)
        qdata = self.quantizer.quantize(tensor)
        deq = self.quantizer.dequantize(qdata)
        self.assertTrue(torch.allclose(tensor, deq, atol=1e-1))
        ratio = self.quantizer.get_compression_ratio(tensor, qdata)
        self.assertGreater(ratio, 1.0)

if __name__ == '__main__':
    unittest.main()
