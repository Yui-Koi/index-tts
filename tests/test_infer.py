import unittest
from unittest.mock import patch, MagicMock
import torch
import torchaudio
from indextts.infer import IndexTTS

class TestInfer(unittest.TestCase):
    @patch('indextts.infer.IndexTTS.__init__', lambda self, **kwargs: None)
    def test_get_conditioning_mel(self):
        tts = IndexTTS()
        tts.device = 'cpu'
        tts.cache_cond_mel = None
        tts.cache_audio_prompt = None

        # Test with a dummy audio file
        audio = torch.randn(1, 22050)
        torchaudio.save("test.wav", audio, 22050)

        # First call should load the audio and compute the mel spectrogram
        cond_mel1 = tts._get_conditioning_mel("test.wav")
        self.assertIsNotNone(cond_mel1)

        # Second call should return the cached mel spectrogram
        cond_mel2 = tts._get_conditioning_mel("test.wav")
        self.assertIsNotNone(cond_mel2)

        # The two mel spectrograms should be the same
        self.assertTrue(torch.allclose(cond_mel1, cond_mel2))

        # Test with a different audio file
        audio = torch.randn(1, 22050)
        torchaudio.save("test2.wav", audio, 22050)

        # This should recompute the mel spectrogram
        cond_mel3 = tts._get_conditioning_mel("test2.wav")
        self.assertIsNotNone(cond_mel3)

        # The new mel spectrogram should be different from the first one
        self.assertFalse(torch.allclose(cond_mel1, cond_mel3))

    @patch('indextts.infer.IndexTTS.__init__', lambda self, **kwargs: None)
    def test_remove_long_silence(self):
        tts = IndexTTS()
        tts.stop_mel_token = 8193

        # Test with a simple case
        codes = torch.tensor([[52] * 40 + [1, 2, 3]])
        codes, code_lens = tts.remove_long_silence(codes)
        self.assertEqual(codes.shape[1], 13)
        self.assertEqual(code_lens[0], 13)

        # Test with a case that has no long silences
        codes = torch.tensor([[52] * 10 + [1, 2, 3]])
        codes, code_lens = tts.remove_long_silence(codes)
        self.assertEqual(codes.shape[1], 13)
        self.assertEqual(code_lens[0], 13)

        # Test with a case that has multiple long silences
        codes = torch.tensor([[52] * 40 + [1, 2, 3] + [52] * 40])
        codes, code_lens = tts.remove_long_silence(codes)
        self.assertEqual(codes.shape[1], 23)
        self.assertEqual(code_lens[0], 23)

    @patch('indextts.infer.IndexTTS.__init__', lambda self, **kwargs: None)
    def test_bucket_sentences(self):
        tts = IndexTTS()

        # Test with a simple case
        sentences = ["This is a short sentence.", "This is a slightly longer sentence.", "This is an even longer sentence.", "This is the longest sentence of them all."]
        buckets = tts.bucket_sentences(sentences, bucket_max_size=2)
        self.assertEqual(len(buckets), 2)
        self.assertEqual(len(buckets[0]), 2)
        self.assertEqual(len(buckets[1]), 2)

        # Test with a case that has only one bucket
        sentences = ["This is a short sentence.", "This is another short sentence."]
        buckets = tts.bucket_sentences(sentences, bucket_max_size=4)
        self.assertEqual(len(buckets), 1)
        self.assertEqual(len(buckets[0]), 2)

        # Test with a case that has multiple buckets of different sizes
        sentences = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
        buckets = tts.bucket_sentences(sentences, bucket_max_size=3)
        self.assertEqual(len(buckets), 4)
        self.assertEqual(len(buckets[0]), 3)
        self.assertEqual(len(buckets[1]), 3)
        self.assertEqual(len(buckets[2]), 3)
        self.assertEqual(len(buckets[3]), 1)

if __name__ == "__main__":
    unittest.main()
