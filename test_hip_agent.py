from hip_agent import HIPAgent
import unittest


class HipAgentTests(unittest.TestCase):

  def setUp(self):
    self.agent = HIPAgent(use_rag=False)
    self.answer_choices = [
        'uses labeled ddNTPs', 'uses only dideoxynucleotides',
        'uses only deoxynucleotides', 'uses labeled dNTPs'
    ]

  def test_format_options(self):
    # Test that answer_choices list gets formatted into the string we expect so it aligns
    # with the System prompt.
    self.assertEqual(
        self.agent._format_options(self.answer_choices),
        "Option A: uses labeled ddNTPs\nOption B: uses only dideoxynucleotides\nOption C: uses only deoxynucleotides\nOption D: uses labeled dNTPs"
    )

  def test_get_index(self):
    # Test that we match the response from the model to the appropriate index.
    response1 = "Option A"
    self.assertEqual(
        self.agent._get_index_of_response(response1, self.answer_choices), 0)

    response2 = "Answer: Option A"
    self.assertEqual(
        self.agent._get_index_of_response(response2, self.answer_choices), 0)

    response3 = "Option A: uses labeled ddNTPs"
    self.assertEqual(
        self.agent._get_index_of_response(response3, self.answer_choices), 0)

    response4 = "It uses labeled ddNTPs"
    self.assertEqual(
        self.agent._get_index_of_response(response4, self.answer_choices), 0)


if __name__ == '__main__':
  unittest.main()

