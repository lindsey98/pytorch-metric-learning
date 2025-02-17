import unittest

import torch

from pytorch_metric_learning.utils import loss_and_miner_utils as lmu

from .. import TEST_DEVICE, TEST_DTYPES


class TestLossAndMinerUtils(unittest.TestCase):
    def test_logsumexp(self):
        for dtype in TEST_DTYPES:
            rtol = 1e-2 if dtype == torch.float16 else 1e-5
            mat = torch.tensor(
                [
                    [-1, 0, 1, 10, 50],
                    [-300, -200, -100, -50, -20],
                    [-300, -200, 0, 200, 300],
                    [100, 200, 300, 400, 500],
                    [0, 0, 0, 0, 0],
                ],
                dtype=dtype,
                requires_grad=True,
            ).to(TEST_DEVICE)
            result = lmu.logsumexp(mat, keep_mask=None, add_one=False, dim=1)
            torch.mean(result).backward(retain_graph=True)
            correct_result = torch.logsumexp(mat, dim=1, keepdim=True)
            self.assertTrue(torch.allclose(result, correct_result, rtol=rtol))

            result = lmu.logsumexp(mat, keep_mask=None, add_one=True, dim=1)
            torch.mean(result).backward(retain_graph=True)
            correct_result = torch.logsumexp(
                torch.cat(
                    [
                        mat,
                        torch.zeros(mat.size(0), dtype=dtype)
                        .to(TEST_DEVICE)
                        .unsqueeze(1),
                    ],
                    dim=1,
                ),
                dim=1,
                keepdim=True,
            )
            self.assertTrue(torch.allclose(result, correct_result, rtol=rtol))

            keep_mask = torch.tensor(
                [
                    [1, 1, 0, 0, 0],
                    [1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0],
                ],
                dtype=torch.bool,
            ).to(TEST_DEVICE)
            result = lmu.logsumexp(mat, keep_mask=keep_mask, add_one=False, dim=1)
            torch.mean(result).backward()

            row0_input = torch.tensor([-1, 0], dtype=dtype).to(TEST_DEVICE)
            row1_input = torch.tensor([-300, -200, -100, -50, -20], dtype=dtype).to(
                TEST_DEVICE
            )
            row2_input = torch.tensor([-200, 0, 200], dtype=dtype).to(TEST_DEVICE)
            row4_input = torch.tensor([0, 0], dtype=dtype).to(TEST_DEVICE)

            row0 = torch.logsumexp(row0_input, dim=0).unsqueeze(0)
            row1 = torch.logsumexp(row1_input, dim=0).unsqueeze(0)
            row2 = torch.logsumexp(row2_input, dim=0).unsqueeze(0)
            row3 = torch.tensor([0.0], dtype=dtype).to(TEST_DEVICE)
            row4 = torch.logsumexp(row4_input, dim=0).unsqueeze(0)
            correct_result = torch.stack([row0, row1, row2, row3, row4], dim=0)
            self.assertTrue(torch.allclose(result, correct_result, rtol=rtol))

    def test_get_all_pairs_triplets_indices(self):
        original_x = torch.arange(10)

        for i in range(1, 11):
            x = original_x.repeat(i)
            correct_num_pos = len(x) * (i - 1)
            correct_num_neg = len(x) * (len(x) - i)
            a1, p, a2, n = lmu.get_all_pairs_indices(x)
            self.assertTrue(len(a1) == len(p) == correct_num_pos)
            self.assertTrue(len(a2) == len(n) == correct_num_neg)

            correct_num_triplets = len(x) * (i - 1) * (len(x) - i)
            a, p, n = lmu.get_all_triplets_indices(x)
            self.assertTrue(len(a) == len(p) == len(n) == correct_num_triplets)

    def test_convert_to_triplets(self):
        a1 = torch.LongTensor([0, 1, 2, 3])
        p = torch.LongTensor([4, 4, 4, 4])
        a2 = torch.LongTensor([4, 5, 6, 7])
        n = torch.LongTensor([5, 5, 6, 6])
        triplets = lmu.convert_to_triplets((a1, p, a2, n), labels=torch.arange(7))
        self.assertTrue(all(len(x) == 0 for x in triplets))

        a2 = torch.LongTensor([0, 4, 5, 6])
        triplets = lmu.convert_to_triplets((a1, p, a2, n), labels=torch.arange(7))
        self.assertTrue(
            triplets == (torch.tensor([0]), torch.tensor([4]), torch.tensor([5]))
        )

        a1 = torch.LongTensor([0, 1, 0, 2])
        p = torch.LongTensor([5, 6, 7, 8])
        a2 = torch.LongTensor([0, 1, 2, 0])
        n = torch.LongTensor([9, 10, 11, 12])
        triplets = lmu.convert_to_triplets((a1, p, a2, n), labels=torch.arange(13))
        triplets = torch.stack(triplets, dim=1)
        found_set = set()
        for t in triplets:
            found_set.add(tuple(t.cpu().numpy()))
        correct_triplets = {
            (0, 5, 9),
            (0, 5, 12),
            (0, 7, 9),
            (0, 7, 12),
            (1, 6, 10),
            (2, 8, 11),
        }

        self.assertTrue(found_set == correct_triplets)

    def test_convert_to_weights(self):
        a = torch.LongTensor([0, 1, 2, 3]).to(TEST_DEVICE)
        p = torch.LongTensor([4, 4, 4, 4]).to(TEST_DEVICE)
        n = torch.LongTensor([5, 5, 6, 6]).to(TEST_DEVICE)
        for dtype in TEST_DTYPES:
            weights = lmu.convert_to_weights(
                (a, p, n), labels=torch.arange(7).to(TEST_DEVICE), dtype=dtype
            )
            correct_weights = torch.tensor(
                [0.25, 0.25, 0.25, 0.25, 1, 0.5, 0.5], dtype=dtype
            ).to(TEST_DEVICE)
            self.assertTrue(torch.all(weights == correct_weights))

        a = torch.LongTensor([]).to(TEST_DEVICE)
        p = torch.LongTensor([]).to(TEST_DEVICE)
        n = torch.LongTensor([]).to(TEST_DEVICE)
        for dtype in TEST_DTYPES:
            weights = lmu.convert_to_weights(
                (a, p, n), labels=torch.arange(7).to(TEST_DEVICE), dtype=dtype
            )
            correct_weights = torch.tensor([1, 1, 1, 1, 1, 1, 1], dtype=dtype).to(
                TEST_DEVICE
            )
            self.assertTrue(torch.all(weights == correct_weights))

    def test_get_random_triplet_indices(self):
        for dtype in TEST_DTYPES:
            for _ in range(10):
                labels = (
                    torch.randint(low=0, high=5, size=(100,))
                    .to(TEST_DEVICE)
                    .type(dtype)
                )
                a, p, n = lmu.get_random_triplet_indices(labels)

                l_a, l_p, l_n = labels[a], labels[p], labels[n]
                self.assertTrue(torch.all(a != p))
                self.assertTrue(torch.all(l_a == l_p))
                self.assertTrue(torch.all(l_p != l_n))

                labels = (
                    torch.randint(low=0, high=5, size=(100,))
                    .to(TEST_DEVICE)
                    .type(dtype)
                )
                ref_labels = (
                    torch.randint(low=0, high=5, size=(50,)).to(TEST_DEVICE).type(dtype)
                )
                a, p, n = lmu.get_random_triplet_indices(labels, ref_labels=ref_labels)

                l_a = labels[a]
                l_p = ref_labels[p]
                l_n = ref_labels[n]
                self.assertTrue(torch.all(l_a == l_p))
                self.assertTrue(torch.all(l_p != l_n))

                # Test one-hot weights.
                labels = torch.randint(0, 3, (10,)).to(TEST_DEVICE).type(dtype)
                w = torch.zeros((len(labels), len(labels))).to(TEST_DEVICE).type(dtype)
                for i, label in enumerate(labels):
                    ind = torch.where(labels != label)[0]
                    j = ind[torch.randint(0, len(ind), (1,))[0]]
                    w[i, j] = 1.0
                a, p, n = lmu.get_random_triplet_indices(
                    labels, t_per_anchor=1, weights=w
                )
                self.assertTrue(torch.all(w[a, n] == 1.0))

                # Test one-hot weights.
                labels = torch.randint(0, 3, (100,)).to(TEST_DEVICE).type(dtype)
                ref_labels = torch.randint(0, 3, (50,)).to(TEST_DEVICE).type(dtype)
                w = (
                    torch.zeros((len(labels), len(ref_labels)))
                    .to(TEST_DEVICE)
                    .type(dtype)
                )
                for i, label in enumerate(labels):
                    ind = torch.where(ref_labels != label)[0]
                    j = ind[torch.randint(0, len(ind), (1,))[0]]
                    w[i, j] = 1.0
                a, p, n = lmu.get_random_triplet_indices(
                    labels, ref_labels=ref_labels, t_per_anchor=1, weights=w
                )
                self.assertTrue(torch.all(w[a, n] == 1.0))

                # test no possible triplets
                labels = torch.arange(100).to(TEST_DEVICE).type(dtype)
                a, p, n = lmu.get_random_triplet_indices(labels)
                self.assertTrue(len(a) == len(p) == len(n) == 0)

                labels = torch.zeros(100).to(TEST_DEVICE).type(dtype)
                a, p, n = lmu.get_random_triplet_indices(labels)
                self.assertTrue(len(a) == len(p) == len(n) == 0)

                labels = torch.zeros(100).to(TEST_DEVICE).type(dtype)
                ref_labels = torch.ones(100).to(TEST_DEVICE).type(dtype)
                a, p, n = lmu.get_random_triplet_indices(labels, ref_labels=ref_labels)
                self.assertTrue(len(a) == len(p) == len(n) == 0)

                labels = torch.zeros(100).to(TEST_DEVICE).type(dtype)
                ref_labels = torch.zeros(100).to(TEST_DEVICE).type(dtype)
                a, p, n = lmu.get_random_triplet_indices(labels, ref_labels=ref_labels)
                self.assertTrue(len(a) == len(p) == len(n) == 0)

                labels = torch.zeros(100).to(TEST_DEVICE).type(dtype)
                ref_labels = torch.ones(100).to(TEST_DEVICE).type(dtype)
                ref_labels[0] = 0
                a, p, n = lmu.get_random_triplet_indices(labels, ref_labels=ref_labels)
                self.assertTrue(len(a) == len(p) == len(n) == 100)


if __name__ == "__main__":
    unittest.main()
