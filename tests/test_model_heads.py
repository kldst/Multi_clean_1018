"""Model heads on tiny random inputs (CPU): temporal SMPL head shapes / gradients, DPT mask head chunking."""

import unittest

import torch

from multismpl.model.heads.person_mask_head import PersonMaskDPTHead
from multismpl.model.heads.pose_transformer_rel import relative_position_bucket
from multismpl.model.heads.smpl_temporal_head import (
    SMPLMultiQueryTemporalRelConfig,
    SMPLMultiQueryTransRotTemporalRelHead,
)


class TemporalRelHeadTest(unittest.TestCase):
    def test_relative_bucket_is_translation_invariant(self):
        times = torch.arange(5)
        first = relative_position_bucket(
            times[None] - times[:, None], num_buckets=16, max_distance=15
        )
        shifted = relative_position_bucket(
            (times + 17)[None] - (times + 17)[:, None],
            num_buckets=16,
            max_distance=15,
        )
        self.assertTrue(torch.equal(first, shifted))

    def test_output_shapes_and_gradient(self):
        config = SMPLMultiQueryTemporalRelConfig(
            transformer_depth=2,
            transformer_heads=4,
            transformer_mlp_dim=64,
            transformer_dim_head=8,
            transformer_dim=32,
            max_T=8,
            rel_num_buckets=16,
            rel_max_distance=15,
        )
        head = SMPLMultiQueryTransRotTemporalRelHead(
            dim_in=48, num_people=3, smpl_cfg=config
        )
        B, T, V, N, C = 2, 3, 2, 5, 48
        tokens = torch.randn(B * T, V, N + 1, C, requires_grad=True)
        outputs = head(
            [tokens],
            patch_start_idx=1,
            smpl_inputs={
                "temporal_num_frames": torch.full((B,), T),
                "views_per_frame": torch.full((B,), V),
            },
        )
        expected = {
            "smpl_pose": (B * T, 3, 72),
            "smpl_beta": (B * T, 3, 10),
            "mesh_translate": (B * T, 3, 3),
            "smpl_presence_logits": (B * T, 3),
            "person_tokens": (B * T, 3, 32),
        }
        for key, shape in expected.items():
            self.assertEqual(tuple(outputs[key].shape), shape)

        loss = sum(outputs[key].float().mean() for key in expected)
        loss.backward()
        self.assertIsNotNone(tokens.grad)
        self.assertTrue(torch.isfinite(tokens.grad).all())



class PersonMaskDPTChunkingTest(unittest.TestCase):
    @staticmethod
    def make_head(chunk_size):
        return PersonMaskDPTHead(
            dim_in=64,
            query_dim=32,
            embed_dim=16,
            features=32,
            out_channels=[32, 32, 32, 32],
            intermediate_layer_idx=[0, 1, 2, 3],
            down_ratio=2,
            patch_size=14,
            frames_chunk_size=chunk_size,
        )

    @staticmethod
    def make_inputs(requires_grad=False):
        torch.manual_seed(7)
        person_tokens = torch.randn(1, 3, 32, requires_grad=requires_grad)
        # 28x28 with patch_size=14 -> four patch tokens plus two special tokens.
        token_layers = [
            torch.randn(1, 2, 6, 64, requires_grad=requires_grad)
            for _ in range(4)
        ]
        images = torch.rand(1, 2, 3, 28, 28)
        return person_tokens, token_layers, images

    def test_chunked_forward_matches_unchunked(self):
        full = self.make_head(chunk_size=8).eval()
        chunked = self.make_head(chunk_size=1).eval()
        chunked.load_state_dict(full.state_dict())
        person_tokens, token_layers, images = self.make_inputs()
        with torch.no_grad():
            expected = full(person_tokens, token_layers, images, patch_start_idx=2)
            actual = chunked(person_tokens, token_layers, images, patch_start_idx=2)
        self.assertEqual(tuple(actual.shape), (1, 2, 3, 14, 14))
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_chunked_training_backward_is_finite(self):
        head = self.make_head(chunk_size=1).train()
        person_tokens, token_layers, images = self.make_inputs(requires_grad=True)
        logits = head(person_tokens, token_layers, images, patch_start_idx=2)
        logits.square().mean().backward()
        self.assertIsNotNone(head.pixel_proj.weight.grad)
        self.assertIsNotNone(head.trunk.projects[0].weight.grad)
        self.assertTrue(torch.isfinite(head.pixel_proj.weight.grad).all())
        self.assertTrue(torch.isfinite(head.trunk.projects[0].weight.grad).all())
        self.assertTrue(torch.isfinite(person_tokens.grad).all())

