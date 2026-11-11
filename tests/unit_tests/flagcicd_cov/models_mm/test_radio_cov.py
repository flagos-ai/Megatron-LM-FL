# Copyright (c) 2026, BAAI. All rights reserved.
"""RADIO ViT: patch embedding with position encodings, class-token interleaving for dynamic resolutions, temporal tubelet grouping, and FP8 class-token padding."""
import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.vision.radio import RADIOViTModel, fp8_pad_hook
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils, get_current_device

HIDDEN_SIZE = 64
PATCH_DIM = 14
IMG_SIZE = 224


def _vision_config(**kwargs):
    return TransformerConfig(
        num_layers=2,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=4,
        use_cpu_initialization=True,
        **kwargs,
    )


def _patchify(images):
    batch, channels = images.shape[0], images.shape[1]
    grid = IMG_SIZE // PATCH_DIM
    x = images.reshape(batch, channels, grid, PATCH_DIM, grid, PATCH_DIM)
    x = x.permute(0, 2, 4, 1, 3, 5)
    return x.reshape(batch, grid * grid, channels * PATCH_DIM * PATCH_DIM)


class TestRADIOFixedResolutionForward:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.layer_spec = get_gpt_layer_with_transformer_engine_spec()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_forward_applies_ln_pre_and_ln_post_around_decoder(self):
        from megatron.core.extensions.transformer_engine import TENorm

        model = RADIOViTModel(
            _vision_config(),
            self.layer_spec,
            ln_pre_impl=TENorm,
            ln_post_impl=TENorm,
            img_h=IMG_SIZE,
            img_w=IMG_SIZE,
            patch_dim=PATCH_DIM,
            add_class_token=False,
        )
        model.to(get_current_device())
        model.eval()
        images = torch.randn(2, 3, IMG_SIZE, IMG_SIZE, device=get_current_device())

        output = model(images)

        patches = _patchify(images)
        embedded, _ = model.embedder(patches)
        pos_enc = model.get_pos_enc(images.shape[0], None, (IMG_SIZE, IMG_SIZE))
        x = embedded + pos_enc
        x = model.ln_pre(x)
        x = x.permute(1, 0, 2).contiguous()
        x = model.decoder(x, attention_mask=None, packed_seq_params=None)
        x = x.permute(1, 0, 2).contiguous()
        expected = model.ln_post(x)
        assert output.shape == torch.Size([2, 256, HIDDEN_SIZE])
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)

    def test_forward_prepends_class_token_rows(self):
        model = RADIOViTModel(
            _vision_config(),
            self.layer_spec,
            img_h=IMG_SIZE,
            img_w=IMG_SIZE,
            patch_dim=PATCH_DIM,
            add_class_token=True,
            class_token_len=2,
        )
        model.to(get_current_device())
        model.eval()
        images = torch.randn(2, 3, IMG_SIZE, IMG_SIZE, device=get_current_device())

        output = model(images)

        patches = _patchify(images)
        embedded, _ = model.embedder(patches)
        pos_enc = model.get_pos_enc(images.shape[0], None, (IMG_SIZE, IMG_SIZE))
        x = torch.cat(
            [model.class_token.expand(images.shape[0], -1, -1), embedded + pos_enc], dim=1
        )
        x = x.permute(1, 0, 2).contiguous()
        x = model.decoder(x, attention_mask=None, packed_seq_params=None)
        expected = x.permute(1, 0, 2).contiguous()
        assert output.shape == torch.Size([2, 258, HIDDEN_SIZE])
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)

    def test_pos_dropout_of_one_drops_position_encoding_entirely(self):
        model = RADIOViTModel(
            _vision_config(),
            self.layer_spec,
            img_h=IMG_SIZE,
            img_w=IMG_SIZE,
            max_img_h=IMG_SIZE,
            max_img_w=IMG_SIZE,
            patch_dim=PATCH_DIM,
            add_class_token=False,
            pos_dropout=1.0,
        )
        model.to(get_current_device())
        model.train()
        patches = torch.randn(2, 256, 3 * PATCH_DIM * PATCH_DIM, device=get_current_device())

        patched, pos_enc = model.apply_pos_enc(patches, input_size=(IMG_SIZE, IMG_SIZE))

        torch.testing.assert_close(patched, patches, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(pos_enc, model.position_embeddings, rtol=1e-5, atol=1e-5)

    def test_pos_dropout_training_zeroes_a_seeded_subset_of_positions(self):
        model = RADIOViTModel(
            _vision_config(),
            self.layer_spec,
            img_h=IMG_SIZE,
            img_w=IMG_SIZE,
            max_img_h=IMG_SIZE,
            max_img_w=IMG_SIZE,
            patch_dim=PATCH_DIM,
            add_class_token=False,
            pos_dropout=0.5,
        )
        model.to(get_current_device())
        model.train()
        patches = torch.randn(2, 256, 3 * PATCH_DIM * PATCH_DIM, device=get_current_device())

        torch.manual_seed(31)
        dropped, _ = model.apply_pos_enc(patches, input_size=(IMG_SIZE, IMG_SIZE))

        torch.manual_seed(31)
        keeps = torch.rand(2, 1, 1, dtype=torch.float32, device=get_current_device()) > 0.5
        expected = patches + torch.where(
            keeps, model.position_embeddings, torch.zeros(1, device=get_current_device())
        )
        torch.testing.assert_close(dropped, expected, rtol=1e-6, atol=1e-6)

    def test_temporal_without_dynamic_resolution_is_rejected(self):
        model = RADIOViTModel(
            _vision_config(),
            self.layer_spec,
            img_h=IMG_SIZE,
            img_w=IMG_SIZE,
            patch_dim=PATCH_DIM,
            add_class_token=False,
            temporal_patch_dim=2,
        )
        model.to(get_current_device())
        images = torch.randn(2, 3, IMG_SIZE, IMG_SIZE, device=get_current_device())

        with pytest.raises(NotImplementedError, match="only supported for dynamic resolution"):
            model(images, num_frames=[1])


class TestRADIOPositionEncodings:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.layer_spec = get_gpt_layer_with_transformer_engine_spec()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _model(self, **kwargs):
        return RADIOViTModel(
            _vision_config(),
            self.layer_spec,
            img_h=IMG_SIZE,
            img_w=IMG_SIZE,
            max_img_h=IMG_SIZE,
            max_img_w=IMG_SIZE,
            patch_dim=PATCH_DIM,
            add_class_token=False,
            **kwargs,
        ).to(get_current_device())

    def _base_grid(self, model):
        return model.position_embeddings.reshape(1, 16, 16, -1).permute(0, 3, 1, 2)

    def test_max_grid_input_returns_position_embeddings_unchanged(self):
        model = self._model()

        pos_enc = model.get_pos_enc(2, None, None)

        torch.testing.assert_close(pos_enc, model.position_embeddings)

    def test_patch_idxs_gather_position_embeddings(self):
        model = self._model()
        patch_idxs = torch.tensor([[0, 5, 3], [2, 7, 1]], device=get_current_device())

        gathered = model.get_pos_enc(2, patch_idxs, (IMG_SIZE, IMG_SIZE))

        expanded = model.position_embeddings.expand(2, -1, -1)
        index = patch_idxs.unsqueeze(-1).expand(-1, -1, HIDDEN_SIZE)
        expected = torch.gather(expanded, dim=1, index=index)
        torch.testing.assert_close(gathered, expected)

    def test_window_select_crops_to_input_grid(self):
        model = self._model(has_cpe=False)

        pos_enc = model.get_pos_enc(1, None, (IMG_SIZE, 112))

        expected = self._base_grid(model)[..., :8].flatten(2).permute(0, 2, 1)
        assert pos_enc.shape == torch.Size([1, 128, HIDDEN_SIZE])
        torch.testing.assert_close(pos_enc, expected)

    def test_window_select_upscales_when_input_exceeds_max_grid(self):
        model = self._model(has_cpe=False)

        pos_enc = model.get_pos_enc(1, None, (512, 224))

        base = self._base_grid(model)
        expected = (
            torch.nn.functional.interpolate(
                base.float(), size=(36, 16), mode="bilinear", align_corners=False
            )
            .to(base.dtype)
            .flatten(2)
            .permute(0, 2, 1)
        )
        assert pos_enc.shape == torch.Size([1, 576, HIDDEN_SIZE])
        torch.testing.assert_close(pos_enc, expected)

    def test_interpolate_only_cpe_resizes_to_input_grid(self):
        model = self._model(interpolate_only_cpe=True)

        pos_enc = model.get_pos_enc(1, None, (IMG_SIZE, 112))

        base = self._base_grid(model)
        expected = (
            torch.nn.functional.interpolate(
                base.float(), size=(16, 8), mode="bilinear", align_corners=False
            )
            .to(base.dtype)
            .flatten(2)
            .permute(0, 2, 1)
        )
        torch.testing.assert_close(pos_enc, expected)

    def test_interpolate_only_cpe_with_aspect_ratio_select_crops_first(self):
        model = self._model(interpolate_only_cpe=True, cpe_aspect_ratio_select=True)

        pos_enc = model.get_pos_enc(1, None, (IMG_SIZE, 112))

        base = self._base_grid(model)
        cropped = base[..., :8]
        expected = (
            torch.nn.functional.interpolate(
                cropped.float(), size=(16, 8), mode="bilinear", align_corners=False
            )
            .to(base.dtype)
            .flatten(2)
            .permute(0, 2, 1)
        )
        torch.testing.assert_close(pos_enc, expected)

    def test_aspect_ratio_select_without_cpe_crops_wider_grid(self):
        model = self._model(has_cpe=False, cpe_aspect_ratio_select=True)

        pos_enc = model.get_pos_enc(1, None, (IMG_SIZE, 112))

        expected = self._base_grid(model)[..., :8].flatten(2).permute(0, 2, 1)
        assert pos_enc.shape == torch.Size([1, 128, HIDDEN_SIZE])
        torch.testing.assert_close(pos_enc, expected)

    def test_eval_cpe_interpolates_to_max_dim_then_crops(self):
        model = self._model()
        model.eval()

        pos_enc = model.get_pos_enc(1, None, (IMG_SIZE, 112))

        base = self._base_grid(model)
        expected = (
            torch.nn.functional.interpolate(
                base.float(), size=(16, 16), mode="bilinear", align_corners=False
            )
            .to(base.dtype)[..., :8]
            .flatten(2)
            .permute(0, 2, 1)
        )
        torch.testing.assert_close(pos_enc, expected)

    def test_training_cpe_augmentation_is_seeded(self):
        model = self._model()
        model.train()

        torch.manual_seed(17)
        augmented_first = model.get_pos_enc(2, None, (IMG_SIZE, 112))
        torch.manual_seed(17)
        augmented_second = model.get_pos_enc(2, None, (IMG_SIZE, 112))
        model.eval()
        eval_pos = model.get_pos_enc(2, None, (IMG_SIZE, 112))

        torch.testing.assert_close(augmented_first, augmented_second)
        assert not torch.allclose(augmented_first, eval_pos)


class TestRADIODynamicResolutionForward:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.layer_spec = get_gpt_layer_with_transformer_engine_spec()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _packed(self, cu_seqlens, max_seqlen=4):
        device = get_current_device()
        cu = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
        return PackedSeqParams(
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            max_seqlen_q=torch.tensor(max_seqlen, dtype=torch.int32, device=device),
            max_seqlen_kv=torch.tensor(max_seqlen, dtype=torch.int32, device=device),
            qkv_format='thd',
        )

    def test_forward_interleaves_class_tokens_and_updates_cu_seqlens(self):
        model = RADIOViTModel(
            _vision_config(),
            self.layer_spec,
            img_h=32,
            img_w=32,
            patch_dim=16,
            add_class_token=True,
            class_token_len=1,
            dynamic_resolution=True,
        ).to(get_current_device())
        model.eval()
        global_t = torch.zeros(1, 8, 3 * 16 * 16, device=get_current_device())
        global_t[0, :, :] = torch.arange(8, device=get_current_device()).unsqueeze(1).float()
        imgs_sizes = torch.tensor([[32, 32], [32, 32]], device=get_current_device())
        packed = self._packed([0, 4, 8])

        output = model(global_t, imgs_sizes=imgs_sizes, packed_seq_params=packed)

        embedded, _ = model.embedder(global_t)
        pos_per_image = model.get_pos_enc(1, None, (32, 32))
        x = embedded + pos_per_image.repeat(1, 2, 1)
        class_token = model.class_token.expand(1, -1, -1)
        expected_input = torch.cat([class_token, x[:, 0:4], class_token, x[:, 4:8]], dim=1).permute(
            1, 0, 2
        )
        expected = (
            model.decoder(
                expected_input.contiguous(), attention_mask=None, packed_seq_params=packed
            )
            .permute(1, 0, 2)
            .contiguous()
        )
        assert output.shape == torch.Size([1, 10, HIDDEN_SIZE])
        torch.testing.assert_close(packed.cu_seqlens_q, torch.tensor([0, 5, 10], dtype=torch.int32))
        torch.testing.assert_close(packed.max_seqlen_q, torch.tensor(5, dtype=torch.int32))
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)

    def test_temporal_grouping_routes_frames_into_tubelets(self):
        model = RADIOViTModel(
            _vision_config(),
            self.layer_spec,
            img_h=32,
            img_w=32,
            patch_dim=16,
            add_class_token=False,
            dynamic_resolution=True,
            temporal_patch_dim=2,
        ).to(get_current_device())
        model.eval()
        embedder_inputs = []
        model.embedder.register_forward_hook(
            lambda module, args, output: embedder_inputs.append(args[0])
        )
        global_t = torch.zeros(1, 20, 3 * 16 * 16, device=get_current_device())
        global_t[0, :, :] = torch.arange(20, device=get_current_device()).unsqueeze(1).float()
        imgs_sizes = torch.tensor([[32, 32]] * 5, device=get_current_device())

        output, out_imgs_sizes, out_num_frames = model(
            global_t,
            imgs_sizes=imgs_sizes,
            packed_seq_params=self._packed([0, 4, 8, 12, 16, 20]),
            num_frames=[1, 4],
        )

        image_rows = global_t[0, 0:4, :].repeat(1, 2)
        video_rows = global_t[0, 4:20, :]
        tubelet_0 = torch.cat([video_rows[0:4], video_rows[4:8]], dim=-1)
        tubelet_1 = torch.cat([video_rows[8:12], video_rows[12:16]], dim=-1)
        expected_input = torch.cat([image_rows, tubelet_0, tubelet_1], dim=0).unsqueeze(0)
        assert len(embedder_inputs) == 1
        torch.testing.assert_close(embedder_inputs[0], expected_input)
        assert out_imgs_sizes.tolist() == [[32, 32]] * 3
        assert out_num_frames == [1, 2]
        assert output.shape == torch.Size([1, 12, HIDDEN_SIZE])

    def test_separate_video_embedder_routes_image_and_video_chunks(self):
        model = RADIOViTModel(
            _vision_config(),
            self.layer_spec,
            img_h=32,
            img_w=32,
            patch_dim=16,
            add_class_token=False,
            dynamic_resolution=True,
            temporal_patch_dim=2,
            separate_video_embedder=True,
        ).to(get_current_device())
        model.eval()
        image_inputs = []
        video_inputs = []
        model.embedder.register_forward_hook(
            lambda module, args, output: image_inputs.append(args[0])
        )
        model.video_embedder.register_forward_hook(
            lambda module, args, output: video_inputs.append(args[0])
        )
        global_t = torch.zeros(1, 20, 3 * 16 * 16, device=get_current_device())
        global_t[0, :, :] = torch.arange(20, device=get_current_device()).unsqueeze(1).float()
        imgs_sizes = torch.tensor([[32, 32]] * 5, device=get_current_device())

        output, out_imgs_sizes, out_num_frames = model(
            global_t,
            imgs_sizes=imgs_sizes,
            packed_seq_params=self._packed([0, 4, 8, 12, 16, 20]),
            num_frames=[1, 4],
        )

        assert len(image_inputs) == 1
        torch.testing.assert_close(image_inputs[0], global_t[:, 0:4, :])
        video_rows = global_t[0, 4:20, :]
        expected_video = torch.cat(
            [
                torch.cat([video_rows[0:4], video_rows[4:8]], dim=-1),
                torch.cat([video_rows[8:12], video_rows[12:16]], dim=-1),
            ],
            dim=0,
        ).unsqueeze(0)
        assert len(video_inputs) == 2
        torch.testing.assert_close(video_inputs[0], expected_video[:, 0:4, :])
        torch.testing.assert_close(video_inputs[1], expected_video[:, 4:8, :])
        assert out_num_frames == [1, 2]
        assert output.shape == torch.Size([1, 12, HIDDEN_SIZE])


class TestRADIOFp8PadHook:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.model = RADIOViTModel(
            _vision_config(),
            get_gpt_layer_with_transformer_engine_spec(),
            img_h=IMG_SIZE,
            img_w=IMG_SIZE,
            patch_dim=PATCH_DIM,
            add_class_token=False,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_fp8_pad_hook_prepends_rows_up_to_sixteen(self):
        class_token = torch.randn(3, HIDDEN_SIZE)
        state_dict = {"vision_model.class_token": class_token.clone()}

        fp8_pad_hook(self.model, state_dict, "", None, False, [], [], [])

        padded = state_dict["vision_model.class_token"]
        assert padded.shape == torch.Size([16, HIDDEN_SIZE])
        torch.testing.assert_close(padded[13:], class_token)

    def test_fp8_pad_hook_pads_multiple_of_sixteen_unchanged(self):
        class_token = torch.randn(32, HIDDEN_SIZE)
        state_dict = {"vision_model.class_token": class_token.clone()}

        fp8_pad_hook(self.model, state_dict, "", None, False, [], [], [])

        torch.testing.assert_close(state_dict["vision_model.class_token"], class_token)

    def test_fp8_pad_hook_returns_without_class_token_key(self):
        state_dict = {"vision_model.embedder.weight": torch.zeros(1)}

        fp8_pad_hook(self.model, state_dict, "", None, False, [], [], [])

        assert set(state_dict.keys()) == {"vision_model.embedder.weight"}
