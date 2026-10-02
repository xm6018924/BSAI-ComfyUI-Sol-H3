"""Join native clean request latents without blending or repeating transport tails."""


class NativeStreamAssembly:
    def __init__(self):
        self.videos = []
        self.audios = []
        self.native_frames = 0
        self.video_latents = 0
        self.audio_latents = 0

    def append(self, video, audio, known_audio, execution):
        import torch
        from taomate_h3.streaming.geometry import (
            direct_5s_plan,
            canonical_continuation_plan,
        )

        index = len(self.videos)
        plan = direct_5s_plan()
        if index:
            plan = canonical_continuation_plan(plan, request_index=index)
        expected = {
            "request_index": index,
            "native_frame_offset": self.native_frames,
            "video_latent_offset": self.video_latents,
            "audio_latent_offset": self.audio_latents,
            "published_native_frames": plan.native_frame_count,
            "published_video_latents": plan.phases[-1].video_latent_stop,
            "published_audio_latents_per_channel": plan.phases[-1].audio_latent_stop,
        }
        for name, value in expected.items():
            actual = getattr(execution, name)
            if type(actual) is not int or actual != value:
                raise ValueError(
                    f"Invalid stream execution {name}: {actual!r} != {value}"
                )
        vp = 2 if index else 0
        ap = 207 - expected["published_audio_latents_per_channel"]
        if (
            type(execution.video_transport_prefix_latents) is not int
            or execution.video_transport_prefix_latents != vp
            or type(execution.audio_transport_prefix_latents_per_channel) is not int
            or execution.audio_transport_prefix_latents_per_channel != ap
        ):
            raise ValueError("Invalid native transport prefix counts")
        if (
            not isinstance(video, torch.Tensor)
            or not isinstance(audio, torch.Tensor)
            or video.device.type != "cpu"
            or audio.device.type != "cpu"
            or not video.is_floating_point()
            or not audio.is_floating_point()
            or video.ndim != 5
            or video.shape[:3] != (1, 24, 37)
            or min(video.shape[3:]) < 1
            or audio.shape != (2, 32, 207)
            or not torch.isfinite(video).all()
            or not torch.isfinite(audio).all()
        ):
            raise ValueError("Expected finite CPU native transport AV tensors")
        if (
            not isinstance(known_audio, torch.Tensor)
            or known_audio.device.type != "cpu"
            or known_audio.dtype != audio.dtype
            or not torch.equal(audio[:, :, ap:], known_audio)
        ):
            raise ValueError("Published audio differs from the matching clean teacher")
        if index:
            previous_video, previous_audio = self.videos[-1], self.audios[-1]
            if (
                video.shape[3:] != previous_video.shape[3:]
                or video.dtype != previous_video.dtype
                or audio.dtype != previous_audio.dtype
            ):
                raise ValueError("Stream spatial geometry or dtype changed")
            if not torch.equal(
                video[:, :, :vp], previous_video[:, :, -vp:]
            ) or not torch.equal(audio[:, :, :ap], previous_audio[:, :, -ap:]):
                raise ValueError(
                    "Transport prefixes differ from previous accepted clean tails"
                )
            if (
                execution.cross_request_kv_reused is not True
                or execution.starting_history_tokens <= 0
            ):
                raise ValueError("Continuation must reuse actual previous KV history")
        elif (
            execution.cross_request_kv_reused is not False
            or execution.starting_history_tokens != 0
        ):
            raise ValueError("First request cannot adopt another task history")
        self.videos.append(video[:, :, vp:].contiguous().clone())
        self.audios.append(audio[:, :, ap:].contiguous().clone())
        self.native_frames += expected["published_native_frames"]
        self.video_latents += expected["published_video_latents"]
        self.audio_latents += expected["published_audio_latents_per_channel"]

    def joined(self):
        import torch

        if not self.videos:
            raise ValueError("Cannot publish an empty stream")
        return torch.cat(self.videos, dim=2), torch.cat(self.audios, dim=2)

    def timing(self):
        if not self.videos:
            raise ValueError("Cannot publish an empty stream")
        return dict(
            requests=len(self.videos),
            native_frames=self.native_frames,
            published_frames=120 * len(self.videos),
            fps=24,
            video_latents=self.video_latents,
            audio_latents=self.audio_latents,
            native_samples=self.audio_latents * 800,
            published_samples=160000 * len(self.videos),
            sample_rate=32000,
        )
