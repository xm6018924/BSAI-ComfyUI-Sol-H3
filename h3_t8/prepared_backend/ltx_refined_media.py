"""Bounded final video encode with untouched original AAC packets/timestamps."""
import json
from pathlib import Path
import subprocess


def audio_packets(path):
    result = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_packets", "-show_data_hash", "sha256", "-show_entries",
        "packet=pts_time,dts_time,duration_time,size,data_hash", "-of", "json", str(path)], text=True, timeout=60))
    rows = result.get("packets", [])
    if not rows:
        raise ValueError("Original delivery has no audio packets")
    return rows


def deliver(pixels, audio_source, destination, *, frame_count=73):
    import torch
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    if type(frame_count) is not int or not 9 <= frame_count <= 192 or (frame_count - 1) % 8:
        raise ValueError('Expected exact8n+1 output frames, up to8s')
    if pixels.ndim != 4 or pixels.shape[0] != frame_count or pixels.shape[-1] != 3:
        raise ValueError("Expected the complete declared RGB latent-grid output")
    _, height, width, _ = pixels.shape
    if height % 2 or width % 2 or min(height, width) < 16 or pixels.device.type != "cpu":
        raise ValueError("Expected even-sized CPU RGB frames")
    if not torch.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 1:
        raise ValueError("RGB pixels must be finite [0,1]")
    source_packets = audio_packets(audio_source)
    command = ["ffmpeg", "-v", "error", "-xerror", "-nostdin", "-n",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", "24", "-i", "pipe:0",
        "-i", str(audio_source), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264",
        "-preset", "fast", "-crf", "18", "-threads:v", "2", "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", str(destination)]
    with destination.with_suffix(".encode.log").open("xb") as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=log, stderr=log,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            for frame in pixels:
                process.stdin.write(frame.mul(255).round().to(torch.uint8).numpy().tobytes())
            process.stdin.close()
            if process.wait(timeout=180):
                raise RuntimeError("Refined-video encoder failed; see local encode log")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
    if source_packets != audio_packets(destination):
        raise RuntimeError("Original audio packets or timestamps changed during mux")
    subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-nostdin", "-i", str(destination), "-f", "null", "-"],
                   check=True, capture_output=True, timeout=120)
    info = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-show_format",
        "-of", "json", str(destination)], text=True, timeout=60))
    streams = [stream for stream in info["streams"] if stream["codec_type"] == "video"]
    if len(streams) != 1:
        raise RuntimeError("Expected exactly one final video stream")
    video = streams[0]
    if (int(video["nb_frames"]), video["r_frame_rate"], video["width"], video["height"]) != (frame_count, "24/1", width, height):
        raise RuntimeError("Final video geometry/fps/frame count changed")
    if abs(float(video["duration"]) - frame_count / 24) > 1e-5:
        raise RuntimeError("Final video timeline changed")
    return {"video_frames": frame_count, "width": width, "height": height, "fps": "24/1",
            "video_seconds": video["duration"], "container_seconds": info["format"]["duration"],
            "audio_packet_count": len(source_packets), "audio_packets_and_timestamps_exact": True,
            "audio_reencoded": False, "full_decode_pass": True, "encoder_threads": 2,
            "limits": "Mechanical media audit, not human picture/lipsync/quality acceptance"}
