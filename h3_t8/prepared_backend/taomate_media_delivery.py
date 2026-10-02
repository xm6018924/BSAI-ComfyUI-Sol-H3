"""CPU-only native Tao timing/mux helper, with full output decode checks."""
import json
from pathlib import Path
import subprocess

import numpy as np


def deliver(pixels, audio_path, destination, video_filter, *, fps=24):
    """Accept native RGB uint8 frames; preserve fixed upstream timing filter."""
    if pixels.dtype != np.uint8 or pixels.ndim != 4 or pixels.shape[-1] != 3:
        raise ValueError('Expected uint8 [T,H,W,3] native RGB')
    if len(pixels) != 124 or fps != 24:
        raise ValueError('This pilot only accepts the native 124-frame first request')
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    height, width = pixels.shape[1:3]
    command = ['ffmpeg','-nostdin','-v','error','-xerror','-n',
        '-f','rawvideo','-pixel_format','rgb24','-video_size',f'{width}x{height}',
        '-framerate',str(fps),'-i','pipe:0','-i',str(audio_path),
        '-map','0:v:0','-map','1:a:0','-vf',video_filter,
        '-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p',
        '-c:a','aac','-b:a','192k','-movflags','+faststart',str(destination)]
    with destination.with_suffix('.encode.log').open('xb') as log:
        process = subprocess.Popen(command,stdin=subprocess.PIPE,stdout=log,stderr=log,
                                   creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:
            for frame in pixels:
                process.stdin.write(frame.tobytes())
            process.stdin.close()
            if process.wait(timeout=120):
                raise RuntimeError(f'FFmpeg failed; see {destination.with_suffix(".encode.log")}')
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
    subprocess.run(['ffmpeg','-nostdin','-v','error','-xerror','-i',str(destination),
                    '-f','null','-'],check=True,capture_output=True,timeout=90)
    info = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams',
        '-show_format','-of','json',str(destination)],text=True))
    video = next(s for s in info['streams'] if s['codec_type']=='video')
    audio = next(s for s in info['streams'] if s['codec_type']=='audio')
    assert (video['width'],video['height'],int(video['nb_frames']),video['r_frame_rate']) == (width,height,120,'24/1')
    assert float(video['duration']) == 5.0 and float(audio['duration']) == 5.0
    assert int(audio['sample_rate']) == 32000 and audio['channels'] == 2
    assert abs(float(info['format']['duration'])-5.0) < .001
    # AAC decoder may expose the padded last packet; evaluate only advertised
    # delivery samples, while separately retaining the stream duration above.
    pcm = subprocess.check_output(['ffmpeg','-nostdin','-v','error','-xerror','-i',str(destination),
        '-map','0:a:0','-af','atrim=end_sample=160000','-c:a','pcm_f32le','-f','f32le','-'],timeout=90)
    decoded = np.frombuffer(pcm,dtype='<f4').reshape(-1,2)
    assert decoded.shape == (160000,2) and np.isfinite(decoded).all()
    import soundfile as sf
    original,rate = sf.read(audio_path,dtype='float32',always_2d=True)
    assert rate == 32000 and original.shape == decoded.shape and np.isfinite(original).all()
    energy = float(np.sum(original.astype('float64')**2))
    error = float(np.sum((original.astype('float64')-decoded)**2))
    return dict(video_frames=120,video_seconds=5.0,audio_seconds=5.0,container_seconds=5.0,
        width=width,height=height,fps=24,sample_rate=32000,channels=2,
        delivery_filter=video_filter,full_ffmpeg_decode_pass=True,
        audio_snr_db=float(10*np.log10(energy/max(error,1e-30))) if energy else None,
        audio_encoded_once_from_saved_exact_delivery=True,audio_bitexact=False,
        limits='AAC is lossy. Decode/timing evidence is not listening, lipsync or image-quality acceptance.')
