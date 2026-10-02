"""Decode saved Tao normalized video with Core H3 VAE; reuse Base10 audio."""
import argparse
import json
from pathlib import Path
import sys
import time

from backend_files import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    request_file = parser.parse_args().request
    root = request_file.parent
    request = json.loads(request_file.read_text())
    report = {'status':'incomplete','scope':__doc__}
    try:
        assert request['schema'] == 't8-taomate-video-decode-v1'
        for path,value in request['identities'].items():
            assert sha(path) == value,path
        sys.path.insert(0,request['core'])
        sys.path.insert(0,str(Path(request['source'])/'src'))
        import comfy.cli_args
        comfy.cli_args.args.cpu = True  # explicit isolated VAE only below
        import torch
        import comfy.sd
        import comfy.utils
        from safetensors.torch import load_file,save_file
        from taomate_h3.inference.media_timing import endpoint_preserving_video_filter
        from taomate_media_delivery import deliver
        torch.set_num_threads(2)
        assert str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-').lower() == request['gpu_uuid'].removeprefix('GPU-').lower()
        tensors = load_file(request['latent'])
        z = tensors['video']
        known = load_file(request['milestones'])['state_9'].reshape(2,207,32).permute(0,2,1)
        assert torch.equal(tensors['audio'],known)
        assert z.shape == (1,24,37,30,54) and torch.isfinite(z).all()
        sd,metadata = comfy.utils.load_torch_file(request['vae'],return_metadata=True)
        vae = comfy.sd.VAE(sd=sd,metadata=metadata,device=torch.device('cpu'),dtype=torch.float16)
        del sd
        vae.throw_exception_if_invalid()
        assert vae.latent_channels == 24 and vae.handles_tiling
        model = vae.first_stage_model.eval().to('cuda:0')
        assert model.decode_output_shape(z.shape) == (1,3,124,480,864)
        calls = []
        handle = model.decoder.register_forward_pre_hook(lambda _m,inputs: calls.append(list(inputs[0].shape)))
        start = time.perf_counter()
        try:
            # Core internally reverses the normalized latent exactly once and
            # streams finalized [0,1] RGB chunks to a CPU output buffer.
            with torch.inference_mode():
                pixels = model.decode(z.to(device='cuda:0',dtype=torch.float16)).cpu()
            torch.cuda.synchronize()
        finally:
            handle.remove()
            model.cpu()
        report['decode_seconds'] = time.perf_counter()-start
        assert pixels.shape == (1,3,124,480,864) and torch.isfinite(pixels).all() and calls
        assert pixels.min() >= 0 and pixels.max() <= 1
        rgb = pixels[0].permute(1,2,3,0).contiguous()
        save_file({'pixels':rgb},str(root/'native-rgb.safetensors'))
        movie = root/'tao-first-request-5s.mp4'
        report['media'] = deliver(rgb.mul(255).round().to(torch.uint8).numpy(),request['audio'],movie,
            endpoint_preserving_video_filter(native_frames=124,published_frames=120))
        from PIL import Image,ImageDraw
        contact = Image.new('RGB',(1296,508),'#141c28')
        draw = ImageDraw.Draw(contact)
        for n,index in enumerate((0,24,49,74,99,123)):
            x,y = n%3*432,n//3*254
            draw.text((x+6,y+2),f'Native frame {index} | NOT HUMAN REVIEWED',fill='white')
            frame = Image.fromarray(rgb[index].mul(255).round().to(torch.uint8).numpy())
            contact.paste(frame.resize((432,240)),(x,y+14))
        contact.save(root/'contact.png')
        for path,value in request['identities'].items():
            assert sha(path) == value,path
        report.update(status='Tao_first_request_decode_media_pass_pending_review',
            decoder_calls=calls,reverse_normalization_count=1,native_frames=124,
            rgb_sha256=sha(root/'native-rgb.safetensors'),video_sha256=sha(movie),
            teacher_audio_bitexact_before_VAE=True,audio_regenerated=False,
            limits='Actual Tao video + preexisting Base10 audio. No human/quality or speed qualification. Core internal tiling; not official multi-GPU VAE parity.')
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        (root/'report.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({'status':report['status']}),flush=True)


if __name__ == '__main__':
    main()
