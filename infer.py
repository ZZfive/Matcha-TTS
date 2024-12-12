import datetime as dt
from pathlib import Path

import IPython.display as ipd
import numpy as np
import soundfile as sf
import torch
from tqdm.auto import tqdm

# Hifigan imports
from matcha.hifigan.config import v1
from matcha.hifigan.denoiser import Denoiser
from matcha.hifigan.env import AttrDict
from matcha.hifigan.models import Generator as HiFiGAN
# Matcha imports
from matcha.models.matcha_tts import MatchaTTS
from matcha.text import sequence_to_text, text_to_sequence
from matcha.utils.utils import intersperse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MATCHA_CHECKPOINT = "/root/code/Matcha-TTS/weights/matcha_ljspeech.ckpt"
HIFIGAN_CHECKPOINT = "/root/code/Matcha-TTS/weights/hifigan_T2_v1"
OUTPUT_FOLDER = "synth_output"


# 加载Matcha-TTS模型
def load_model(checkpoint_path):
    model = MatchaTTS.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    return model


# 加载Hifigan模型
def load_vocoder(checkpoint_path):
    h = AttrDict(v1)
    hifigan = HiFiGAN(h).to(device)
    hifigan.load_state_dict(torch.load(checkpoint_path, map_location=device)['generator'])
    _ = hifigan.eval()
    hifigan.remove_weight_norm()
    return hifigan


# 处理文本
@torch.inference_mode()
def process_text(text: str):
    x = torch.tensor(intersperse(text_to_sequence(text, ['english_cleaners2'])[0], 0), dtype=torch.long, device=device)[None]
    x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=device)
    x_phones = sequence_to_text(x.squeeze(0).tolist())  # 最终输入encoder中的音素序列
    return {
        'x_orig': text,  # 输入文本
        'x': x,  # 文本处理后音素序列对应的ids序列
        'x_lengths': x_lengths,  # 文本处理后音素序列的长度
        'x_phones': x_phones  # 文本处理后音素序列
    }


# 合成音频
@torch.inference_mode()
def synthesise(model, text, n_timesteps, temperature, length_scale, spks=None):
    text_processed = process_text(text)
    start_t = dt.datetime.now()
    output = model.synthesise(
        text_processed['x'], 
        text_processed['x_lengths'],
        n_timesteps=n_timesteps,
        temperature=temperature,
        spks=spks,
        length_scale=length_scale
    )
    # merge everything to one dict    
    output.update({'start_t': start_t, **text_processed})
    return output

# 将mel转换为音频
@torch.inference_mode()
def to_waveform(denoiser, mel, vocoder):
    audio = vocoder(mel).clamp(-1, 1)
    audio = denoiser(audio.squeeze(0), strength=0.00025).cpu().squeeze()
    return audio.cpu().squeeze()

# 保存mel和音频
def save_to_folder(filename: str, output: dict, folder: str):
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    np.save(folder / f'{filename}', output['mel'].cpu().numpy())
    sf.write(folder / f'{filename}.wav', output['waveform'], 22050, 'PCM_24')


if __name__ == "__main__":
    model = load_model(MATCHA_CHECKPOINT)
    vocoder = load_vocoder(HIFIGAN_CHECKPOINT)
    denoiser = Denoiser(vocoder, mode='zeros')

    texts = [
        "The Secret Service believed that it was very doubtful that any President would ride regularly in a vehicle with a fixed top, even though transparent."
    ]
    ## Number of ODE Solver steps
    n_timesteps = 10

    ## Changes to the speaking rate
    length_scale=1.0

    ## Sampling temperature
    temperature = 0.667

    outputs, rtfs = [], []
    rtfs_w = []
    for i, text in enumerate(tqdm(texts)):
        output = synthesise(model, text, n_timesteps, temperature, length_scale) #, torch.tensor([15], device=device, dtype=torch.long).unsqueeze(0))
        output['waveform'] = to_waveform(denoiser, output['mel'], vocoder)  # 将mel转换为音频

        # Compute Real Time Factor (RTF) with HiFi-GAN
        t = (dt.datetime.now() - output['start_t']).total_seconds()
        rtf_w = t * 22050 / (output['waveform'].shape[-1])

        ## Pretty print
        print(f"{'*' * 53}")
        print(f"Input text - {i}")
        print(f"{'-' * 53}")
        print(output['x_orig'])
        print(f"{'*' * 53}")
        print(f"Phonetised text - {i}")
        print(f"{'-' * 53}")
        print(output['x_phones'])
        print(f"{'*' * 53}")
        print(f"RTF:\t\t{output['rtf']:.6f}")
        print(f"RTF Waveform:\t{rtf_w:.6f}")
        rtfs.append(output['rtf'])
        rtfs_w.append(rtf_w)

        ## Display the synthesised waveform
        ipd.display(ipd.Audio(output['waveform'], rate=22050))

        ## Save the generated waveform
        save_to_folder(i, output, OUTPUT_FOLDER)

    print(f"Number of ODE steps: {n_timesteps}")
    print(f"Mean RTF:\t\t\t\t{np.mean(rtfs):.6f} ± {np.std(rtfs):.6f}")
    print(f"Mean RTF Waveform (incl. vocoder):\t{np.mean(rtfs_w):.6f} ± {np.std(rtfs_w):.6f}")
