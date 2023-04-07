import os
import whisper
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE
import argparse
import warnings
import yt_dlp
from .utils import slugify, str2bool, write_srt, write_vtt
import tempfile
import pathlib
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("video", nargs="+", type=str,
                        help="video URLs to transcribe")
    parser.add_argument("--model", default="small",
                        choices=whisper.available_models(), help="name of the Whisper model to use")
    parser.add_argument("--format", default="vtt",
                        choices=["vtt", "srt"], help="the subtitle format to output")
    parser.add_argument("--output_dir", "-o", type=str,
                        default=".", help="directory to save the outputs")
    parser.add_argument("--verbose", type=str2bool, default=False,
                        help="Whether to print out the progress and debug messages")
    parser.add_argument("--task", type=str, default="transcribe", choices=[
                        "transcribe", "translate"], help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
    parser.add_argument("--language", type=str, default=None, choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]),
                        help="language spoken in the audio, skip to perform language detection")

    parser.add_argument("--break-lines", type=int, default=0, 
                        help="Whether to break lines into a bottom-heavy pyramid shape if line length exceeds N characters. 0 disables line breaking.")

    args = parser.parse_args().__dict__
    model_name: str = args.pop("model")
    
    output_dir: str = pathlib.Path(args.pop("output_dir"))
    if not output_dir.is_dir():
        print("output_dir {output_dir} is not a dir")
        return
    
    subtitles_format: str = args.pop("format")
    os.makedirs(output_dir, exist_ok=True)

    if model_name.endswith(".en"):
        warnings.warn(
            f"{model_name} is an English-only model, forcing English detection.")
        args["language"] = "en"

    model = whisper.load_model(model_name, device=DEVICE)
    video_urls = args.pop("video")
    audios = get_audio(video_urls)
    break_lines = args.pop("break_lines")

    for title, audio_path in audios.items():
        warnings.filterwarnings("ignore")
        result = model.transcribe(audio_path, **args)
        warnings.filterwarnings("default")

        if (subtitles_format == 'vtt'):
            file_path = output_dir.joinpath(f"{pathlib.Path(audio_path).stem}.vtt")
        else:
            file_path = output_dir.joinpath(f"{pathlib.Path(audio_path).stem}.srt")

        if (subtitles_format == 'vtt'):
            with open(file_path, 'w', encoding="utf-8") as vtt:
                write_vtt(result["segments"], file=vtt, line_length=break_lines)

            print(f"Saved VTT to {file_path.resolve()}")
        else:
            with open(file_path, 'w', encoding="utf-8") as srt:
                write_srt(result["segments"], file=srt, line_length=break_lines)

            print(f"Saved SRT to {file_path.resolve()}")


def get_audio(urls):
    temp_dir = tempfile.gettempdir()

    ydl = yt_dlp.YoutubeDL({
        'quiet': True,
        'verbose': False,
        'format': 'bestaudio',
        "outtmpl": os.path.join(temp_dir, "%(id)s.%(ext)s"),
        'postprocessors': [{'preferredcodec': 'mp3', 'preferredquality': '192', 'key': 'FFmpegExtractAudio', }],
    })

    paths = {}

    for url in urls:
        result = ydl.extract_info(url, download=True)
        print(
            f"Downloaded video \"{result['title'].encode('utf-8')}\". Generating subtitles..."
        )
        paths[result["title"]] = os.path.join(temp_dir, f"{result['id']}.mp3")

    return paths


if __name__ == '__main__':
    main()
