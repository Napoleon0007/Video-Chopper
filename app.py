import os
import re
import threading
import tempfile
import subprocess
from collections import defaultdict
from difflib import SequenceMatcher

import numpy as np
import soundfile as sf
import imageio_ffmpeg
from flask import Flask, render_template, request, jsonify, send_file

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
os.environ['IMAGEIO_FFMPEG_EXE'] = FFMPEG

from moviepy import VideoFileClip, AudioFileClip, ImageClip, concatenate_videoclips
from faster_whisper import WhisperModel

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 600 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

job = {'status': 'idle', 'message': 'Ready.', 'percent': 0, 'output': None, 'segments': 0}
job_lock = threading.Lock()

_whisper_model = None
_whisper_lock = threading.Lock()


def get_whisper(model_name='base'):
    """Lazy-load Whisper. Cached for the process lifetime."""
    global _whisper_model
    with _whisper_lock:
        if _whisper_model is None:
            _whisper_model = WhisperModel(model_name, device='cpu', compute_type='int8')
        return _whisper_model


def upd(**kwargs):
    with job_lock:
        job.update(kwargs)


def normalize_word(w):
    return re.sub(r'[^\w]', '', str(w).strip().lower())


def transcode_to_wav16(src_path):
    """Whisper expects 16kHz mono."""
    tmp = tempfile.mktemp(suffix='.wav')
    subprocess.run(
        [FFMPEG, '-y', '-i', src_path, '-ar', '16000', '-ac', '1', '-f', 'wav', tmp],
        capture_output=True, check=True
    )
    return tmp


def transcribe_words(model, audio_path):
    """Run Whisper and return a flat list of word-level dicts."""
    segments, _info = model.transcribe(
        audio_path,
        word_timestamps=True,
        vad_filter=True,
        language='en',
        beam_size=5,
    )
    words = []
    for seg in segments:
        for w in (seg.words or []):
            norm = normalize_word(w.word)
            if not norm:
                continue
            words.append({
                'word': w.word.strip(),
                'norm': norm,
                'start': float(w.start),
                'end': float(w.end),
            })
    return words


def best_fuzzy_match(target, candidates_by_norm, min_score=0.72):
    """Return (norm_key, candidate_list) for the closest spelling match, or None."""
    best = None
    best_score = min_score
    for norm_key in candidates_by_norm:
        score = SequenceMatcher(None, target, norm_key).ratio()
        if score > best_score:
            best_score = score
            best = norm_key
    if best is None:
        return None
    return best, candidates_by_norm[best]


def process(song_path, video_path, gap_mode, sensitivity, model_name='base'):
    try:
        upd(status='processing', message='Loading Whisper model...', percent=4)
        model = get_whisper(model_name)

        upd(message='Transcoding audio for transcription...', percent=8)
        song_wav = transcode_to_wav16(song_path)
        orig_wav = transcode_to_wav16(video_path)

        upd(message='Transcribing song (Whisper)...', percent=14)
        song_words = transcribe_words(model, song_wav)

        upd(message='Transcribing original video (Whisper)...', percent=40)
        orig_words = transcribe_words(model, orig_wav)

        os.unlink(song_wav)
        os.unlink(orig_wav)

        if not song_words:
            upd(status='error', message='Whisper found no words in the song.')
            return
        if not orig_words:
            upd(status='error', message='Whisper found no words in the original video.')
            return

        upd(message=f'Indexing {len(orig_words)} source words...', percent=68)

        # Index original words by normalized form
        orig_index = defaultdict(list)
        for ow in orig_words:
            orig_index[ow['norm']].append(ow)

        upd(message=f'Matching {len(song_words)} song words to source...', percent=72)

        # Get song duration from the audio file directly so we can fill trailing gap
        data, sr = sf.read(song_path, dtype='float32', always_2d=False)
        if data.ndim > 1:
            data = data[:, 0]
        song_duration = len(data) / sr

        # Pick best instance for each song word — prefer least-used so we
        # don't repeat the same source frame for repeated song words.
        used_count = defaultdict(int)
        matched = 0
        fuzzy = 0
        unmatched = 0

        timeline = []
        song_pos = 0.0

        for sw in song_words:
            # Fill any gap (silence/beats) preceding this word
            if sw['start'] > song_pos + 0.02:
                timeline.append({
                    'type': 'gap',
                    'duration': sw['start'] - song_pos,
                })

            song_dur = max(0.05, sw['end'] - sw['start'])
            norm = sw['norm']

            # 1. Exact match
            candidates = orig_index.get(norm)
            match_key = norm if candidates else None

            # 2. Fuzzy fallback
            is_fuzzy = False
            if not candidates:
                fuzzy_result = best_fuzzy_match(norm, orig_index)
                if fuzzy_result:
                    match_key, candidates = fuzzy_result
                    is_fuzzy = True

            if candidates:
                idx = used_count[match_key] % len(candidates)
                ow = candidates[idx]
                used_count[match_key] += 1
                timeline.append({
                    'type': 'cut',
                    'word': sw['word'],
                    'song_dur': song_dur,
                    'orig_time': ow['start'],
                    'orig_dur': max(0.05, ow['end'] - ow['start']),
                    'fuzzy': is_fuzzy,
                })
                if is_fuzzy:
                    fuzzy += 1
                else:
                    matched += 1
            else:
                timeline.append({
                    'type': 'unmatched',
                    'word': sw['word'],
                    'duration': song_dur,
                })
                unmatched += 1

            song_pos = sw['end']

        # Trailing gap (if song continues after last word)
        if song_pos < song_duration - 0.02:
            timeline.append({'type': 'gap', 'duration': song_duration - song_pos})

        upd(
            message=f'Cutting timeline: {matched} matches, {fuzzy} fuzzy, {unmatched} unmatched...',
            percent=78,
            segments=matched + fuzzy,
        )

        video_clip = VideoFileClip(video_path)
        fps = video_clip.fps
        video_duration = video_clip.duration
        black_frame = np.zeros((int(video_clip.h), int(video_clip.w), 3), dtype=np.uint8)
        last_frame = video_clip.get_frame(0)

        clips = []
        for item in timeline:
            if item['type'] == 'cut':
                target_dur = item['song_dur']
                o_start = max(0.0, min(item['orig_time'], video_duration - 0.05))
                o_end_raw = min(o_start + item['orig_dur'], video_duration)
                if o_end_raw - o_start < 0.04:
                    o_end_raw = min(o_start + 0.06, video_duration)
                source_dur = o_end_raw - o_start

                vc = video_clip.subclipped(o_start, o_end_raw)

                # Time-stretch: songs often speed up or slow down speech, so the
                # song-word duration ≠ source-word duration. We stretch the
                # source clip to match the song timing. speed = source/target;
                # speed > 1 plays faster (compressed), speed < 1 plays slower.
                if abs(source_dur - target_dur) > 0.015:
                    speed = source_dur / target_dur
                    speed = max(0.25, min(4.0, speed))
                    vc = vc.with_speed_scaled(speed)
                vc = vc.with_duration(target_dur)
                clips.append(vc)
                last_frame = video_clip.get_frame(min(o_end_raw - 0.01, video_duration - 0.05))
            else:
                dur = item.get('duration', 0)
                if dur < 0.02:
                    continue
                if gap_mode == 'freeze':
                    clips.append(ImageClip(last_frame.copy()).with_duration(dur).with_fps(fps))
                else:
                    clips.append(ImageClip(black_frame).with_duration(dur).with_fps(fps))

        if not clips:
            upd(status='error', message='No clips assembled.')
            video_clip.close()
            return

        upd(message=f'Rendering ({matched + fuzzy} word cuts)...', percent=90, segments=matched + fuzzy)

        final = concatenate_videoclips(clips, method='compose')
        audio_clip = AudioFileClip(song_path)
        safe_end = min(song_duration, final.duration, audio_clip.duration) - 0.001
        final = final.with_audio(audio_clip.subclipped(0, max(0, safe_end)))

        output_path = os.path.join(OUTPUT_DIR, 'music_video.mp4')
        final.write_videofile(
            output_path,
            fps=fps,
            codec='libx264',
            audio_codec='aac',
            preset='medium',
            ffmpeg_params=['-pix_fmt', 'yuv420p'],
            logger=None,
        )

        video_clip.close()
        final.close()

        upd(
            status='done',
            message=f'Done! {matched} exact + {fuzzy} fuzzy word cuts ({unmatched} unmatched).',
            percent=100,
            output=output_path,
        )

    except Exception as e:
        import traceback
        upd(status='error', message=str(e) + '\n\n' + traceback.format_exc())


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def start_process():
    with job_lock:
        if job['status'] == 'processing':
            return jsonify({'error': 'Already processing'}), 400

    song_file = request.files.get('song')
    video_file = request.files.get('video')
    gap_mode = request.form.get('gap_mode', 'black')
    sensitivity = float(request.form.get('sensitivity', '0.25'))
    model_name = request.form.get('model', 'base')

    if not song_file or not video_file:
        return jsonify({'error': 'Both song and video are required'}), 400

    song_path = os.path.join(UPLOAD_DIR, 'song' + os.path.splitext(song_file.filename)[1])
    video_path = os.path.join(UPLOAD_DIR, 'video' + os.path.splitext(video_file.filename)[1])
    song_file.save(song_path)
    video_file.save(video_path)

    upd(status='processing', message='Starting...', percent=0, output=None, segments=0)
    threading.Thread(
        target=process,
        args=(song_path, video_path, gap_mode, sensitivity, model_name),
        daemon=True,
    ).start()
    return jsonify({'ok': True})


@app.route('/status')
def status():
    with job_lock:
        return jsonify({k: job[k] for k in ('status', 'message', 'percent', 'segments')})


@app.route('/download')
def download():
    with job_lock:
        path = job.get('output')
    if not path or not os.path.exists(path):
        return 'No output file', 404
    return send_file(path, as_attachment=True, download_name='music_video.mp4')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7435))
    print(f'Video Chopper running at http://localhost:{port}')
    app.run(host='0.0.0.0', port=port, debug=False)
