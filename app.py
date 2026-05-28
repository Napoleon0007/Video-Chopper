import os
import re
import json
import random
import hashlib
import secrets
import threading
import tempfile
import subprocess
import time as _time
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
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024 * 1024  # 4 GB — B-roll batches can be huge

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
CACHE_DIR = os.path.join(BASE_DIR, 'cache')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


def file_hash(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        while True:
            buf = f.read(1 << 16)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def cached_transcribe(model, audio_path, model_name, chunked=False, source=False, progress_cb=None):
    """Transcribe with on-disk cache keyed by file content + model + mode."""
    h = file_hash(audio_path)
    mode = 'chunked' if chunked else ('source' if source else 'single')
    cache_path = os.path.join(CACHE_DIR, f'{h}_{model_name}_{mode}.json')
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    if chunked:
        words = transcribe_words_chunked(model, audio_path, progress_cb=progress_cb)
    else:
        words = transcribe_words(model, audio_path, source=source)
    with open(cache_path, 'w') as f:
        json.dump(words, f)
    return words

job = {'status': 'idle', 'message': 'Ready.', 'percent': 0, 'output': None, 'segments': 0}
job_lock = threading.Lock()

_whisper_model = None
_whisper_lock = threading.Lock()


_whisper_models = {}


def get_whisper(model_name='base'):
    """Lazy-load Whisper. Cached per model name for the process lifetime."""
    with _whisper_lock:
        m = _whisper_models.get(model_name)
        if m is None:
            m = WhisperModel(model_name, device='cpu', compute_type='int8')
            _whisper_models[model_name] = m
        return m


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


def estimate_bpm_and_beats(audio_path, min_bpm=70, max_bpm=180):
    """Onset-envelope based tempo + uniform beat grid.

    Returns (bpm, beat_times). Good enough for stable-tempo electronic / hip-hop
    music; less reliable on tempo-shifting tracks.
    """
    data, sr = sf.read(audio_path, dtype='float32', always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if len(data) < sr:
        return 120.0, []

    # Onset strength: positive derivative of log-energy in ~10ms frames
    hop_ms = 10
    hop = int(sr * hop_ms / 1000)
    frame = int(sr * 25 / 1000)
    n = max(0, (len(data) - frame) // hop)
    rms = np.zeros(n, dtype=np.float32)
    for i in range(n):
        s = i * hop
        rms[i] = np.sqrt(np.mean(data[s:s + frame] ** 2))
    log_rms = np.log1p(rms * 100.0)
    onset = np.maximum(np.diff(log_rms, prepend=log_rms[0]), 0)

    if onset.max() < 1e-6:
        return 120.0, []

    # Autocorrelate onset envelope to find dominant period (= beat period)
    frames_per_sec = 1000.0 / hop_ms
    min_lag = int(frames_per_sec * 60.0 / max_bpm)
    max_lag = int(frames_per_sec * 60.0 / min_bpm)

    onset_centered = onset - onset.mean()
    ac = np.correlate(onset_centered, onset_centered, mode='full')
    ac = ac[len(ac) // 2:]                # only positive lags
    ac = ac[:max_lag + 1]
    if len(ac) <= min_lag:
        return 120.0, []

    # Best lag in BPM range
    best_lag = int(np.argmax(ac[min_lag:max_lag + 1])) + min_lag
    bpm = 60.0 * frames_per_sec / best_lag

    # Anchor: first onset above the 80th percentile
    threshold = float(np.percentile(onset, 80))
    anchor_frame = 0
    for i, v in enumerate(onset):
        if v > threshold:
            anchor_frame = i
            break
    anchor_time = anchor_frame * hop_ms / 1000.0
    beat_period = 60.0 / bpm

    # Walk backward to the first beat at/after t=0, then forward to end
    duration = len(data) / sr
    while anchor_time - beat_period >= 0:
        anchor_time -= beat_period
    beats = []
    t = anchor_time
    while t < duration:
        if t >= 0:
            beats.append(round(t, 4))
        t += beat_period

    return round(bpm, 1), beats


# Song: VAD off, all bail-out thresholds disabled — chopped speech with
# repetitive phrases would otherwise trip the hallucination guard.
WHISPER_KWARGS = dict(
    word_timestamps=True,
    vad_filter=False,
    language='en',
    beam_size=1,
    no_speech_threshold=0.1,
    compression_ratio_threshold=100.0,
    log_prob_threshold=-10.0,
    condition_on_previous_text=False,
)

# Source video: VAD ON skips silences (much faster); keep defaults
# otherwise since the audio is clean continuous speech.
WHISPER_KWARGS_SOURCE = dict(
    word_timestamps=True,
    vad_filter=True,
    language='en',
    beam_size=1,
    condition_on_previous_text=False,
)


def transcribe_words(model, audio_path, source=False):
    """Run Whisper on a single file."""
    kwargs = WHISPER_KWARGS_SOURCE if source else WHISPER_KWARGS
    segments, _info = model.transcribe(audio_path, **kwargs)
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


def transcribe_words_chunked(model, audio_path, chunk_sec=30.0, overlap_sec=2.0, progress_cb=None):
    """Force full coverage by transcribing fixed-size chunks independently.

    Whisper's internal bail-out (hallucination guard, no_speech) sometimes
    abandons the tail of a file with repetitive content. Splitting into
    chunks guarantees every section gets attempted.

    progress_cb(chunk_idx, total_chunks) is called before each chunk so the
    job status can show progress instead of sitting frozen at the start
    percent for the full duration of the transcription.
    """
    import soundfile as sf
    data, sr = sf.read(audio_path, dtype='float32', always_2d=False)
    if data.ndim > 1:
        data = data[:, 0]
    total_sec = len(data) / sr

    # Estimate total chunk count
    step = max(0.1, chunk_sec - overlap_sec)
    total_chunks = max(1, int((total_sec + step - 0.05) // step) + 1)

    all_words = []
    pos = 0.0
    chunk_idx = 0
    while pos < total_sec - 0.05:
        if progress_cb:
            try:
                progress_cb(chunk_idx, total_chunks)
            except Exception:
                pass
        end = min(pos + chunk_sec, total_sec)
        s_idx = int(pos * sr)
        e_idx = int(end * sr)
        chunk = data[s_idx:e_idx]
        tmp = tempfile.mktemp(suffix=f'_chunk{chunk_idx}.wav')
        sf.write(tmp, chunk, sr)
        final_chunk = (end >= total_sec - 0.05)
        try:
            segments, _info = model.transcribe(tmp, **WHISPER_KWARGS)
            for seg in segments:
                for w in (seg.words or []):
                    norm = normalize_word(w.word)
                    if not norm:
                        continue
                    abs_start = float(w.start) + pos
                    abs_end = float(w.end) + pos
                    # Skip near-duplicates from the previous chunk's overlap
                    if all_words and abs(all_words[-1]['start'] - abs_start) < 0.15 \
                            and all_words[-1]['norm'] == norm:
                        continue
                    all_words.append({
                        'word': w.word.strip(),
                        'norm': norm,
                        'start': abs_start,
                        'end': abs_end,
                    })
        finally:
            os.unlink(tmp)
        chunk_idx += 1
        if final_chunk:
            break
        pos = end - overlap_sec

    # Keep sorted just in case
    all_words.sort(key=lambda w: w['start'])
    return all_words


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


def process(song_path, video_path, gap_mode, sensitivity, model_name='base',
            broll_paths=None, browser_song_words=None):
    broll_paths = broll_paths or []
    try:
        source_model_name = model_name  # default 'base'

        upd(message='Transcoding audio for BPM detection...', percent=6)
        song_wav = transcode_to_wav16(song_path)
        orig_wav = transcode_to_wav16(video_path)

        upd(message='Detecting BPM and beat grid...', percent=10)
        bpm, beat_times = estimate_bpm_and_beats(song_wav)

        if browser_song_words:
            # Browser already did the slow part — drop the words straight in,
            # normalising them so the matching code can use them.
            upd(message=f'Song transcribed in browser — {len(browser_song_words)} words.', percent=30)
            song_words = []
            for w in browser_song_words:
                norm = normalize_word(w.get('word', ''))
                if not norm:
                    continue
                song_words.append({
                    'word': str(w.get('word', '')).strip(),
                    'norm': norm,
                    'start': float(w.get('start', 0)),
                    'end': float(w.get('end', 0)),
                })
        else:
            # Fallback: transcribe server-side using tiny model
            song_model_name = 'tiny'
            upd(status='processing', message='Loading Whisper models...', percent=12)
            song_model = get_whisper(song_model_name)

            upd(message=f'Transcribing song ({bpm:.0f} BPM, tiny model)...', percent=14)

            def _song_progress(i, total):
                pct = 14 + int(24 * i / max(1, total))
                upd(message=f'Transcribing song chunk {i + 1}/{total}...', percent=pct)

            song_words = cached_transcribe(
                song_model, song_wav, song_model_name, chunked=True, progress_cb=_song_progress
            )

        source_model = get_whisper(source_model_name)
        upd(message='Transcribing original video...', percent=40)
        orig_words = cached_transcribe(source_model, orig_wav, source_model_name, chunked=False, source=True)

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

        # Smooth: merge consecutive cuts when the source positions are adjacent.
        # If "ANC government" was chopped intact from the original, both word-cuts
        # become one continuous 1s clip instead of two sub-second jump cuts.
        smoothed = []
        for item in timeline:
            if (
                smoothed
                and smoothed[-1].get('type') == 'cut'
                and item.get('type') == 'cut'
            ):
                prev = smoothed[-1]
                prev_orig_end = prev['orig_time'] + prev['orig_dur']
                orig_gap = item['orig_time'] - prev_orig_end
                if -0.1 <= orig_gap <= 0.25:
                    prev['song_dur'] += item['song_dur']
                    prev['orig_dur'] = item['orig_time'] + item['orig_dur'] - prev['orig_time']
                    continue
            smoothed.append(item)
        timeline = smoothed

        cut_count = sum(1 for x in timeline if x.get('type') == 'cut')
        upd(
            message=f'Cuts: {matched} matches, {fuzzy} fuzzy, {unmatched} unmatched → {cut_count} smoothed clips',
            percent=78,
            segments=cut_count,
        )

        video_clip = VideoFileClip(video_path)
        fps = video_clip.fps
        video_duration = video_clip.duration
        target_w = int(video_clip.w)
        target_h = int(video_clip.h)
        black_frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        last_frame = video_clip.get_frame(0)

        # Load B-roll clips once and reuse
        broll_clips = []
        for p in broll_paths:
            try:
                bc = VideoFileClip(p)
                if bc.duration and bc.duration > 0.2:
                    broll_clips.append(bc)
            except Exception:
                continue

        beat_period = (60.0 / bpm) if bpm else 0.5

        # song_pos tracks the running output time so we know which beat
        # boundaries land inside the current gap.
        song_pos = 0.0
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

                if abs(source_dur - target_dur) > 0.015:
                    speed = source_dur / target_dur
                    speed = max(0.25, min(4.0, speed))
                    vc = vc.with_speed_scaled(speed)
                vc = vc.with_duration(target_dur)
                clips.append(vc)
                last_frame = video_clip.get_frame(min(o_end_raw - 0.01, video_duration - 0.05))
                song_pos += target_dur
            else:
                dur = item.get('duration', 0)
                if dur < 0.02:
                    continue
                gap_start = song_pos
                gap_end = song_pos + dur

                # Cut on the beat: split the gap at every beat boundary so
                # whatever fills it (B-roll or source-as-broll) lands on the BPM.
                splits = [gap_start]
                for b in beat_times:
                    if gap_start + 0.08 < b < gap_end - 0.08:
                        splits.append(b)
                splits.append(gap_end)
                min_seg = max(0.12, beat_period * 0.5)
                cleaned = [splits[0]]
                for s in splits[1:]:
                    if s - cleaned[-1] >= min_seg:
                        cleaned.append(s)
                if len(cleaned) < 2 or cleaned[-1] < gap_end:
                    cleaned[-1] = gap_end

                for i in range(len(cleaned) - 1):
                    seg_dur = cleaned[i + 1] - cleaned[i]
                    if seg_dur < 0.05:
                        continue

                    if broll_clips:
                        # Random snippet from uploaded B-roll pool
                        broll = random.choice(broll_clips)
                        max_start = max(0.0, broll.duration - seg_dur - 0.05)
                        start = random.uniform(0.0, max_start) if max_start > 0.1 else 0.0
                        bc = broll.subclipped(start, start + seg_dur)
                        bc = bc.resized(new_size=(target_w, target_h))
                        bc = bc.without_audio().with_duration(seg_dur).with_fps(fps)
                        clips.append(bc)
                    elif gap_mode == 'source':
                        # Random snippet from the source video itself —
                        # always something to look at, audio stripped, on beat.
                        max_start = max(0.0, video_duration - seg_dur - 0.1)
                        start = random.uniform(0.0, max_start) if max_start > 0.1 else 0.0
                        sc = video_clip.subclipped(start, start + seg_dur).without_audio()
                        sc = sc.with_duration(seg_dur)
                        clips.append(sc)
                    elif gap_mode == 'freeze':
                        clips.append(ImageClip(last_frame.copy()).with_duration(seg_dur).with_fps(fps))
                    else:
                        clips.append(ImageClip(black_frame).with_duration(seg_dur).with_fps(fps))

                song_pos += dur

        if not clips:
            upd(status='error', message='No clips assembled.')
            video_clip.close()
            return

        upd(message=f'Compositing ({matched + fuzzy} word cuts)...', percent=88, segments=matched + fuzzy)

        final = concatenate_videoclips(clips, method='compose')
        audio_clip = AudioFileClip(song_path)
        safe_end = min(song_duration, final.duration, audio_clip.duration) - 0.001
        final = final.with_audio(audio_clip.subclipped(0, max(0, safe_end)))

        output_path = os.path.join(OUTPUT_DIR, 'music_video.mp4')

        # Render with progress callback so the bar moves during encoding.
        from proglog import ProgressBarLogger

        total_frames = max(1, int(final.duration * fps))

        class _RenderLogger(ProgressBarLogger):
            def __init__(self):
                super().__init__()
                self._last_pct = 88

            def bars_callback(self, bar, attr, value, old_value=None):
                if attr != 'index':
                    return
                if bar == 't':
                    pct = 88 + int(11 * value / total_frames)
                    if pct != self._last_pct:
                        self._last_pct = pct
                        upd(
                            message=f'Encoding frame {value}/{total_frames}...',
                            percent=min(99, pct),
                        )

        final.write_videofile(
            output_path,
            fps=fps,
            codec='libx264',
            audio_codec='aac',
            audio_bitrate='96k',
            preset='fast',
            ffmpeg_params=[
                '-pix_fmt', 'yuv420p',
                '-crf', '28',
                '-movflags', '+faststart',
                '-profile:v', 'high',
                '-level', '4.0',
            ],
            logger=_RenderLogger(),
        )

        video_clip.close()
        final.close()

        for bc in broll_clips:
            try:
                bc.close()
            except Exception:
                pass

        broll_note = f' + {len(broll_clips)} B-roll on beat ({bpm:.0f} BPM)' if broll_clips else ''
        upd(
            status='done',
            message=f'Done! {matched} exact + {fuzzy} fuzzy word cuts ({unmatched} unmatched){broll_note}.',
            percent=100,
            output=output_path,
        )

    except Exception as e:
        import traceback
        upd(status='error', message=str(e) + '\n\n' + traceback.format_exc())


@app.route('/')
def index():
    return render_template('index.html')


def _safe_upload_path(name):
    """Resolve a client-supplied filename to a real file inside UPLOAD_DIR."""
    if not name:
        return None
    base = os.path.basename(name)
    if not base or '..' in base or '/' in base or '\\' in base:
        return None
    p = os.path.join(UPLOAD_DIR, base)
    if os.path.exists(p):
        return p
    return None


@app.route('/upload', methods=['POST'])
def upload_one():
    """Single-file upload endpoint. Each large file gets its own short POST
    so Railway's proxy doesn't time out a 5-minute multi-file upload."""
    f = request.files.get('file')
    role = request.form.get('role', 'misc')
    if not f or not f.filename:
        return jsonify({'error': 'no file'}), 400
    ext = os.path.splitext(f.filename)[1].lower() or '.bin'
    name = f'{role}_{int(_time.time() * 1000)}_{secrets.token_hex(4)}{ext}'
    p = os.path.join(UPLOAD_DIR, name)
    f.save(p)
    return jsonify({'ok': True, 'name': name, 'size': os.path.getsize(p), 'role': role})


@app.route('/process', methods=['POST'])
def start_process():
    with job_lock:
        if job['status'] == 'processing':
            return jsonify({'error': 'Already processing'}), 400

    gap_mode = request.form.get('gap_mode', 'black')
    sensitivity = float(request.form.get('sensitivity', '0.25'))
    model_name = request.form.get('model', 'base')

    browser_song_words_raw = request.form.get('song_words')
    browser_song_words = None
    if browser_song_words_raw:
        try:
            browser_song_words = json.loads(browser_song_words_raw)
            if not isinstance(browser_song_words, list):
                browser_song_words = None
        except Exception:
            browser_song_words = None

    # New path: client uploaded files via /upload, now sends server-side names.
    song_path = _safe_upload_path(request.form.get('song_name'))
    video_path = _safe_upload_path(request.form.get('video_name'))

    broll_paths = []
    broll_names_json = request.form.get('broll_names')
    if broll_names_json:
        try:
            for n in json.loads(broll_names_json):
                p = _safe_upload_path(n)
                if p:
                    broll_paths.append(p)
        except Exception:
            pass

    # Legacy path: single multipart POST with file objects (still supported).
    if not song_path:
        sf = request.files.get('song')
        if sf:
            song_path = os.path.join(UPLOAD_DIR, 'song' + os.path.splitext(sf.filename)[1])
            sf.save(song_path)
    if not video_path:
        vf = request.files.get('video')
        if vf:
            video_path = os.path.join(UPLOAD_DIR, 'video' + os.path.splitext(vf.filename)[1])
            vf.save(video_path)
    for i, bf in enumerate(request.files.getlist('broll') or []):
        if not bf or not bf.filename:
            continue
        ext = os.path.splitext(bf.filename)[1] or '.mp4'
        p = os.path.join(UPLOAD_DIR, f'broll_legacy_{i}{ext}')
        bf.save(p)
        broll_paths.append(p)

    if not song_path or not video_path:
        return jsonify({'error': 'Both song and video are required'}), 400

    upd(status='processing', message='Starting...', percent=0, output=None, segments=0)
    threading.Thread(
        target=process,
        args=(song_path, video_path, gap_mode, sensitivity, model_name, broll_paths, browser_song_words),
        daemon=True,
    ).start()
    return jsonify({'ok': True})


@app.route('/status')
def status():
    with job_lock:
        return jsonify({k: job[k] for k in ('status', 'message', 'percent', 'segments')})


@app.route('/reset', methods=['POST', 'GET'])
def reset():
    """Force the job state back to idle. Use when a previous job hung and
    POST /process keeps returning 'Already processing'."""
    with job_lock:
        job.update({'status': 'idle', 'message': 'Reset.', 'percent': 0, 'output': None, 'segments': 0})
    return jsonify({'ok': True})


@app.route('/download')
def download():
    with job_lock:
        path = job.get('output')
    if not path or not os.path.exists(path):
        return 'No output file', 404
    return send_file(path, as_attachment=True, download_name='music_video.mp4')


def _warmup():
    """Load Whisper models in a background thread so the first request is fast."""
    try:
        get_whisper('tiny')
        print('[warmup] tiny model loaded.')
        get_whisper('base')
        print('[warmup] base model loaded.')
    except Exception as e:
        print(f'[warmup] failed: {e}')


threading.Thread(target=_warmup, daemon=True).start()


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7435))
    print(f'Video Chopper running at http://localhost:{port}')
    try:
        from waitress import serve
        # threads=8 lets uploads, status polling and processing run in parallel
        serve(app, host='0.0.0.0', port=port, threads=8, channel_timeout=3600, cleanup_interval=60)
    except ImportError:
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
