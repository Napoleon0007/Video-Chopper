import os
import threading
import tempfile
import subprocess
import numpy as np
import soundfile as sf
from scipy.signal import resample, butter, filtfilt
import imageio_ffmpeg
from flask import Flask, render_template, request, jsonify, send_file

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
os.environ['IMAGEIO_FFMPEG_EXE'] = FFMPEG

from moviepy import VideoFileClip, AudioFileClip, ImageClip, concatenate_videoclips

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 600 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

job = {'status': 'idle', 'message': 'Ready.', 'percent': 0, 'output': None, 'segments': 0}
job_lock = threading.Lock()


def upd(**kwargs):
    with job_lock:
        job.update(kwargs)


def load_audio_ffmpeg(src_path, sr=8000):
    tmp = tempfile.mktemp(suffix='.wav')
    subprocess.run(
        [FFMPEG, '-y', '-i', src_path, '-ar', str(sr), '-ac', '1', '-f', 'wav', tmp],
        capture_output=True, check=True
    )
    data, file_sr = sf.read(tmp, dtype='float32', always_2d=False)
    os.unlink(tmp)
    return data, file_sr


def speech_filter(data, sr):
    """Bandpass 300–3400 Hz to isolate voice; cuts music and background noise."""
    lo = 300 / (sr / 2)
    hi = 3400 / (sr / 2)
    b, a = butter(4, [lo, hi], btype='band')
    return filtfilt(b, a, data).astype(np.float32)


SPEED_RATES = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30]


def detect_speed(query, orig_n, rough_orig_idx, window_samples, SR):
    best_rate = 1.0
    best_score = -1.0
    search_start = max(0, rough_orig_idx - window_samples)
    search_end = min(len(orig_n), rough_orig_idx + 2 * window_samples)
    local = orig_n[search_start:search_end]
    for rate in SPEED_RATES:
        target_len = max(8, int(len(query) / rate))
        q_resampled = resample(query, target_len).astype(np.float32)
        if len(q_resampled) > len(local):
            continue
        corr = np.correlate(local, q_resampled, mode='valid')
        if len(corr) == 0:
            continue
        sc = float(corr.max()) / len(q_resampled)
        if sc > best_score:
            best_score = sc
            best_rate = rate
    return best_rate


def process(song_path, video_path, gap_mode, sensitivity):
    try:
        SR = 8000
        # Smaller windows = word-level precision; hop = output clip duration
        WINDOW_SEC = 0.8
        HOP_SEC = 0.5
        window_samples = int(WINDOW_SEC * SR)
        hop_samples = int(HOP_SEC * SR)
        SILENCE_RMS = 0.015  # windows below this energy are treated as silence

        upd(status='processing', message='Loading song audio...', percent=5)
        song_raw, _ = load_audio_ffmpeg(song_path, SR)

        upd(message='Extracting original video audio...', percent=12)
        orig_raw, _ = load_audio_ffmpeg(video_path, SR)

        upd(message='Filtering to speech frequencies...', percent=20)
        # Bandpass to voice range so correlation ignores music/background
        song_f = speech_filter(song_raw, SR)
        orig_f = speech_filter(orig_raw, SR)

        song_rms = np.sqrt(np.mean(song_f ** 2)) + 1e-8
        orig_rms = np.sqrt(np.mean(orig_f ** 2)) + 1e-8
        song_n = (song_f / song_rms).astype(np.float32)
        orig_n = (orig_f / orig_rms).astype(np.float32)

        upd(message='Building audio fingerprint...', percent=26)

        n_fft = 1
        while n_fft < len(orig_n) + window_samples:
            n_fft <<= 1

        orig_padded = np.zeros(n_fft, dtype=np.float32)
        orig_padded[:len(orig_n)] = orig_n
        orig_fft = np.fft.rfft(orig_padded)

        upd(message='Scanning every window of the song...', percent=30)

        n_windows = max(1, (len(song_n) - window_samples) // hop_samples)
        matches = []

        for i in range(n_windows):
            s = i * hop_samples
            e = s + window_samples
            if e > len(song_n):
                break

            query = song_n[s:e]

            # Skip silence — no voice to match
            query_rms = float(np.sqrt(np.mean(query ** 2)))
            if query_rms < SILENCE_RMS:
                matches.append({
                    'song_time': s / SR,
                    'orig_time': 0.0,
                    'orig_idx': 0,
                    'score': 0.0,
                    'query': query,
                    'silent': True,
                })
                continue

            q_padded = np.zeros(n_fft, dtype=np.float32)
            q_padded[:len(query)] = query[::-1]
            q_fft = np.fft.rfft(q_padded)

            corr = np.fft.irfft(orig_fft * q_fft)[:len(orig_n)]
            best_idx = int(np.argmax(corr))
            score = float(corr[best_idx]) / window_samples
            # corr[k] = dot(query, orig[k-ws+1:k+1]), so orig_start = k-ws+1
            orig_start_sample = max(0, best_idx - window_samples + 1)

            matches.append({
                'song_time': s / SR,
                'orig_time': orig_start_sample / SR,
                'orig_idx': orig_start_sample,
                'score': score,
                'query': query,
                'silent': False,
            })

            if i % 10 == 0:
                pct = 30 + int(44 * i / n_windows)
                upd(message=f'Scanning... {i + 1}/{n_windows} windows', percent=pct)

        upd(message='Calibrating match threshold...', percent=76)

        voiced = [m['score'] for m in matches if not m.get('silent')]
        score_arr = np.array(voiced) if voiced else np.array([0.0])
        # Use median as noise floor — more matches get through vs 75th percentile
        noise_floor = float(np.percentile(score_arr, 50))
        score_max = float(score_arr.max())
        threshold = noise_floor + sensitivity * (score_max - noise_floor)

        upd(message='Detecting playback speeds...', percent=79)
        for m in matches:
            if not m.get('silent') and m['score'] >= threshold:
                m['speed'] = detect_speed(m['query'], orig_n, m['orig_idx'], window_samples, SR)
            else:
                m['speed'] = 1.0

        matched_count = sum(1 for m in matches if not m.get('silent') and m['score'] >= threshold)
        upd(
            message=f'Building {matched_count} individual clips (no segment grouping)...',
            percent=83,
            segments=matched_count,
        )

        video_clip = VideoFileClip(video_path)
        fps = video_clip.fps
        video_duration = video_clip.duration
        song_dur = len(song_raw) / SR

        clips = []
        last_frame = video_clip.get_frame(0)

        # Every window → its own clip. No grouping. Reordered audio works naturally.
        for m in matches:
            clip_dur = HOP_SEC

            is_match = not m.get('silent') and m['score'] >= threshold
            if is_match:
                speed = m.get('speed', 1.0)
                raw_dur = clip_dur * speed
                o_start = max(0.0, min(m['orig_time'], video_duration - 0.1))
                o_end = max(o_start + 0.05, min(o_start + raw_dur, video_duration))

                vc = video_clip.subclipped(o_start, o_end)
                if abs(speed - 1.0) > 0.03:
                    vc = vc.with_speed_scaled(speed)
                vc = vc.with_duration(clip_dur)
                clips.append(vc)
                last_frame = video_clip.get_frame(min(o_end, video_duration - 0.05))
            else:
                if gap_mode == 'freeze':
                    clips.append(ImageClip(last_frame).with_duration(clip_dur).with_fps(fps))
                else:
                    clips.append(ImageClip(np.zeros_like(last_frame)).with_duration(clip_dur).with_fps(fps))

        # Tail — cover any remaining song duration
        assembled_dur = len(clips) * HOP_SEC
        if assembled_dur < song_dur - 0.05:
            remaining = song_dur - assembled_dur
            if gap_mode == 'freeze':
                clips.append(ImageClip(last_frame).with_duration(remaining).with_fps(fps))
            else:
                clips.append(ImageClip(np.zeros_like(last_frame)).with_duration(remaining).with_fps(fps))

        if not clips:
            upd(status='error', message='No voice segments matched. Try lowering the sensitivity slider.')
            video_clip.close()
            return

        upd(message=f'Rendering ({matched_count} synced clips)...', percent=88, segments=matched_count)

        final = concatenate_videoclips(clips, method='compose')
        audio_clip = AudioFileClip(song_path)
        safe_end = min(song_dur, final.duration, audio_clip.duration) - 0.001
        song_audio = audio_clip.subclipped(0, max(0, safe_end))
        final = final.with_audio(song_audio)

        output_path = os.path.join(OUTPUT_DIR, 'music_video.mp4')
        final.write_videofile(output_path, fps=fps, codec='libx264', audio_codec='aac', logger=None)

        video_clip.close()
        final.close()

        upd(status='done', message=f'Done! {matched_count} clips synced.', percent=100, output=output_path)

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
    gap_mode = request.form.get('gap_mode', 'freeze')
    sensitivity = float(request.form.get('sensitivity', '0.25'))

    if not song_file or not video_file:
        return jsonify({'error': 'Both song and video are required'}), 400

    song_path = os.path.join(UPLOAD_DIR, 'song' + os.path.splitext(song_file.filename)[1])
    video_path = os.path.join(UPLOAD_DIR, 'video' + os.path.splitext(video_file.filename)[1])
    song_file.save(song_path)
    video_file.save(video_path)

    upd(status='processing', message='Starting...', percent=0, output=None, segments=0)
    threading.Thread(target=process, args=(song_path, video_path, gap_mode, sensitivity), daemon=True).start()
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
