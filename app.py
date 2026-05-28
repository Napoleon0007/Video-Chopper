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
    """Decode any audio/video source to a float32 mono array at target SR."""
    tmp = tempfile.mktemp(suffix='.wav')
    subprocess.run(
        [FFMPEG, '-y', '-i', src_path, '-ar', str(sr), '-ac', '1', '-f', 'wav', tmp],
        capture_output=True, check=True
    )
    data, _ = sf.read(tmp, dtype='float32', always_2d=False)
    os.unlink(tmp)
    return data


def bandpass_voice(audio, sr):
    """300–3400 Hz bandpass — strips beats/music so correlation locks onto voice only."""
    nyq = sr / 2.0
    b, a = butter(4, [300 / nyq, min(3400 / nyq, 0.99)], btype='band')
    return filtfilt(b, a, audio).astype(np.float32)


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
        WINDOW_SEC = 0.5          # each window = exactly one output clip
        window_samples = int(WINDOW_SEC * SR)
        SILENCE_THRESH = 0.015    # raw RMS below this = silence/beat, no match attempt

        upd(status='processing', message='Loading song audio...', percent=5)
        song = load_audio_ffmpeg(song_path, SR)

        upd(message='Extracting original video audio...', percent=12)
        orig = load_audio_ffmpeg(video_path, SR)

        upd(message='Filtering to speech band (300–3400 Hz)...', percent=18)
        song_filt = bandpass_voice(song, SR)
        orig_filt = bandpass_voice(orig, SR)

        upd(message='Building fingerprint...', percent=22)

        song_rms = np.sqrt(np.mean(song_filt ** 2)) + 1e-8
        orig_rms = np.sqrt(np.mean(orig_filt ** 2)) + 1e-8
        song_n = (song_filt / song_rms).astype(np.float32)
        orig_n = (orig_filt / orig_rms).astype(np.float32)

        n_fft = 1
        while n_fft < len(orig_n) + window_samples:
            n_fft <<= 1

        orig_padded = np.zeros(n_fft, dtype=np.float32)
        orig_padded[:len(orig_n)] = orig_n
        orig_fft = np.fft.rfft(orig_padded)

        upd(message='Matching clips to original...', percent=32)

        # Non-overlapping windows: each window becomes exactly one output clip
        n_windows = len(song_n) // window_samples
        matches = []

        for i in range(n_windows):
            s = i * window_samples
            e = s + window_samples
            query = song_n[s:e]

            # Silence check on raw (unfiltered) audio
            raw_rms = float(np.sqrt(np.mean(song[s:e] ** 2)))

            if raw_rms < SILENCE_THRESH:
                matches.append({'song_time': s / SR, 'silent': True, 'score': 0.0, 'query': query})
            else:
                q_padded = np.zeros(n_fft, dtype=np.float32)
                q_padded[:len(query)] = query[::-1]
                q_fft = np.fft.rfft(q_padded)

                corr = np.fft.irfft(orig_fft * q_fft)[:len(orig_n)]
                best_idx = int(np.argmax(corr))
                score = float(corr[best_idx]) / window_samples
                orig_start_sample = max(0, best_idx - window_samples + 1)

                matches.append({
                    'song_time': s / SR,
                    'orig_time': orig_start_sample / SR,
                    'orig_idx': orig_start_sample,
                    'score': score,
                    'query': query,
                    'silent': False,
                })

            if i % 20 == 0:
                pct = 32 + int(40 * i / n_windows)
                upd(message=f'Matching... {i + 1}/{n_windows} clips', percent=pct)

        upd(message='Setting threshold...', percent=74)

        voice_scores = [m['score'] for m in matches if not m['silent']]
        if voice_scores:
            score_arr = np.array(voice_scores)
            score_median = float(np.median(score_arr))
            score_max = float(score_arr.max())
            threshold = score_median + sensitivity * (score_max - score_median)
        else:
            threshold = 0.0

        upd(message='Detecting clip speeds...', percent=76)
        for m in matches:
            if not m['silent'] and m['score'] >= threshold:
                m['speed'] = detect_speed(m['query'], orig_n, m['orig_idx'], window_samples, SR)
            else:
                m['speed'] = 1.0

        upd(message='Building clip sequence...', percent=80)

        video_clip = VideoFileClip(video_path)
        fps = video_clip.fps
        video_duration = video_clip.duration

        clips = []
        matched_count = 0
        black_frame = np.zeros((int(video_clip.h), int(video_clip.w), 3), dtype=np.uint8)
        last_frame = video_clip.get_frame(0)

        for m in matches:
            is_match = not m['silent'] and m['score'] >= threshold

            if not is_match:
                if gap_mode == 'freeze':
                    clips.append(ImageClip(last_frame.copy()).with_duration(WINDOW_SEC).with_fps(fps))
                else:
                    clips.append(ImageClip(black_frame).with_duration(WINDOW_SEC).with_fps(fps))
            else:
                speed = m.get('speed', 1.0)
                raw_dur = WINDOW_SEC * speed
                o_start = max(0.0, min(m['orig_time'], video_duration - 0.1))
                o_end = max(o_start + 0.05, min(o_start + raw_dur, video_duration))

                vc = video_clip.subclipped(o_start, o_end)
                if abs(speed - 1.0) > 0.03:
                    vc = vc.with_speed_scaled(speed)
                vc = vc.with_duration(WINDOW_SEC)
                clips.append(vc)

                last_frame = video_clip.get_frame(min(o_end, video_duration - 0.05))
                matched_count += 1

        if not clips:
            upd(status='error', message='No clips assembled. Try lowering sensitivity.')
            video_clip.close()
            return

        upd(
            message=f'Rendering {len(clips)} clips ({matched_count} matched)...',
            percent=90,
            segments=matched_count,
        )

        final = concatenate_videoclips(clips, method='compose')
        audio_clip = AudioFileClip(song_path)
        song_dur = len(song) / SR
        safe_end = min(song_dur, final.duration, audio_clip.duration) - 0.001
        final = final.with_audio(audio_clip.subclipped(0, max(0, safe_end)))

        output_path = os.path.join(OUTPUT_DIR, 'music_video.mp4')
        final.write_videofile(output_path, fps=fps, codec='libx264', audio_codec='aac', logger=None)

        video_clip.close()
        final.close()

        upd(status='done', message=f'Done! {matched_count} voice clips cut and synced.', percent=100, output=output_path)

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
