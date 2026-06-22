==========================================================
  ClipStudio — Windows par chalane ka tareeqa
==========================================================

ClipStudio ka poora code Windows par chalta hai (koi alag/special build
nahi chahiye). Sirf yeh launcher chahiye tha — ab maujood hai.

----------------------------------------------------------
ZAROORAT (requirements)
----------------------------------------------------------
- Windows 10 / 11
- Python 3.10 ya us se naya  ->  https://www.python.org/downloads/
  (install karte waqt "Add Python to PATH" zaroor TICK karein)
- Internet (pehli baar dependencies + footage download ke liye)
- GPU optional: NVIDIA (NVENC) / Intel (QSV) / AMD (AMF) ho to encode tez,
  warna CPU (libx264) par bhi chalega — bas thoda slow.

----------------------------------------------------------
PEHLI BAAR SETUP (one time)
----------------------------------------------------------
1. ClipStudio ka poora folder apne Windows PC par copy karein.
2. ".env" file folder ke root mein honi chahiye (ismein API keys: DEEPSEEK_API_KEY waghera).
   - Agar nahi hai: ".env.example" ko copy kar ke ".env" naam dein aur apni keys daalein.
   - (.env secret hai, git mein nahi jaati — isliye use khud copy karna padta hai.)
3. Yeh file double-click karein:  windows\Start-ClipStudio.bat
   (ya root mein:  run-windows.bat)
   - Pehli baar yeh khud .venv banata hai + saari dependencies install karta hai
     (thoda waqt lagega — sirf pehli baar).

----------------------------------------------------------
ROZ CHALANA (har baar)
----------------------------------------------------------
- windows\Start-ClipStudio.bat  double-click.
- Browser khud khul jayega:  http://127.0.0.1:5151
- Portal mein: apna SCRIPT paste karein + (optional) VOICEOVER .mp3 upload karein,
  brain = DeepSeek V4 Pro, phir Render.
- Yeh terminal/console window khula rehne dein (band karne se portal ruk jata hai).

----------------------------------------------------------
KUCH CHAL NA RAHA HO TO
----------------------------------------------------------
- windows\Diagnose.bat  double-click  ->  "vidlore_diag.txt" banegi; wo bhej dein.
- GPU/encode check ke liye (optional) "tools" folder mein:
    check_windows_gpu_acceleration.bat   (GPU encode kaam kar raha hai?)
    fetch_windows_nvenc_ffmpeg.py        (NVENC wala ffmpeg dobara laana ho to)
- Browser khud na khule to manually kholein: http://127.0.0.1:5151

----------------------------------------------------------
WINDOWS-SPECIFIC: code mein kya change hua?
----------------------------------------------------------
- KUCH NAHI badla — code pehle se cross-platform hai:
    * Encoder khud chunta hai: Windows par NVENC/QSV/AMF, warna CPU libx264
      (vidlore/assemble.py: _pick_video_encoder). macOS par VideoToolbox.
    * ffmpeg "imageio-ffmpeg" se aata hai (pip dependency) — Windows par bhi.
    * ffprobe/venv ke liye ".exe" handling pehle se maujood.
- Sirf yeh add hua: yeh Windows launcher (run-windows.bat) + yeh folder.
==========================================================
