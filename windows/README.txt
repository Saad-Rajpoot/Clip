==========================================================
  ClipStudio — Windows par chalane ka tareeqa
==========================================================

ClipStudio ka poora code Windows par chalta hai (koi alag/special build
nahi chahiye). Sirf yeh launcher chahiye tha — ab maujood hai.

----------------------------------------------------------
ZAROORAT (requirements)
----------------------------------------------------------
- Windows 10 / 11
- Python 3.10, 3.11 ya 3.12  ->  https://www.python.org/downloads/release/python-3129/
  (install karte waqt "Add Python to PATH" zaroor TICK karein)
  ZAROORI: Python 3.13 / 3.14 abhi NAHI chalega - ML libraries (onnxruntime,
  opencv, numpy, faster-whisper) ki Windows wheels sirf 3.10-3.12 ke liye hain.
  Agar 3.14 install hai to bhi 3.12 alag se install kar lein; launcher khud
  3.12 dhoond kar use kar lega.
- Internet (pehli baar dependencies + footage download ke liye)
- GPU optional: NVIDIA (NVENC) / Intel (QSV) / AMD (AMF) ho to encode tez,
  warna CPU (libx264) par bhi chalega — bas thoda slow.

----------------------------------------------------------
PEHLI BAAR SETUP (one time)
----------------------------------------------------------
0. PEHLE YEH CHECK KAREIN — laptop x86-64 hona chahiye:
       python -c "import platform;print(platform.machine())"
   -> "AMD64" aana chahiye. Agar "ARM64" aaye (Snapdragon X waghera) to yeh setup
      chalega hi nahi — ML packages (onnxruntime/ctranslate2/PyAV) ke ARM64 Windows
      wheels mojood nahi hain. Us surat mein x86-64 machine istemal karein.
   Saath hi: "Microsoft Visual C++ 2015-2022 x64 Redistributable" install karein
   (vc_redist.x64.exe) — iske baghair ML packages import par chup-chaap fail hote hain.

1. ClipStudio ka poora folder COPY karein (zip / USB / robocopy).
   ⚠ "git clone" se kaam NAHI chalega: saari bhaari files gitignored hain, aur unke
   baghair render shuru hote hi ruk jata hai. Neeche step 2 ki list dekhein.

2. YEH 4 CHEEZEIN GIT MEIN NAHI AATIN — haath se copy karna zaroori hai:

   a) CLIP models (~608 MB) — INKE BAGHAIR RENDER SHURU HI NAHI HOGA
      Mac par:  ~/.cache/vidlore_clip/
      Chahiye:  clip_vision.onnx, clip_text.onnx, tokenizer.json
      Windows par rakhein:  <folder>\models\clip\
      ⚠ Folder ka naam BILKUL "models\clip" ho — launcher yehi dhoondta hai.
        Mac wala naam ("models/vidlore_clip") rakhein to chup-chaap fail hoga.

   b) Music library (118 mp3) — INKE BAGHAIR BUILD ERROR DEGA
      Mac par:  vidlore/assets/music/  (poora folder, sub-folders samet)
      Windows par:  wahi jagah, ya VIDLORE_MUSIC_DIR us folder par set karein.
      Check:  python -c "import vidlore.musiclib as m;s=m.scan();print(len(s),sum(len(v) for v in s.values()))"
              -> "11 118" aana chahiye.

   c) Face-ID models (~39 MB) — inke baghair render chalega, magar GALAT CHARACTER
      wali footage reject nahi hogi (quality gir jayegi, aur log mein warning aayegi)
      Mac par:  ~/.cache/clipstudio_models/{yunet.onnx, sface.onnx}
      Windows par:  C:\Users\<aap>\.cache\clipstudio_models\

   d) ".env" (API keys) — git mein kabhi nahi jaati
      ⚠ ".env.example" se NAA banayein — us mein DEEPSEEK_API_KEY hi nahi hai,
        jo default brain hai. Mac ki asli .env copy karein.
      Zaroori: DEEPSEEK_API_KEY (brain) + GEMINI_API_KEY (footage verifier).

3. ffmpeg AUR ffprobe — dono chahiye:
   <folder>\ffmpeg\bin\ffmpeg.exe  +  <folder>\ffmpeg\bin\ffprobe.exe
   (asli ffprobe hona chahiye — ffmpeg ki copy ka naam badalna kaafi NAHI.
    Warna A/V-sync check har render ko aakhri step par abort karega.)
   NVIDIA GPU ho to:  python tools\fetch_windows_nvenc_ffmpeg.py --dest <folder>\ffmpeg

4. HD (720-1080p) DOWNLOADS — launcher YEH KHUD KAR LETA HAI (koi qadam nahi)
   Pehli baar chalane par run-windows.bat khud:
     - ".hdvenv" banata hai + yt-dlp / yt-dlp-ejs / bgutil-ytdlp-pot-provider install karta hai
     - Deno install karta hai (agar PATH par na ho) — sirf is user ke liye, admin nahi chahiye
     - bundle ka ".pot" server khud dhoond leta hai
   Node ki ZAROORAT NAHI: usay koi cheez run nahi karti (PO-token server Deno par chalta hai).
   Band karna ho to:  set VIDLORE_HD_SETUP=0
   Check:  python -c "from vidlore.clipstudio import hd_download as h;print(h.available())"
           -> True aana chahiye. False = sab footage 360p aayegi.

5. Yeh file double-click karein:  windows\Start-ClipStudio.bat
   (ya root mein:  run-windows.bat)
   - Pehli baar yeh khud .venv banata hai + saari dependencies install karta hai
     (thoda waqt lagega — sirf pehli baar).
   ⚠ SIRF launcher se chalayein. Seedha "python -m vidlore.clipstudio.web" chalane
     par Windows cp1252 encoding istemal karta hai aur crash hota hai — launcher
     PYTHONUTF8=1 set karta hai.

6. Pehla render CHHOTA rakhein (1-2 minute ka script), poora essay nahi — taake
   agar setup mein kuch reh gaya ho to 20 minute mein pata chal jaye.
   Pehle  windows-diagnose.bat  chala lein: woh batata hai kaun si key/binary mili.

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
- "ML deps install nahi huin" / pip error  ->  99% wajah Python 3.13/3.14.
    Python 3.12 install karein (upar link), phir launcher dobara chalayein.
    Launcher purana galat .venv khud hata kar 3.12 se naya bana dega.
- windows\Diagnose.bat  double-click  ->  "vidlore_diag.txt" banegi; wo bhej dein.
- GPU/encode check ke liye (optional) "tools" folder mein:
    check_windows_gpu_acceleration.bat   (GPU encode kaam kar raha hai?)
    fetch_windows_nvenc_ffmpeg.py        (NVENC wala ffmpeg dobara laana ho to)
- Browser khud na khule to manually kholein: http://127.0.0.1:5151

----------------------------------------------------------
WINDOWS-SPECIFIC: code mein kya change hua?
----------------------------------------------------------
- App ka code cross-platform pehle se hai (koi vidlore/*.py nahi badli):
    * Encoder khud chunta hai: Windows par NVENC/QSV/AMF, warna CPU libx264
      (vidlore/assemble.py: _pick_video_encoder). macOS par VideoToolbox.
    * ffmpeg "imageio-ffmpeg" se aata hai (pip dependency) — Windows par bhi.
    * ffprobe/venv ke liye ".exe" handling pehle se maujood.
- Windows setup ke liye yeh add hua:
    * run-windows.bat launcher + yeh "windows" folder.
    * requirements-clipstudio.txt — ClipStudio ka ML stack (yt-dlp, onnxruntime,
      opencv, scenedetect, rapidocr, av, google-genai). macOS par yeh ek alag
      .clipstudio_libs folder mein hota hai; Windows par launcher ise .venv mein
      install karta hai. (Isliye .clipstudio_libs ko Windows par copy NAHI karna.)
    * Launcher Python 3.10-3.12 enforce karta hai (3.13/3.14 par ML wheels nahi).
==========================================================
