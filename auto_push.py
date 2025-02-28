import os
import datetime

# Konfigurasi
REPO_DIR = "C:/my_yolov8_app"  # Ubah ke path repo Git-mu
COMMIT_MESSAGE = f"Auto-commit: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

# Masuk ke direktori repository
os.chdir(REPO_DIR)

# Jalankan perintah Git
os.system("git add .")
os.system(f'git commit -m "{COMMIT_MESSAGE}"')
os.system("git push origin main")

print("✅ Auto-push selesai!")
