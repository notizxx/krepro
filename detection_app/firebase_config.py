import firebase_admin
from firebase_admin import credentials, db

# Load Firebase credentials
cred = credentials.Certificate("detection_app/serviceAccountKey.json")  # Ganti dengan path yang benar
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://buahhh-8ca9f-default-rtdb.firebaseio.com'  # Ganti dengan database Firebase Anda
})

# Referensi ke database
ref = db.reference('fruit_prices')
